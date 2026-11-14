#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import itertools
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np
import torch
from jaxtyping import Bool, Int
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset, Sampler, default_convert

from amets.common.typing import BatchedSample, Sample


@dataclass
class Collate:
    """
    A Callable abstract class for PyTorch DataLoader's collate_fn argument.
    """

    max_length: Optional[int]
    seq_fields: tuple[str, ...]
    pad_func_map: dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]] = field(
        default_factory=dict
    )
    target_field: str = "target"

    def __post_init__(self):
        self.pad_func_map = defaultdict(self._default_pad_func) | self.pad_func_map

    @staticmethod
    def _default_pad_func() -> Callable[[Sequence[int], np.dtype], np.ndarray]:
        return np.zeros

    def __call__(self, batch: list[Sample]) -> BatchedSample:
        raise NotImplementedError


# Fields whose length differs from the target. The length check skips them.
VARIABLE_LENGTH_FIELDS = {"context_target", "regime_probs"}


class PackCollate(Collate):
    """Packs uneven sequences with the first fit decreasing bin packing strategy."""

    def __call__(self, batch: list[Sample]) -> BatchedSample:
        # Check length consistency, skipping variable-length fields and dicts.
        assert all([
            len(sample[self.target_field]) == len(sample[key])
            for sample in batch
            for key in self.seq_fields
            if key not in VARIABLE_LENGTH_FIELDS
            and not isinstance(sample.get(key), dict)
        ]), "All fields must have the same length."


        assert all(
            [len(sample[self.target_field]) <= self.max_length for sample in batch]
        ), f"Sample length must be less than or equal to max_length ({self.max_length})"

        packed_batch, bin_spaces = self.first_fit_decreasing_bin_packing(batch)
        sample_id = self.get_sample_id(packed_batch, bin_spaces)
        
        merged_batch = self.merge_batch(packed_batch, bin_spaces) | dict(
            sample_id=sample_id
        )
        return merged_batch

    def first_fit_decreasing_bin_packing(
        self,
        batch: list[Sample],
    ) -> tuple[list[list[Sample]], Int[np.ndarray, "batch"]]:
        """
        Implements the first fit decreasing bin packing strategy.
        1. Sort the batch by sequence length, long to short.
        2. Initialize an empty list of bins, where each bin has a maximum size of max_length.
        3. Iterate through the sorted batch, inserting samples into the first bin which,
        when the sample is added, does not exceed max_length.

        Args:
            batch: List of samples.

        Returns:
            A tuple of (packed_batch, bin_spaces), where packed_batch is the
            batch after packing and bin_spaces is the length of each bin.
        """
        batch = sorted(
            batch, key=lambda sample: len(sample[self.target_field]), reverse=True
        )
        bin_spaces: Int[np.ndarray, "batch"] = np.full(len(batch), self.max_length)
        packed_batch: list[list[Sample]] = [[]]

        for sample in batch:
            length = len(sample[self.target_field])
            criterion: Bool[np.ndarray, "batch"] = bin_spaces - length >= 0
            bin_id: int = criterion.argmax()
            if len(packed_batch) <= bin_id:
                if len(packed_batch) != bin_id:
                    raise ValueError
                packed_batch.append([])

            packed_batch[bin_id].append(sample)
            bin_spaces[bin_id] -= length

        return packed_batch, bin_spaces[: len(packed_batch)]

    def get_sample_id(
        self, batch: list[list[Sample]], bin_spaces: Int[np.ndarray, "batch"]
    ) -> Int[torch.Tensor, "batch seq"]:
        """
        Create an array of integers representing the sample id in a sequence.
        Sample id starts from 1, and 0 represents padding.

        Args:
            batch: Packed samples.
            bin_spaces: Length of each bin.

        Returns:
            Integer array indicating the sample index within a sequence.
        """
        sample_id = torch.stack(
            [
                torch.cat(
                    [
                        torch.ones(len(sample[self.target_field])) * (idx + 1)
                        for idx, sample in enumerate(bin_)
                    ]
                    + [torch.zeros(space)],  # padding
                )
                for bin_, space in zip(batch, bin_spaces)
            ]
        ).to(torch.long)
        return sample_id

    def merge_batch(
        self, batch: list[list[Sample]], bin_spaces: Int[np.ndarray, "batch"]
    ) -> BatchedSample:
        """Combines packed samples into BatchedSample format."""
        # Fields that are not packed like the other seq_fields.
        special_fields = {"context_target", "regime_probs"}
        
        # Pack the regular seq_fields.
        merged = {
            key: torch.stack(
                [
                    torch.cat(
                        [default_convert(sample[key]) for sample in bin_]
                        + [
                            default_convert(
                                self.pad_func_map[key](
                                    (space,) + bin_[0][key].shape[1:],
                                    bin_[0][key].dtype,
                                )
                            )
                        ]
                    )
                    for bin_, space in zip(batch, bin_spaces)
                ],
            )
            for key in self.seq_fields
            if key not in special_fields
        }
        
        # Collect context_target separately as a list of dicts, one per sample.
        # Each dict maps variate_id to an array. forward() uses them for
        # per-variate regime prediction on GPU.
        context_target_list = []
        for bin_ in batch:
            for sample in bin_:
                ctx = sample["context_target"]
                context_target_list.append(ctx)
        merged["context_target_list"] = context_target_list
        return merged


# Fields that are not tensors. Slicing and concatenation handle them separately.
NON_TENSOR_FIELDS = {"context_target_list"}


@dataclass
class SliceableBatchedSample:
    """A BatchedSample that can be sliced."""

    data: BatchedSample

    def __post_init__(self):
        # Validate lengths over tensor fields only.
        tensor_keys = [k for k in self.data.keys() if k not in NON_TENSOR_FIELDS]
        if tensor_keys:
            first_key = tensor_keys[0]
            assert all(
                [
                    len(self.data[key]) == len(self.data[first_key])
                    for key in tensor_keys
                ]
            )

    def __len__(self) -> int:
        # Take the length from the first tensor field.
        for key in self.data.keys():
            if key not in NON_TENSOR_FIELDS:
                return len(self.data[key])
        return 0

    def __getitem__(self, item: slice) -> "SliceableBatchedSample":
        # Slice tensor fields, then slice context_target_list to match.
        sliced_data = {}
        
        # Slice all tensor fields.
        for key in self.data.keys():
            if key not in NON_TENSOR_FIELDS:
                sliced_data[key] = self.data[key][item]
        
        # Slice context_target_list to the samples in the sliced bins.
        if "context_target_list" in self.data and "sample_id" in sliced_data:
            # Find which samples are in the sliced portion.
            sample_id_sliced = sliced_data["sample_id"]  # (sliced_batch_size, seq_len)
            
            # Count samples per bin to find the sample index range.
            batch_size = sample_id_sliced.shape[0]
            samples_per_bin = [sample_id_sliced[b].max().item() for b in range(batch_size)]
            
            # Count samples in the sliced portion.
            total_samples = sum(samples_per_bin)
            
            # The start index in context_target_list is the number of samples in
            # the bins before the slice.
            if item.start is not None and item.start > 0 and "sample_id" in self.data:
                # Count samples in bins before the slice.
                original_sample_id = self.data["sample_id"]
                samples_before = sum([original_sample_id[b].max().item() for b in range(item.start)])
            else:
                samples_before = 0
            
            start_idx = samples_before
            end_idx = samples_before + total_samples
            sliced_data["context_target_list"] = self.data["context_target_list"][start_idx:end_idx]
        elif "context_target_list" in self.data:
            # Without sample_id, pass the list through unchanged.
            sliced_data["context_target_list"] = self.data["context_target_list"]
        
        return SliceableBatchedSample(sliced_data)


class Metadata(NamedTuple):
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass
class BatchedSampleQueue:
    """
    Queue data structure storing batched samples.

    Args:
        container: Internal queue data structure storing batched samples.
        schema: Format specification for batched samples.
    """

    container: deque[SliceableBatchedSample] = field(default_factory=deque)
    schema: Optional[dict[str, Metadata]] = None

    def _check_schema(self, batch: SliceableBatchedSample):
        """
        Ensure that all samples in the batch follows the required schema.
        If a schema has not been specified, then all samples in the batch should have the same schema.
        """
        if self.schema is None:
            self.schema = {
                key: Metadata(
                    shape=tuple(batch.data[key].shape[1:]), dtype=batch.data[key].dtype
                )
                for key in batch.data.keys()
                if key not in NON_TENSOR_FIELDS  # Skip non-tensor fields
            }
        else:
            mismatches = []
            for key, metadata in self.schema.items():
                if key in NON_TENSOR_FIELDS:
                    continue
                if key not in batch.data:
                    mismatches.append(f"{key}: missing from batch")
                    continue
                actual_shape = tuple(batch.data[key].shape[1:])
                actual_dtype = batch.data[key].dtype
                if metadata.shape != actual_shape:
                    mismatches.append(f"{key}: shape expected {metadata.shape}, got {actual_shape}")
                if metadata.dtype != actual_dtype:
                    mismatches.append(f"{key}: dtype expected {metadata.dtype}, got {actual_dtype}")
            # Check for fields in the batch that are not in the schema.
            for key in batch.data.keys():
                if key in NON_TENSOR_FIELDS:
                    continue
                if key not in self.schema:
                    mismatches.append(f"{key}: extra field not in schema")
            assert not mismatches, f"batch must have the same schema as the first batch. Mismatches: {mismatches}"

    def append(self, batch: SliceableBatchedSample | BatchedSample):
        """Appends a batch to the end of the queue."""
        if not isinstance(batch, SliceableBatchedSample):
            batch = SliceableBatchedSample(batch)
        self._check_schema(batch)
        self.container.append(batch)

    def appendleft(self, batch: SliceableBatchedSample | BatchedSample):
        """Appends a batch to the start of the queue."""
        if not isinstance(batch, SliceableBatchedSample):
            batch = SliceableBatchedSample(batch)
        self._check_schema(batch)
        self.container.appendleft(batch)

    def popleft(self, size: int) -> BatchedSample:
        """Pops a batch from the start of the queue."""
        if size > len(self):
            raise ValueError(
                f"pop size ({size}) must be less than or equal to queue size ({len(self)})"
            )

        out = BatchedSampleQueue()
        while len(out) < size:
            curr = self.container.popleft()
            if len(out) + len(curr) > size:
                self.appendleft(curr[size - len(out) :])
                curr = curr[: size - len(out)]
            out.append(curr)
        return out.as_batched_data()

    def as_batched_data(self) -> BatchedSample:
        """Returns the queue as a BatchedSample."""
        result = {
            key: torch.cat([batch.data[key] for batch in self.container], dim=0)
            for key in self.schema.keys()
        }
        # Concatenate the lists of non-tensor fields.
        for batch in self.container:
            for key in NON_TENSOR_FIELDS:
                if key in batch.data:
                    if key not in result:
                        result[key] = []
                    result[key].extend(batch.data[key])
        return result

    def __len__(self) -> int:
        """Total number of samples in the queue."""
        return sum(len(batch) for batch in self.container)


@dataclass
class _BatchedSampleIterator:
    """
    Iterator returning batched samples with a fixed batch size.

    Args:
        dataloader_iter: Iterator returning batched samples, which may not have a
            fixed batch size.
        batch_size: The batch size of batched samples to return.
        drop_last: Whether to drop the last batch if it has fewer than
            batch_size samples.
        fill_last: Whether to pad the last batch if it has fewer than
            batch_size samples.
        pad_func_map: Mapping to padding functions.
    """

    dataloader_iter: Iterator[BatchedSample]
    batch_size: int
    drop_last: bool
    fill_last: bool
    pad_func_map: dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]]

    def __post_init__(self):
        self.queue = BatchedSampleQueue()

    def __iter__(self):
        return self

    def __next__(self) -> BatchedSample:
        while (data := self._next_batch()) is None:
            continue
        return data

    def _next_batch(self) -> Optional[BatchedSample]:
        """Return the next batch of samples.

        Returns:
            The next batch, or None if more data must be pulled first.

        Raises:
            StopIteration: If the queue is empty.
        """
        if len(self.queue) < self.batch_size:
            # If the queue holds too few samples, pull the next batch from
            # dataloader_iter and return None.
            try:
                data = next(self.dataloader_iter)
                self.queue.append(data)
                return None
            except StopIteration:
                # dataloader_iter is exhausted. Apply drop_last or fill_last.
                if self.drop_last or len(self.queue) == 0:
                    raise StopIteration
                elif self.fill_last:
                    self._pad_queue(self.batch_size - len(self.queue))

        batch = self.queue.popleft(min(self.batch_size, len(self.queue)))
        return batch

    def _pad_queue(self, size: int):
        if self.queue.schema is None:
            raise ValueError("schema must be set before padding")
        padding = {
            key: default_convert(
                self.pad_func_map[key]((size,) + metadata.shape, np.dtype(np.float32))
            ).to(metadata.dtype)
            for key, metadata in self.queue.schema.items()
        }
        self.queue.append(padding)

    def has_next(self) -> bool:
        """Check if iterator still has next."""
        if len(self.queue) < self.batch_size:
            try:
                next_batch = next(self)
                self.queue.appendleft(next_batch)
            except StopIteration:
                return False
        return True


class DataLoader:
    """
    Wrapper on PyTorch's DataLoader class implementing:
    - packing
    - number of batches per epoch
    - cycle
    """

    pad_func_map: dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]] = {
        "target": np.zeros,
        "observed_mask": np.zeros,
        "time_id": np.zeros,
        "variate_id": np.zeros,
        "prediction_mask": np.zeros,
        "patch_size": np.zeros,
        "sample_id": np.zeros,
    }

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        batch_size_factor: float = 1.0,
        cycle: bool = False,
        num_batches_per_epoch: Optional[int] = None,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Collate] = None,
        pin_memory: bool = False,
        drop_last: bool = True,
        fill_last: bool = False,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        """
        Only wrapper specific arguments are documented below.
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        for documentation on PyTorch DataLoader arguments.

        Args:
            batch_size_factor: Multiply the batch_size given to PyTorch's DataLoader.
            cycle: Whether to cycle the dataloader infinitely.
            num_batches_per_epoch: Number of batches per epoch.
            fill_last: Whether to pad the last batch.
        """
        if num_batches_per_epoch is not None:
            assert cycle, "can only set 'num_batches_per_epoch' when 'cycle=True'"

        self.dataloader = TorchDataLoader(
            dataset=dataset,
            batch_size=int(batch_size * batch_size_factor),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and num_workers > 0,
        )
        self.batch_size = batch_size
        self.cycle = cycle
        self.num_batches_per_epoch = num_batches_per_epoch
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.fill_last = fill_last
        self.iterator: Optional[_BatchedSampleIterator] = None

    def __iter__(self) -> Iterator:
        if self.iterator is None or not self.iterator.has_next():
            dataloader_iter = (
                iter(self.dataloader)
                if not self.cycle
                else itertools.chain.from_iterable(itertools.repeat(self.dataloader))
            )
            self.iterator = _BatchedSampleIterator(
                dataloader_iter=dataloader_iter,
                batch_size=self.batch_size,
                drop_last=self.drop_last,
                fill_last=self.fill_last,
                pad_func_map=(  # If `collate_fn` is None, use `self.pad_func_map`.
                    self.pad_func_map
                    if self.collate_fn is None
                    else self.collate_fn.pad_func_map
                ),
            )
        return itertools.islice(self.iterator, self.num_batches_per_epoch)

    @property
    def worker_init_fn(self) -> Optional[Callable[[int], None]]:
        """Getter for worker_init_fn."""
        return self.dataloader.worker_init_fn

    @worker_init_fn.setter
    def worker_init_fn(self, worker_init_fn: Optional[Callable[[int], None]]):
        """Setter for worker_init_fn."""
        self.dataloader.worker_init_fn = worker_init_fn
