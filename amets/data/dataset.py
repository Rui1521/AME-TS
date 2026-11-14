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


import logging
from enum import Enum
from typing import Any, Optional

import numpy as np
from torch.utils.data import Dataset

from amets.common.sampler import Sampler, get_sampler
from amets.common.typing import (
    BatchedData,
    BatchedDateTime,
    BatchedString,
    Data,
    FlattenedData,
    MultivarTimeSeries,
    UnivarTimeSeries,
)
from amets.data.indexer import Indexer
from amets.transform import Transformation
from amets.transform._base import Chain
from amets.transform.patch import GetPatchSize

logger = logging.getLogger(__name__)


class SampleTimeSeriesType(Enum):
    """
    How to sample from the dataset.

    - none: Do not sample. Return the series at the current index.
    - uniform: Sample each time series with equal probability.
    - proportional: Sample each time series with probability proportional to its length.
    """

    NONE = "none"
    UNIFORM = "uniform"
    PROPORTIONAL = "proportional"


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        indexer: Indexer[dict[str, Any]],
        transform: Transformation,
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE,
        dataset_weight: float = 1.0,
        name: Optional[str] = None,
    ):
        """
        Args:
            indexer: Underlying Indexer object.
            transform: Transformation to apply to time series.
            sample_time_series: Defines how a time series is obtained from the dataset.
                With NONE, index i reads series i % num_ts, so a dataset_weight below 1
                only ever reads the first ceil(num_ts * weight) series. UNIFORM draws a
                random usable series on every access.
            dataset_weight: Multiplicative factor to apply to dataset size.
            name: Dataset name, used in log messages.

        The dataset leaves out series shorter than the transform's GetPatchSize
        accepts, because GetPatchSize would raise on them. They do not count
        toward the dataset size, so its share of samples drops by the skipped
        fraction.
        ``short_series_skipped`` holds the count.
        """
        self.indexer = indexer
        self.transform = transform
        self.sample_time_series = sample_time_series
        self.dataset_weight = dataset_weight
        self.name = name
        self._index = self._usable_index()

        if sample_time_series == SampleTimeSeriesType.NONE:
            self.probabilities = None
        elif sample_time_series == SampleTimeSeriesType.UNIFORM:
            self.probabilities = indexer.get_uniform_probabilities()
        elif sample_time_series == SampleTimeSeriesType.PROPORTIONAL:
            self.probabilities = indexer.get_proportional_probabilities()
        else:
            raise ValueError(f"Unknown sample type {sample_time_series}")
        if self.probabilities is not None and self._index is not None:
            p = self.probabilities[self._index]
            self.probabilities = p / p.sum() if p.sum() > 0 else p

    def _usable_index(self) -> Optional[np.ndarray]:
        """Indices of series long enough for GetPatchSize, or None if all are."""
        self.short_series_skipped = 0
        steps = (self.transform.transformations if isinstance(self.transform, Chain)
                 else [self.transform])
        gps = next((t for t in steps if isinstance(t, GetPatchSize)), None)
        # Subclasses that index the indexer themselves keep every series.
        if (gps is None or len(self.indexer) == 0
                or type(self)._get_data is not TimeSeriesDataset._get_data):
            return None
        hf = getattr(self.indexer, "dataset", None)
        if hf is not None and "target" in hf.column_names and "freq" in hf.column_names:
            from amets.data.gift_eval_split import series_lengths

            lengths = series_lengths(hf)[0]
            freqs = hf.data.column("freq").to_pylist()
        else:
            items = [self.indexer[i] for i in range(len(self.indexer))]
            lengths = np.array([np.shape(x[gps.target_field])[-1] for x in items])
            freqs = [x["freq"] for x in items]
        need = {f: gps.min_length(f) for f in set(freqs)}
        ok = np.array([need[f] is not None and n >= need[f] for n, f in zip(lengths, freqs)])
        self.short_series_skipped = int((~ok).sum())
        if self.short_series_skipped == 0:
            return None
        logger.warning(
            "%s: skipping %d of %d series shorter than GetPatchSize accepts (min length by "
            "freq %s). Dataset size %d -> %d series (x weight %g).",
            self.name or "dataset", self.short_series_skipped, len(ok), need, len(ok),
            int(ok.sum()), self.dataset_weight)
        return np.flatnonzero(ok)

    def __getitem__(self, idx: int) -> dict[str, FlattenedData]:
        """Obtain a time series from the dataset, flattened.

        Args:
            idx: Index of the time series to retrieve. Ignored if
                sample_time_series is specified.

        Returns:
            Transformed time series data.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of length {len(self)}"
            )

        # Try up to 1000 times to get a non-constant sample.
        max_retries = 1000
        for attempt in range(max_retries):
            try:
                if self.sample_time_series != SampleTimeSeriesType.NONE:
                    sample_idx = np.random.choice(len(self.probabilities), p=self.probabilities)
                else:
                    sample_idx = idx
                
                return self.transform(self._flatten_data(self._get_data(sample_idx)))
            
            except ValueError as e:
                if "Constant time series" in str(e) or "NaN values" in str(e):
                    # Try the next random sample.
                    continue
                else:
                    raise
        
        # If all 1000 retries failed, add small noise to the target and retry.
        import warnings
        warnings.warn(f"TimeSeriesDataset: 1000 retries exhausted for idx={idx}, falling back to noise injection")
        sample_idx = np.random.choice(len(self.probabilities), p=self.probabilities) if self.sample_time_series != SampleTimeSeriesType.NONE else idx
        data = self._flatten_data(self._get_data(sample_idx))
        # Small noise lets the sample pass the constant-series check.
        if "target" in data:
            target = data["target"]
            if isinstance(target, np.ndarray):
                data["target"] = target + np.random.normal(0, 1e-4, target.shape).astype(target.dtype)
            elif isinstance(target, list):
                data["target"] = [
                    ts + np.random.normal(0, 1e-4, ts.shape).astype(ts.dtype)
                    if isinstance(ts, np.ndarray) else ts
                    for ts in target
                ]
        # Retry up to 100 times with other random samples plus noise.
        for _ in range(100):
            try:
                return self.transform(data)
            except ValueError as e:
                if "Constant time series" in str(e) or "NaN values" in str(e):
                    alt_idx = np.random.choice(len(self.probabilities), p=self.probabilities) if self.sample_time_series != SampleTimeSeriesType.NONE else np.random.randint(len(self))
                    data = self._flatten_data(self._get_data(alt_idx))
                    # Add noise so the sample is not constant.
                    if "target" in data:
                        target = data["target"]
                        if isinstance(target, np.ndarray):
                            data["target"] = target + np.random.normal(0, 1e-4, target.shape).astype(target.dtype)
                        elif isinstance(target, list):
                            data["target"] = [
                                ts + np.random.normal(0, 1e-4, ts.shape).astype(ts.dtype)
                                if isinstance(ts, np.ndarray) else ts
                                for ts in target
                            ]
                    continue
                raise
        raise RuntimeError(f"Failed to find non-constant sample after all retries (idx={idx})")

    @property
    def num_ts(self) -> int:
        """
        Number of time series in the dataset, excluding skipped short series.
        """
        return len(self.indexer) if self._index is None else len(self._index)

    def __len__(self) -> int:
        """
        Number of time series multiplied by dataset_weight, rounded up.
        """
        return int(np.ceil(self.num_ts * self.dataset_weight))

    def _get_data(self, idx: int) -> dict[str, Data | BatchedData]:
        """
        Read a time series from the Indexer.
        """
        idx = idx % self.num_ts
        return self.indexer[idx if self._index is None else int(self._index[idx])]

    @staticmethod
    def _flatten_data(data: dict[str, Data]) -> dict[str, FlattenedData]:
        """
        Convert time series fields into lists of univariate time series.
        """
        return {
            k: (
                [v]
                if isinstance(v, UnivarTimeSeries)
                else list(v) if isinstance(v, MultivarTimeSeries) else v
            )
            for k, v in data.items()
        }


class MultiSampleTimeSeriesDataset(TimeSeriesDataset):
    """
    Samples multiple time series and stacks them into a single time series.
    The time series in the underlying dataset must be aligned, with the same
    start and end dates.
    """

    def __init__(
        self,
        indexer: Indexer[dict[str, Any]],
        transform: Transformation,
        max_ts: int,
        combine_fields: tuple[str, ...],
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE,
        dataset_weight: float = 1.0,
        sampler: Sampler = get_sampler("beta_binomial", a=2, b=5),
    ):
        """
        Args:
            indexer: Underlying Indexer object.
            transform: Transformation to apply to time series.
            max_ts: Maximum number of time series that can be stacked together.
            combine_fields: Fields which should be stacked.
            sample_time_series: Defines how a time series is obtained from the dataset.
            dataset_weight: Multiplicative factor to apply to dataset size.
            sampler: How to sample the other time series.
        """
        super().__init__(indexer, transform, sample_time_series, dataset_weight)
        self.max_ts = max_ts
        self.combine_fields = combine_fields
        self.sampler = sampler

    def _get_data(self, idx: int) -> dict[str, BatchedData]:
        n_series = self.sampler(min(self.num_ts, self.max_ts))
        choices = np.concatenate([np.arange(idx), np.arange(idx + 1, self.num_ts)])
        others = np.random.choice(choices, n_series - 1, replace=False)
        samples = self.indexer[np.concatenate([[idx], others])]
        return samples

    def _flatten_data(
        self, samples: dict[str, BatchedData]
    ) -> dict[str, FlattenedData]:
        for field in samples.keys():
            if field in self.combine_fields:
                item = samples[field]
                if isinstance(item, list) and isinstance(item[0], MultivarTimeSeries):
                    samples[field] = [
                        univar for sample in samples[field] for univar in sample
                    ]
            elif isinstance(samples[field], BatchedDateTime):
                samples[field] = np.asarray(samples[field][0])
            elif isinstance(samples[field], BatchedString):
                samples[field] = samples[field][0]
            else:
                raise AssertionError(
                    f"Field {field} not accounted for in {self.indexer} MultiSampleTimeSeriesDataset"
                )
        return samples


class EvalDataset(TimeSeriesDataset):
    """
    Dataset class for validation.
    Use it with the Eval transformations.
    """

    def __init__(
        self,
        windows: int,
        indexer: Indexer[dict[str, Any]],
        transform: Transformation,
    ):
        """
        Args:
            windows: Number of windows to perform evaluation on.
        """
        super().__init__(
            indexer,
            transform,
            SampleTimeSeriesType.NONE,
            dataset_weight=windows,
        )

    def _get_data(self, idx: int) -> dict[str, Data]:
        window, idx = divmod(idx, self.num_ts)
        item = self.indexer[idx]
        item["window"] = window
        return item

