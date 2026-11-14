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

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from amets.common.typing import UnivarTimeSeries

from ._base import Transformation
from ._mixin import MapFuncMixin


@dataclass
class PatchCrop(MapFuncMixin, Transformation):
    """Crop fields in a data_entry in the temporal dimension based on a patch_size.

    Args:
        min_time_patches: Minimum number of patches for the time dimension.
        max_patches: Maximum number of patches for time * dim (if flattening).
        will_flatten: Whether time series fields will be flattened subsequently.
        offset: Whether to offset the start of the crop.
        fields: Fields to crop.
    """

    min_time_patches: int
    max_patches: int
    will_flatten: bool = False
    offset: bool = True
    fields: tuple[str, ...] = ("target",)
    optional_fields: tuple[str, ...] = ("past_feat_dynamic_real",)

    def __post_init__(self):
        assert (
            self.min_time_patches <= self.max_patches
        ), "min_patches must be <= max_patches"
        assert len(self.fields) > 0, "fields must be non-empty"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        a, b = self._get_boundaries(data_entry)
        self.map_func(
            partial(self._crop, a=a, b=b),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        
        # Reject a crop whose target is constant, including all zeros.
        if "target" in data_entry:
            cropped_target = data_entry["target"]  # List of cropped time series.
            all_constant = all(np.std(ts) < 1e-6 for ts in cropped_target)
            if all_constant:
                raise ValueError("Constant time series. Skipping this sample.")
        
        return data_entry

    @staticmethod
    def _crop(data_entry: dict[str, Any], field: str, a: int, b: int) -> Sequence:
        return [ts[a:b] for ts in data_entry[field]]

    @staticmethod
    def _find_active_region(field: list[UnivarTimeSeries]) -> tuple[int, int]:
        """Find the first and last non-zero timestep across all variates."""
        time = field[0].shape[0]
        # A timestep is active if any variate is non-zero and not NaN.
        active = np.zeros(time, dtype=bool)
        for ts in field:
            active |= (ts != 0) & ~np.isnan(ts)

        if not active.any():
            # All values are zero or NaN. Fall back to the full range. The constant-series
            # check runs after the crop.
            return 0, time

        first_active = int(np.argmax(active))
        last_active = int(time - 1 - np.argmax(active[::-1]))
        return first_active, last_active + 1  # exclusive end

    def _get_boundaries(self, data_entry: dict[str, Any]) -> tuple[int, int]:
        patch_size = data_entry["patch_size"]
        field: list[UnivarTimeSeries] = data_entry[self.fields[0]]
        time = field[0].shape[0]
        nvar = (
            sum(len(data_entry[f]) for f in self.fields)
            + sum(len(data_entry[f]) for f in self.optional_fields if f in data_entry)
            if self.will_flatten
            else 1
        )

        # Skip leading and trailing zeros by restricting the crop to the active region.
        active_start, active_end = self._find_active_region(field)
        active_len = active_end - active_start

        # If the active region is too short, fall back to the full range.
        if active_len < self.min_time_patches * patch_size:
            active_start = 0
            active_len = time
            active_end = time

        offset = (
            np.random.randint(
                active_len % patch_size + 1
            )  # Offset by [0, patch_size) so the start is not always a multiple of patch_size.
            if self.offset
            else 0
        )
        total_patches = (
            active_len - offset
        ) // patch_size  # Total number of patches in the active region.

        # Divide max_patches by nvar if the time series is flattened later, and cap
        # it at the number of available patches.
        max_patches = min(self.max_patches // nvar, total_patches)
        if max_patches < self.min_time_patches:
            raise ValueError(
                f"max_patches={max_patches} < min_time_patches={self.min_time_patches}"
            )

        num_patches = np.random.randint(
            self.min_time_patches, max_patches + 1
        )  # Number of patches in the crop.
        max_first = total_patches - num_patches
        # Beta(2, 1) biases the crop toward the end of the series, where data is
        # most recent. Its mean position is 2/3 of the way through the series.
        first = int(np.random.beta(2, 1) * (max_first + 1))
        first = min(first, max_first)

        start = active_start + offset + first * patch_size
        stop = start + num_patches * patch_size
        return start, stop


@dataclass
class EvalCrop(MapFuncMixin, Transformation):
    offset: int
    distance: int
    prediction_length: int
    context_length: int
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        a, b = self._get_boundaries(data_entry)
        self.map_func(
            partial(self._crop, a=a, b=b),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )


        return data_entry

    @staticmethod
    def _crop(data_entry: dict[str, Any], field: str, a: int, b: int) -> Sequence:
        return [ts[a : b or None] for ts in data_entry[field]]

    def _get_boundaries(self, data_entry: dict[str, Any]) -> tuple[int, int]:
        field: list[UnivarTimeSeries] = data_entry[self.fields[0]]
        time = field[0].shape[0]
        window = data_entry["window"]
        fcst_start = self.offset + window * self.distance
        a = fcst_start - self.context_length
        b = fcst_start + self.prediction_length

        if self.offset >= 0:
            assert time >= b > a >= 0
        else:
            assert 0 >= b > a >= -time

        return a, b
