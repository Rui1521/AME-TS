#  SPDX-License-Identifier: Apache-2.0
#
#  build_eval_data() builds on the windowing of gift_eval/data.py from
#  https://github.com/SalesforceAIResearch/gift-eval
#  (Copyright (c) 2023, Salesforce, Inc., Apache License 2.0).

"""Model-free pieces of scripts/evaluate.py: task keys, patch-size grid, val/test
instances and noise seeds.

This module has no torch dependency, so tools can reuse the evaluation logic
without loading a model.
"""

from __future__ import annotations

import zlib
from typing import List, NamedTuple, Sequence

import numpy as np
from gluonts.dataset.split import split as gluonts_split

# Dataset names as GIFT-Eval reports them.
PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

# Default patch-size range per frequency, copied from amets/transform/patch.py.
DEFAULT_PATCH_RANGES = {
    "S": (4, 128), "s": (4, 128),
    "T": (4, 128), "min": (4, 128),
    "H": (4, 64),  "h": (4, 64),
    "D": (8, 64),  "d": (8, 64),
    "W": (8, 32),
    "M": (4, 32),  "ME": (4, 32), "MS": (4, 32),
    "Q": (1, 8),   "QE": (1, 8),  "QS": (1, 8),
    "Y": (1, 8),   "YE": (1, 8),  "YS": (1, 8),
    "A": (1, 8),
}


def get_all_patch_sizes_for_freq(freq: str, model_patch_sizes: Sequence[int]) -> List[int]:
    """Return all valid patch sizes for a given frequency."""
    from gluonts.time_feature import norm_freq_str
    import pandas as pd
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
        freq_name = norm_freq_str(offset.name)
    except Exception:
        freq_name = freq.upper()
    start, stop = DEFAULT_PATCH_RANGES.get(freq_name, (8, 64))
    valid_sizes = sorted([ps for ps in model_patch_sizes if start <= ps <= stop])
    if not valid_sizes:
        raise ValueError(
            f"No valid patch size for freq '{freq}' (normalized: '{freq_name}'). "
            f"Model patch_sizes={model_patch_sizes}, range=({start}, {stop})."
        )
    return valid_sizes


def grid_patch_sizes(valid_patch_sizes: Sequence[int], min_series_len: int) -> List[int]:
    """Patch sizes the grid actually runs: skip ps when 2*ps exceeds the shortest series."""
    return [ps for ps in valid_patch_sizes if ps * 2 <= min_series_len]


def val_min_input(valid_patch_sizes: Sequence[int], min_series_len: int) -> int:
    """History a series needs before val: the largest patch size the grid runs.

    Depends only on the task (freq, shortest series, model patch sizes), never on
    the candidate, so all candidates of a task see the same val items.
    """
    run = grid_patch_sizes(valid_patch_sizes, min_series_len)
    return max(run) if run else min(valid_patch_sizes)


def noise_seed(ds_config: str, split: str, mode: str) -> int:
    """Seed for the forecast sample noise, shared by all candidates of a task.

    With common random numbers, two candidates with identical point forecasts
    get identical MASE, and reruns reproduce the val ranking exactly.
    """
    return zlib.crc32(f"{ds_config}|{split}|{mode}".encode())


def normalize_freq(freq: str) -> str:
    """Strip anchor suffix: 'Q-DEC' -> 'Q', 'W-SUN' -> 'W'."""
    return freq.split('-')[0] if '-' in freq else freq


def build_ds_config(ds_name, term, dataset_properties):
    """Build dataset/freq/term string matching GIFT-Eval format."""
    if "/" in ds_name:
        ds_key = ds_name.split("/")[0].lower()
        ds_freq = ds_name.split("/")[1]
    else:
        ds_key = ds_name.lower()
        ds_freq = dataset_properties.get(PRETTY_NAMES.get(ds_key, ds_key), {}).get("frequency", "")
    ds_key = PRETTY_NAMES.get(ds_key, ds_key)
    freq_norm = normalize_freq(ds_freq)
    return f"{ds_key}/{freq_norm}/{term}" if freq_norm else f"{ds_key}/{term}"


# Val and test instances.

class _MinLength:
    """Re-iterable view of a gluonts dataset keeping series with >= min_len points."""

    def __init__(self, dataset, min_len: int):
        self.dataset = dataset
        self.min_len = min_len

    def __iter__(self):
        for entry in self.dataset:
            if np.asarray(entry["target"]).shape[-1] >= self.min_len:
                yield entry

    def __len__(self):
        return sum(1 for _ in self)


class EvalData(NamedTuple):
    data: object        # gluonts TestData
    n_series: int       # series in the task (after to_univariate, if any)
    n_kept: int         # series that contribute instances to this split


def entry_lengths(dataset, to_univariate: bool):
    """Length (shape[-1]) of every entry of ``dataset.gluonts_dataset``.

    Reads the Arrow ``target`` column without formatting any row. Returns None if
    the multivariate targets are ragged. The caller then counts by iterating.
    """
    import pyarrow.compute as pc
    col = dataset.hf_dataset.data.column("target")
    if dataset.target_dim == 1:
        return pc.list_value_length(col).to_numpy(zero_copy_only=False)
    dims = pc.list_value_length(col).to_numpy(zero_copy_only=False)
    inner = pc.list_value_length(pc.list_flatten(col)).to_numpy(zero_copy_only=False)
    starts = np.r_[0, np.cumsum(dims)[:-1]]
    first = inner[starts]
    if not np.array_equal(np.repeat(first, dims), inner):
        return None
    return inner if to_univariate else first


def build_eval_data(dataset, split: str, min_input: int = 1,
                    to_univariate: "bool | None" = None) -> EvalData:
    """Instances for one task. ``dataset`` is a gift_eval ``Dataset``.

    Test uses GIFT-Eval's own test_data. Val uses the same number of windows,
    with the same prediction_length and distance, placed immediately before the
    test region. The last val label ends where test begins, and no val input
    contains a test point. Gluonts' negative offsets wrap around for series
    shorter than the offset, which would put test points into val inputs. Val
    therefore drops series with fewer than ``min_input`` points before val, and
    n_kept counts the series it keeps. Callers pass ``val_min_input(...)``.

    With ``to_univariate`` given (how ``dataset`` was built), n_series and n_kept
    come from the Arrow column lengths, so rows are formatted only once, when the
    instances are generated. Without it, the counts iterate the dataset (slow).
    """
    pred_len, windows = dataset.prediction_length, dataset.windows
    offset = pred_len * 2 * windows
    lengths = entry_lengths(dataset, to_univariate) if to_univariate is not None else None
    if lengths is not None:
        n_series = len(lengths)
    else:
        n_series = sum(1 for _ in dataset.gluonts_dataset)
    if split == "test":
        return EvalData(dataset.test_data, n_series, n_series)
    if split != "val":
        raise ValueError(f"Unknown split: {split}")
    kept = _MinLength(dataset.gluonts_dataset, offset + min_input)
    n_kept = int((lengths >= offset + min_input).sum()) if lengths is not None else len(kept)
    if n_kept == 0:
        raise ValueError(
            f"No series of {dataset.name} ({dataset.term.value}) has "
            f"2*windows*pred_len + {min_input} = {offset + min_input} points. There are no "
            f"val instances.")
    if n_kept < n_series:
        print(f"  WARNING: val drops {n_series - n_kept}/{n_series} series shorter than "
              f"{offset + min_input} (2*windows*pred_len + {min_input})")
    _, template = gluonts_split(kept, offset=-offset)
    data = template.generate_instances(
        prediction_length=pred_len, windows=windows, distance=pred_len,
    )
    return EvalData(data, n_series, n_kept)
