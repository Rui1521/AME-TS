#  SPDX-License-Identifier: Apache-2.0
#
#  prediction_length() and windows() mirror gift_eval/data.py from
#  https://github.com/SalesforceAIResearch/gift-eval
#  (Copyright (c) 2023, Salesforce, Inc., Apache License 2.0).

"""Train/val/test regions of the GIFT-Eval benchmark.

For each task, a dataset and term pair, GIFT-Eval puts ``windows * pred_len``
test points at the end of each series. Val is the same number of points right
before test. Train is everything before val. A dataset's point cut is the
largest ``2 * windows * pred_len`` over its terms that appear among the 97
tasks. The frequency variants of one source, such as ``electricity/15T``,
``/H``, ``/D`` and ``/W``, cover the same time span. ``group_cuts`` therefore
cuts all of them at the earliest val-start time over the group, so no variant's
training data covers another variant's val or test period.

The module uses only numpy, pyarrow and pandas. It does not import torch, so
the split builder and the leak checker run without a GPU stack. The windows and
prediction_length formulas are reimplemented because ``gift_eval.data`` uses
``X | Y`` annotations that fail to import on Python < 3.10.
``crosscheck_with_gift_eval`` compares against the upstream class whenever it
imports.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from amets.common.env import dataset_properties_path, gift_eval_repo

# The 55 GIFT-Eval dataset configs of configs/data/ame_*.yaml, in the names
# GiftEvalDatasetBuilder and gift_eval.data.Dataset use (nested name/freq dirs).
GIFT_EVAL_55 = [
    "m4_yearly", "m4_quarterly", "m4_monthly", "m4_weekly", "m4_daily", "m4_hourly",
    "electricity/15T", "electricity/H", "electricity/D", "electricity/W",
    "solar/10T", "solar/H", "solar/D", "solar/W",
    "jena_weather/10T", "jena_weather/H", "jena_weather/D",
    "temperature_rain_with_missing",
    "ett1/15T", "ett1/H", "ett1/D", "ett1/W",
    "ett2/15T", "ett2/H", "ett2/D", "ett2/W",
    "LOOP_SEATTLE/5T", "LOOP_SEATTLE/H", "LOOP_SEATTLE/D",
    "SZ_TAXI/15T", "SZ_TAXI/H",
    "M_DENSE/H", "M_DENSE/D",
    "hospital", "covid_deaths",
    "us_births/D", "us_births/M", "us_births/W",
    "car_parts_with_missing", "restaurant",
    "hierarchical_sales/D", "hierarchical_sales/W",
    "saugeenday/D", "saugeenday/M", "saugeenday/W",
    "kdd_cup_2018_with_missing/H", "kdd_cup_2018_with_missing/D",
    "bitbrains_fast_storage/5T", "bitbrains_fast_storage/H",
    "bitbrains_rnd/5T", "bitbrains_rnd/H",
    "bizitobs_application", "bizitobs_service",
    "bizitobs_l2c/5T", "bizitobs_l2c/H",
]
NUM_TASKS = 97
TERMS = ("short", "medium", "long")

# Series fields cut along their last (time) axis. Any other array column whose
# last axis matches the target length is ambiguous and raises.
TIME_FIELDS = ("target", "past_feat_dynamic_real", "feat_dynamic_real")

# Constants mirrored from gift_eval/data.py.
TEST_SPLIT = 0.1
MAX_WINDOW = 20
M4_PRED_LENGTH_MAP = {"A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
PRED_LENGTH_MAP = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}
TERM_MULTIPLIER = {"short": 1, "medium": 10, "long": 15}
_DEPRECATED_FREQ = {"Y": "A", "YE": "A", "QE": "Q", "ME": "M", "h": "H",
                    "min": "T", "s": "S", "us": "U"}

# Dataset names mirrored from build_ds_config in scripts/evaluate.py.
PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}


def task_key(ds_name: str, term: str, dataset_properties: dict) -> str:
    """Task key as in the Seasonal-Naive CSV, e.g. ``loop_seattle/5T/short``.

    Uses the same logic as ``build_ds_config`` in scripts/evaluate.py. It does
    not import that module because it needs torch. ``load_tasks`` checks that the
    55 configs cover exactly the 97 tasks, so any drift between the two raises.
    """
    if "/" in ds_name:
        ds_key, ds_freq = ds_name.split("/")[0].lower(), ds_name.split("/")[1]
    else:
        ds_key = ds_name.lower()
        ds_freq = dataset_properties.get(PRETTY_NAMES.get(ds_key, ds_key), {}).get("frequency", "")
    ds_key = PRETTY_NAMES.get(ds_key, ds_key)
    freq = ds_freq.split("-")[0] if "-" in ds_freq else ds_freq
    return f"{ds_key}/{freq}/{term}" if freq else f"{ds_key}/{term}"


def seasonal_naive_csv() -> Path:
    env_path = os.environ.get("SEASONAL_NAIVE_RESULTS")
    if env_path:
        return Path(env_path)
    return gift_eval_repo() / "results" / "seasonal_naive" / "all_results.csv"


def load_tasks() -> dict[str, list[str]]:
    """Map each of the 55 dataset configs to its terms among the 97 tasks.

    Raises:
        ValueError: If the task list is not 97 long, a task maps to no config,
            or a config has no task.
    """
    path = seasonal_naive_csv()
    if not path.is_file():
        raise FileNotFoundError(f"Seasonal-Naive results not found at {path} "
                                ". Set SEASONAL_NAIVE_RESULTS or GIFT_EVAL_REPO.")
    with open(path) as f:
        tasks = {row["dataset"] for row in csv.DictReader(f)}
    if len(tasks) != NUM_TASKS:
        raise ValueError(f"{path} lists {len(tasks)} tasks, expected {NUM_TASKS}")
    with open(dataset_properties_path()) as f:
        props = json.load(f)

    terms: dict[str, list[str]] = {}
    covered: set[str] = set()
    for ds in GIFT_EVAL_55:
        keys = {t: task_key(ds, t, props) for t in TERMS}
        terms[ds] = [t for t in TERMS if keys[t] in tasks]
        covered.update(keys[t] for t in terms[ds])
        if not terms[ds]:
            raise ValueError(f"{ds}: no task among the {NUM_TASKS} (keys {sorted(keys.values())})")
    if covered != tasks:
        raise ValueError(f"tasks not covered by the 55 configs: {sorted(tasks - covered)}")
    return terms


def _norm_freq(freq: str) -> str:
    from gluonts.time_feature import norm_freq_str
    from pandas.tseries.frequencies import to_offset

    f = norm_freq_str(to_offset(freq).name)
    return _DEPRECATED_FREQ.get(f, f)


def prediction_length(name: str, freq: str, term: str) -> int:
    f = _norm_freq(freq)
    base = M4_PRED_LENGTH_MAP[f] if "m4" in name else PRED_LENGTH_MAP[f]
    return TERM_MULTIPLIER[term] * base


def windows(name: str, min_series_length: int, pred_len: int) -> int:
    if "m4" in name:
        return 1
    w = math.ceil(TEST_SPLIT * min_series_length / pred_len)
    return min(max(1, w), MAX_WINDOW)


def series_length(target: np.ndarray) -> int:
    return int(np.shape(target)[-1])


def min_series_length(hf_dataset) -> int:
    """Shortest target length (first variate for multivariate), as gift_eval."""
    import pyarrow.compute as pc

    col = hf_dataset.data.column("target")
    if np.ndim(hf_dataset.with_format("numpy")[0]["target"]) > 1:
        lengths = pc.list_value_length(pc.list_flatten(pc.list_slice(col, 0, 1)))
    else:
        lengths = pc.list_value_length(col)
    return int(min(lengths.to_numpy()))


def dataset_cut(name: str, hf_dataset, terms: list[str]) -> dict:
    """Training cut of one dataset config, with the per-term breakdown."""
    freq = hf_dataset[0]["freq"]
    min_len = min_series_length(hf_dataset)
    per_term = {}
    for term in terms:
        p = prediction_length(name, freq, term)
        w = windows(name, min_len, p)
        per_term[term] = {"prediction_length": p, "windows": w,
                          "test_points": w * p, "val_points": w * p}
    cut = max(2 * v["test_points"] for v in per_term.values())
    return {"cut": cut, "freq": freq, "min_series_length": min_len, "terms": per_term}


def crosscheck_with_gift_eval(name: str, raw_root: Path, info: dict) -> str:
    """Compare ``dataset_cut`` against gift_eval.data.Dataset when it imports.

    Returns "ok" or "unavailable: <reason>". Raises ValueError on a mismatch.
    """
    try:
        from amets.common.env import add_gift_eval_to_path

        add_gift_eval_to_path()
        from gift_eval.data import Dataset
    except Exception as e:  # Python < 3.10 or a missing checkout
        return f"unavailable: {type(e).__name__}: {e}"
    old = os.environ.get("GIFT_EVAL")
    os.environ["GIFT_EVAL"] = str(raw_root)
    try:
        for term, v in info["terms"].items():
            ds = Dataset(name=name, term=term, storage_env_var="GIFT_EVAL")
            got = (ds.prediction_length, ds.windows)
            want = (v["prediction_length"], v["windows"])
            if got != want:
                raise ValueError(f"{name}/{term}: gift_eval (pred_len, windows)={got}, "
                                 f"local={want}")
    finally:
        if old is None:
            os.environ.pop("GIFT_EVAL", None)
        else:
            os.environ["GIFT_EVAL"] = old
    return "ok"


# Time-based cut across the frequency variants of one source.
# Every train-split manifest entry records CUT_RULE. Loaders refuse any other value.
CUT_RULE = ("time-v2: per source, keep periods ending before the earliest val start "
            "minus one coarsest period; single-variant sources use the point cut")

def source_group(name: str) -> str:
    """Source of a dataset config: ``electricity/H`` -> ``electricity``. A flat name is its own source."""
    return name.split("/")[0]


def source_groups(names: list[str] = GIFT_EVAL_55) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(source_group(n), []).append(n)
    return groups


def series_lengths(hf_dataset) -> tuple[np.ndarray, np.ndarray]:
    """Per row: time length of the first variate and number of target variates.

    Works chunk by chunk, because one combined array overflows Arrow's 32-bit
    list offsets beyond about 2.1B values. lotsa_buildings_900k has 15.7B.
    """
    import pyarrow.compute as pc

    multivar = np.ndim(hf_dataset.with_format("numpy")[0]["target"]) > 1
    lengths, n_vars = [], []
    for col in hf_dataset.data.column("target").chunks:
        if multivar:
            inner_len = pc.list_value_length(pc.list_flatten(col)).to_numpy()
            n_var = np.bincount(pc.list_parent_indices(col).to_numpy(), minlength=len(col))
            length = inner_len[np.r_[0, np.cumsum(n_var)[:-1]]] if len(col) else inner_len[:0]
        else:
            n_var = np.ones(len(col), dtype=np.int64)
            length = pc.list_value_length(col).to_numpy()
        lengths.append(length)
        n_vars.append(n_var)
    if not lengths:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(lengths).astype(np.int64), np.concatenate(n_vars).astype(np.int64)


def _parr(ordinals, freq: str):
    import pandas as pd

    return pd.arrays.PeriodArray(np.asarray(ordinals, dtype=np.int64), dtype=pd.PeriodDtype(freq))


def _step(freq: str) -> int:
    """Ordinal increment of one period (pandas counts ``5T`` ordinals in minutes)."""
    from pandas.tseries.frequencies import to_offset

    return int(to_offset(freq).n)


def period_seconds(ordinals, freq: str, end: bool = False) -> np.ndarray:
    """First (or last) second of each period, as int64 seconds since 1970.

    Uses period-ordinal arithmetic only. It creates no nanosecond Timestamps, so
    the 1677 to 2262 range limit does not apply.
    """
    return np.asarray(_parr(ordinals, freq).asfreq("S", how="end" if end else "start").asi8)


def periods_before(t_sec: np.ndarray, start_ord: np.ndarray, freq: str) -> np.ndarray:
    """Per row: how many periods from the series start end strictly before second ``t_sec[i]``."""
    import pandas as pd
    from pandas.tseries.frequencies import to_offset

    t_sec = np.asarray(t_sec, dtype=np.int64)
    step = _step(freq)
    if step == 1:  # Consecutive ordinals (H, D, W-SUN, M, Q, A). Drop the period holding t.
        return np.asarray(_parr(t_sec, "S").asfreq(freq).asi8 - start_ord, dtype=np.int64)
    off = to_offset(freq)
    if not isinstance(off, pd.offsets.Tick):
        raise ValueError(f"unsupported multiplied non-fixed frequency {freq!r}")
    width = int(pd.Timedelta(off).total_seconds())
    return (t_sec - period_seconds(start_ord, freq)) // width


def series_timing(hf_dataset) -> dict:
    """Start period ordinals, lengths, freq and item ids of every row."""
    freqs = set(hf_dataset.unique("freq"))
    if len(freqs) != 1:
        raise ValueError(f"rows disagree on freq: {sorted(freqs)}")
    freq = freqs.pop()
    start_sec = hf_dataset.data.column("start").cast("timestamp[s]").to_numpy().astype(np.int64)
    length, n_var = series_lengths(hf_dataset)
    item_ids = (hf_dataset.data.column("item_id").to_pylist()
                if "item_id" in hf_dataset.column_names else [None] * len(hf_dataset))
    return {"freq": freq, "start_ord": np.asarray(_parr(start_sec, "S").asfreq(freq).asi8),
            "length": length, "n_var": n_var, "item_id": item_ids}


def coarse_period_seconds(freq: str, ordinal: int) -> int:
    """Longest of the 12 periods ending at ``ordinal`` (covers month and leap-year lengths)."""
    o = ordinal - _step(freq) * np.arange(12)
    return int((period_seconds(o, freq, end=True) - period_seconds(o, freq) + 1).max())


def _iso(sec) -> str:
    return str(np.datetime64(int(sec), "s"))


def group_cuts(names: list[str], raws: dict, terms: dict[str, list[str]]) -> dict:
    """Keep lengths for every variant of one source.

    With one variant, each series keeps ``len - point_cut`` points, which end
    at its own val start. With several variants, such as ``electricity/15T``,
    ``/H``, ``/D`` and ``/W``, the val start of series i is the first period of
    its last ``point_cut`` points, or the series start if the series is shorter.
    Let t* be the earliest val start over the group. It is taken per item when
    every variant has the same set of unique item ids, and over the whole group
    otherwise.
    Every series keeps the periods that end strictly before t* minus one period
    of the coarsest variant. The margin makes the cut independent of whether a
    variant labels a period by its start or its end. All times are int64
    seconds from period ordinals.

    Raises:
        ValueError: If the variants' end times differ by more than one coarsest
            period, per item or across each variant's latest end.
    Returns:
        ``{"variants": {name: {"keep", "point_cut", "timing", "cut_info", "summary"}},
        "summary": {...}}``
    """
    info = {}
    for n in names:
        t = series_timing(raws[n])
        cut = dataset_cut(n, raws[n], terms[n])
        t["point_cut"], t["cut_info"] = cut["cut"], cut
        info[n] = t

    if len(names) == 1:
        n, t = names[0], info[names[0]]
        keep = np.maximum(t["length"] - t["point_cut"], 0).astype(np.int64)
        summary = {"freq": t["freq"], "point_cut": int(t["point_cut"]),
                   "cut_points_min": int((t["length"] - keep).min()),
                   "cut_points_max": int((t["length"] - keep).max()),
                   "cut_days_min": None, "cut_days_max": None, "margin_days": 0.0,
                   "extra_points_vs_point_cut_max": 0}
        return {"variants": {n: {"keep": keep, "point_cut": t["point_cut"], "timing": t,
                                 "cut_info": t["cut_info"], "summary": summary}},
                "summary": {"variants": list(names), "mode": "single_variant",
                            "per_variant": {n: summary}}}

    for t in info.values():
        step = _step(t["freq"])
        last = t["start_ord"] + step * (t["length"] - 1)
        t["val_start"] = period_seconds(t["start_ord"] + step * np.maximum(
            t["length"] - t["point_cut"], 0), t["freq"])
        t["end"] = period_seconds(last, t["freq"], end=True)
        t["period_s"] = coarse_period_seconds(t["freq"], int(last[0]))
    period = max(t["period_s"] for t in info.values())

    ids = [t["item_id"] for t in info.values()]
    per_item = (all(None not in i and len(set(i)) == len(i) for i in ids)
                and all(set(i) == set(ids[0]) for i in ids))
    if per_item:
        best_vs, lo, hi = {}, {}, {}
        for t in info.values():
            for it, v, e in zip(t["item_id"], t["val_start"], t["end"]):
                best_vs[it] = min(best_vs.get(it, v), v)
                lo[it], hi[it] = min(lo.get(it, e), e), max(hi.get(it, e), e)
        bad = [it for it in lo if hi[it] - lo[it] > period]
        if bad:
            raise ValueError(f"{names}: end times of {len(bad)} items differ by more than one "
                             f"coarse period ({period / 86400:g} d). For example {bad[0]!r}: "
                             f"{(hi[bad[0]] - lo[bad[0]]) / 86400:g} d")
        for t in info.values():
            t["t_star"] = np.array([best_vs[it] for it in t["item_id"]], dtype=np.int64)
    else:
        latest = {n: int(t["end"].max()) for n, t in info.items()}
        if max(latest.values()) - min(latest.values()) > period:
            raise ValueError(f"{names}: latest end times differ by more than one coarse "
                             f"period ({period / 86400:g} d): "
                             f"{ {n: _iso(v) for n, v in latest.items()} }")
        t0 = min(int(t["val_start"].min()) for t in info.values())
        for t in info.values():
            t["t_star"] = np.full(len(t["length"]), t0, dtype=np.int64)

    variants = {}
    for n, t in info.items():
        cap = np.maximum(t["length"] - t["point_cut"], 0)
        keep = np.clip(periods_before(t["t_star"] - period, t["start_ord"], t["freq"]), 0, cap)
        keep_nomargin = np.clip(periods_before(t["t_star"], t["start_ord"], t["freq"]), 0, cap)
        cut_pts = t["length"] - keep
        cut_days = (t["end"] - (t["t_star"] - period)) / 86400
        variants[n] = {
            "keep": keep.astype(np.int64), "point_cut": t["point_cut"], "timing": t,
            "cut_info": t["cut_info"],
            "summary": {
                "freq": t["freq"], "point_cut": int(t["point_cut"]),
                "end_min": _iso(t["end"].min()), "end_max": _iso(t["end"].max()),
                "val_start_min": _iso(t["val_start"].min()),
                "cut_time_min": _iso((t["t_star"] - period).min()),
                "cut_time_max": _iso((t["t_star"] - period).max()),
                "cut_points_min": int(cut_pts.min()), "cut_points_max": int(cut_pts.max()),
                "cut_days_min": round(float(cut_days.min()), 3),
                "cut_days_max": round(float(cut_days.max()), 3),
                "margin_days": round(period / 86400, 3),
                "margin_points_max": int((keep_nomargin - keep).max()),
                "extra_points_vs_point_cut_max": int((cut_pts - np.minimum(
                    t["point_cut"], t["length"])).max()),
            }}
    return {"variants": variants,
            "summary": {"variants": list(names), "mode": "per_item" if per_item else "group_wide",
                        "coarse_period_days": round(period / 86400, 3),
                        "margin_days": round(period / 86400, 3),
                        "earliest_val_start": _iso(min(t["val_start"].min() for t in info.values())),
                        "per_variant": {n: v["summary"] for n, v in variants.items()}}}


def offset_seconds(start_sec: int, freq: str, offsets) -> np.ndarray:
    """First second of the period ``offsets`` steps after a series' start."""
    o = int(_parr([start_sec], "S").asfreq(freq).asi8[0])
    return period_seconds(o + _step(freq) * np.asarray(offsets, dtype=np.int64), freq)
