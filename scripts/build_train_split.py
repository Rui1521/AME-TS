#!/usr/bin/env python3
"""Write the GIFT-Eval train split used for pretraining.

Reads the raw benchmark from $GIFT_EVAL_DATA. For each dataset config, writes
HF-Arrow with the input's schema to $PRETRAIN_DATA/<name>. The path is nested,
for example ``electricity/H``, as GiftEvalDatasetBuilder resolves it.

Cut: the script groups configs by source, so ``electricity/15T``, ``/H``, ``/D``
and ``/W`` form one group. For every variant, the val start of a series is the first
period of its last ``point_cut`` points, where point_cut is the largest
``2 * windows * pred_len`` over the variant's terms among the 97 tasks. Every
series of every variant in the group keeps only the periods that end before the
earliest val start over the group minus one period of the coarsest variant, so
the cut holds whether a variant labels a period by its start or its end. The
bound is per item when all variants share the same unique item ids, and over
the whole group otherwise (``gift_eval_split.group_cuts``). The build fails if
the variants' end dates differ by more than one coarsest period. A source with
a single variant, such as m4_* or hospital, uses its point cut.

The script slices all time-indexed columns, such as target and
past_feat_dynamic_real, to the same kept length and copies other columns. It
does not use gluonts ``split``, whose negative offsets wrap on short series.
Series with fewer than --min_length kept points are dropped. The default is 1.
A dataset that would lose every series is an error. The manifest counts kept
series shorter than 16 points per dataset.
$PRETRAIN_DATA/gift_eval_train_split_manifest.json records the cut rule, times,
cut sizes and counts per dataset and per source group. A leak check can
re-verify the cut from the raw data.

Idempotent: the script skips a dataset whose output exists with the same source
fingerprint, kept lengths and --min_length. Any other existing output is an
error unless --overwrite is set.

Usage (from the repository root):
    python scripts/build_train_split.py
    python scripts/build_train_split.py --datasets m4_weekly electricity/H --dry_run
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amets.common.env import env  # noqa: E402
from amets.data.pool_guard import TRAIN_SPLIT_POOL  # noqa: E402
from amets.data.gift_eval_split import (  # noqa: E402
    CUT_RULE,
    GIFT_EVAL_55,
    TIME_FIELDS,
    crosscheck_with_gift_eval,
    group_cuts,
    load_tasks,
    series_length,
    source_group,
    source_groups,
)

MANIFEST_NAME = "gift_eval_train_split_manifest.json"
SHORT_REPORT = 16  # the manifest counts kept series shorter than this


def patch_min_length(freq: str) -> int:
    """Shortest series GetPatchSize(min_time_patches=2) accepts with patch sizes 4..128.

    Mirrors DefaultPatchSizeConstraints (amets/transform/patch.py). The smallest
    allowed patch is 8 for D/B/W and 4 for S/T/H/M/Q/A, and the minimum length is
    twice that.
    """
    from gluonts.time_feature import norm_freq_str
    from pandas.tseries.frequencies import to_offset

    return 16 if norm_freq_str(to_offset(freq).name) in ("D", "d", "B", "W") else 8


def source_fingerprint(path: Path) -> dict:
    """Cheap identity of an on-disk HF dataset: arrow fingerprint and file sizes."""
    state = json.loads((path / "state.json").read_text())
    files = sorted(f["filename"] for f in state["_data_files"])
    return {"fingerprint": state.get("_fingerprint"),
            "files": {f: (path / f).stat().st_size for f in files}}


def time_fields(hf_dataset) -> list[str]:
    return [c for c in TIME_FIELDS if c in hf_dataset.column_names]


def _has_time_axis(v, length: int) -> bool:
    """True if a 1-D or nested (2-D, ragged) array-like has a last axis of ``length``."""
    if isinstance(v, np.ndarray) and v.dtype != object:
        return v.ndim >= 1 and v.shape[-1] == length
    if isinstance(v, (list, tuple, np.ndarray)):
        return len(v) == length or any(_has_time_axis(u, length) for u in v
                                       if isinstance(u, (list, tuple, np.ndarray)))
    return False


def cut_rows(hf_dataset, fields: tuple, keep: tuple, dropped: tuple):
    """Yield rows with every time field sliced to ``keep[i]``, skipping ``dropped``."""
    ds = hf_dataset.with_format("numpy", columns=list(fields), output_all_columns=True)
    dropped = set(dropped)
    for i in range(len(ds)):
        if i in dropped:
            continue
        row = ds[i]
        length = series_length(row["target"])
        for f in fields:
            if np.shape(row[f])[-1] != length:
                raise ValueError(f"row {i}: {f} has length {np.shape(row[f])[-1]}, "
                                 f"target has {length}")
        for c, v in row.items():
            if c not in fields and _has_time_axis(v, length):
                raise ValueError(f"row {i}: column {c!r} has an axis of the target's "
                                 "length. Add it to TIME_FIELDS if it is time-indexed.")
        if not 1 <= keep[i] <= length:  # a negative stop would wrap, so never slice with one
            raise ValueError(f"row {i}: keep {keep[i]} outside [1, {length}] but not dropped")
        for f in fields:
            row[f] = np.asarray(row[f])[..., :keep[i]]
        yield row


def build_one(name: str, raw, variant: dict, group: str, raw_root: Path, out_root: Path,
              min_length: int, overwrite: bool, dry_run: bool, manifest: dict) -> dict:
    import datasets

    src, dst = raw_root / name, out_root / name
    keep, n_var = variant["keep"], variant["timing"]["n_var"]
    length = variant["timing"]["length"]
    info = dict(variant["cut_info"])
    point_cut = info.pop("cut")
    info["gift_eval_crosscheck"] = crosscheck_with_gift_eval(name, raw_root, {"terms": info["terms"]})
    entry = {
        "source_path": str(src),
        "output_path": str(dst),
        "source": source_fingerprint(src),
        "min_length": min_length,
        "cut_rule": CUT_RULE,
        "group": group,
        "point_cut": point_cut,
        **info,
        **{k: v for k, v in variant["summary"].items() if k not in ("freq", "point_cut")},
        "keep_sha256": hashlib.sha256(keep.astype("<i8").tobytes()).hexdigest(),
        "series_in": len(raw),
    }

    old = manifest.get(name)
    if dst.exists() and not dry_run:
        same = old is not None and all(
            old.get(k) == entry[k] for k in ("source", "keep_sha256", "min_length", "cut_rule"))
        if same and not overwrite:
            print(f"SKIP {name}: up to date")
            return old
        if not overwrite:
            raise FileExistsError(f"{name}: {dst} exists and does not match this build. "
                                  "Pass --overwrite.")

    fields = time_fields(raw)
    drop = keep < min_length
    dropped = np.flatnonzero(drop).tolist()
    if len(dropped) == len(raw):
        raise ValueError(f"{name}: all {len(raw)} series keep fewer than --min_length "
                         f"{min_length} points. Nothing is left to train on.")
    if not dry_run:
        tmp = Path(tempfile.mkdtemp(prefix=".tmp-split-", dir=out_root))
        try:
            out = datasets.Dataset.from_generator(
                cut_rows,
                # Pass tuples, not lists, because datasets shards generators over list kwargs.
                gen_kwargs={"hf_dataset": raw, "fields": tuple(fields),
                            "keep": tuple(int(k) for k in keep), "dropped": tuple(dropped)},
                features=raw.features,
                cache_dir=str(tmp / "cache"),
            )
            if len(out) != len(raw) - len(dropped):
                raise RuntimeError(f"{name}: wrote {len(out)} rows, expected "
                                   f"{len(raw) - len(dropped)}")
            out.save_to_disk(str(tmp / "data"))
            del out
            if dst.exists():
                shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            (tmp / "data").rename(dst)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    short = (keep >= min_length) & (keep < SHORT_REPORT)
    pmin = patch_min_length(variant["timing"]["freq"])
    entry.update(
        time_fields=fields,
        series_shorter_than_16=int(short.sum()),
        patch_min_length=pmin,
        series_below_patch_min=int(((keep >= min_length) & (keep < pmin)).sum()),
        series_out=len(raw) - len(dropped),
        series_dropped=len(dropped),
        dropped_indices=dropped,
        points_in=int((length * n_var).sum()),
        points_out=int((keep * n_var)[~drop].sum()),
    )
    days = ("" if entry["cut_days_min"] is None else
            f" ({entry['cut_days_min']}-{entry['cut_days_max']} days, margin "
            f"{entry['margin_days']} d)")
    print(f"{'DRY ' if dry_run else ''}{name}: point_cut={point_cut} cut "
          f"{entry['cut_points_min']}-{entry['cut_points_max']} points{days} "
          f"terms={list(info['terms'])} "
          f"series {entry['series_in']}->{entry['series_out']}, points "
          f"{entry['points_in']}->{entry['points_out']}, "
          f"crosscheck={info['gift_eval_crosscheck'].split(':')[0]}")
    return entry


def write_manifest(path: Path, raw_root: Path, entries: dict, groups: dict) -> None:
    short = {n: e["series_shorter_than_16"] for n, e in sorted(entries.items())
             if e.get("series_shorter_than_16")}
    below = {n: e["series_below_patch_min"] for n, e in sorted(entries.items())
             if e.get("series_below_patch_min")}
    manifest = {
        "series_shorter_than_16": {"total": sum(short.values()), "by_dataset": short},
        # GetPatchSize(min_time_patches=2) raises on these series. See patch_min_length.
        "series_below_patch_min": {"total": sum(below.values()), "by_dataset": below},
        "protocol": "train = everything before the val region, cut in time across each source",
        "raw_root": str(raw_root),
        "updated": _dt.datetime.now().isoformat(timespec="seconds"),
        "groups": dict(sorted(groups.items())),
        "datasets": dict(sorted(entries.items())),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, path)
    # pool_guard trains on GIFT-Eval datasets only from a root carrying this marker.
    marker = path.parent / f".pool_{TRAIN_SPLIT_POOL}.json"
    marker.write_text(json.dumps({"name": TRAIN_SPLIT_POOL, "role": "training"}))


def main(argv=None) -> int:
    import datasets

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw_root", type=Path, default=env.GIFT_EVAL_DATA,
                   help="Raw GIFT-Eval benchmark root. Default: $GIFT_EVAL_DATA.")
    p.add_argument("--out_root", type=Path, default=env.PRETRAIN_DATA,
                   help="Training pool root. Default: $PRETRAIN_DATA.")
    p.add_argument("--datasets", nargs="+", default=GIFT_EVAL_55,
                   help="Subset of the 55 configs to write. Default: all. The cut always "
                        "uses every variant of each source.")
    p.add_argument("--min_length", type=int, default=1,
                   help="Drop series with fewer kept points. Default: 1, which keeps short series.")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace existing outputs that do not match this build.")
    p.add_argument("--dry_run", action="store_true",
                   help="Compute cuts and counts without writing anything.")
    args = p.parse_args(argv)
    if args.min_length < 1:
        p.error("--min_length must be >= 1")

    raw_root, out_root = args.raw_root.resolve(), args.out_root.resolve()
    if out_root == raw_root or raw_root in out_root.parents or out_root in raw_root.parents:
        p.error(f"out_root {out_root} and the benchmark root {raw_root} must not nest")
    unknown = sorted(set(args.datasets) - set(GIFT_EVAL_55))
    if unknown:
        p.error(f"not GIFT-Eval dataset configs: {unknown}")

    terms = load_tasks()
    all_groups = source_groups()
    manifest_path = out_root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    entries, groups = manifest.get("datasets", {}), manifest.get("groups", {})
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    wanted = list(dict.fromkeys(args.datasets))
    for g in dict.fromkeys(source_group(n) for n in wanted):
        members = all_groups[g]
        missing = [n for n in members if not (raw_root / n / "state.json").is_file()]
        if missing:
            raise FileNotFoundError(f"source {g}: the cut needs every variant, missing raw "
                                    f"{[str(raw_root / n) for n in missing]}")
        raws = {n: datasets.load_from_disk(str(raw_root / n)) for n in members}
        cuts = group_cuts(members, raws, terms)
        groups[g] = cuts["summary"]
        if len(members) > 1:
            print(f"source {g}: {cuts['summary']['mode']} cut, earliest val start "
                  f"{cuts['summary']['earliest_val_start']}, margin "
                  f"{cuts['summary']['margin_days']} d")
        for n in members:
            if n not in wanted:
                continue
            entries[n] = build_one(n, raws[n], cuts["variants"][n], g, raw_root, out_root,
                                   args.min_length, args.overwrite, args.dry_run, entries)
            if not args.dry_run:  # write after every dataset so a crash leaves a consistent manifest
                write_manifest(manifest_path, raw_root, entries, groups)

    print(f"\n{'dataset':32s} {'series in':>10s} {'out':>8s} {'dropped':>8s} "
          f"{'cut pts':>15s} {'cut days':>17s}")
    for n in wanted:
        e = entries[n]
        days = ("single variant" if e["cut_days_min"] is None else
                f"{e['cut_days_min']:>8.1f}-{e['cut_days_max']:<8.1f}")
        print(f"{n:32s} {e['series_in']:10d} {e['series_out']:8d} {e['series_dropped']:8d} "
              f"{e['cut_points_min']:>7d}-{e['cut_points_max']:<7d} {days}")
    tot = {k: sum(entries[n][k] for n in wanted)
           for k in ("series_in", "series_out", "series_dropped", "series_shorter_than_16",
                     "points_in", "points_out")}
    print(f"TOTAL over {len(wanted)} datasets: {tot}")
    if not args.dry_run:
        print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
