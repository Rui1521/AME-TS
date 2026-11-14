"""Per-task inference-config selection on the validation split.

Reads one or more val CSVs written by ``scripts/evaluate.py --split val``
(one row per task x max_context x eval_mode x patch_size), ranks candidates per
task by val MASE[0.5], and writes ``selection.json``:

    {"selected_on": "val", "metric": ..., "source_csvs": [...],
     "checkpoint": {"path": ..., "sha256": ...},
     "tasks": {task: {"rank1": cfg, "top5": [cfg, ...],
                      "val_n_series": int, "val_n_kept": int}},
     "missing_tasks": [...], "ignored_tasks": [...]}

with ``cfg = {"max_context", "eval_mode", "patch_size", "val_mase"}``. ``top5`` is
restricted to the rank-1 config's eval_mode. Univariate and multivariate
forecasts cover different items, so their point forecasts cannot be combined.

The script selects only the 97 canonical tasks listed in the seasonal-naive file.
Other tasks in the CSV go to ``ignored_tasks``. Canonical tasks with no finite val
result go to ``missing_tasks`` and are printed. Any row whose split is not "val"
or whose selection is not "grid" is an error, because this script must never see
test numbers. All rows must carry the same checkpoint_sha256. The script records
it, and evaluate.py --selection refuses any other checkpoint.

With ``--val-best-csv``, the script also writes the rank-1 val rows, one per task,
for ``score.py --split val``.

CLI:
    python scripts/select_configs.py --val-csv val_all.csv \
        --out selection.json --val-best-csv val_best.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score import (  # noqa: E402
    EXPECTED_N_TASKS, MASE_COL, default_seasonal_naive_path, load_seasonal_naive)

TOP_K = 5
CONFIG_COLS = ("max_context", "eval_mode", "patch_size")
META_COLS = ("selection", "n_series", "n_eval_series", "checkpoint_sha256")


def _read(paths) -> Tuple[List[str], List[dict]]:
    header: List[str] = []
    rows: List[dict] = []
    for p in paths:
        with Path(p).open(newline="") as f:
            reader = csv.DictReader(f)
            h = reader.fieldnames or []
            if header and h != header:
                raise ValueError(f"Header of {p} differs from {paths[0]}")
            header = h
            rows.extend(reader)
    for col in ("dataset", "split", MASE_COL) + CONFIG_COLS + META_COLS:
        if col not in header:
            raise ValueError(f"Val CSV has no '{col}' column (written by evaluate.py --split val)")
    bad = sorted({r["split"] for r in rows} - {"val"})
    if bad:
        raise ValueError(f"Val CSVs contain rows with split {bad}. Selection may use val only.")
    bad = sorted({r["selection"] for r in rows} - {"grid"})
    if bad:
        raise ValueError(f"Val CSVs contain non-grid rows ({bad}). Pass the val grid CSVs only.")
    return header, rows


def select(val_csvs, seasonal_naive_csv=None, top_k: int = TOP_K,
           expected_n_tasks: int = EXPECTED_N_TASKS,
           checkpoint_path: str = None) -> Tuple[dict, Dict[str, dict]]:
    """Return the selection dict and a map from each task to its rank-1 CSV row."""
    val_csvs = [str(p) for p in val_csvs]
    canonical = load_seasonal_naive(
        seasonal_naive_csv or default_seasonal_naive_path(), expected_n_tasks)
    _, rows = _read(val_csvs)
    shas = sorted({r["checkpoint_sha256"] for r in rows})
    if len(shas) != 1 or not shas[0]:
        raise ValueError(f"Val rows must come from exactly one checkpoint. Got sha256s {shas}")

    by_task: Dict[str, Dict[tuple, dict]] = {}
    for r in rows:
        try:
            mase = float(r[MASE_COL])
        except ValueError:
            continue
        if not math.isfinite(mase):
            continue
        try:
            key = (int(float(r["max_context"])), r["eval_mode"], int(float(r["patch_size"])))
        except ValueError:
            raise ValueError(f"Bad max_context/patch_size in val row for {r['dataset']}: "
                             f"{r['max_context']!r}, {r['patch_size']!r}") from None
        cands = by_task.setdefault(r["dataset"], {})
        if key in cands:
            raise ValueError(
                f"Duplicate val result for {r['dataset']} {dict(zip(CONFIG_COLS, key))}. "
                "Rerun into a fresh output dir instead of appending.")
        cands[key] = dict(r, _mase=mase)

    tasks: Dict[str, dict] = {}
    best_rows: Dict[str, dict] = {}
    for task in sorted(canonical):
        cands = by_task.get(task)
        if not cands:
            continue
        # Sort by val MASE. The config tuple breaks exact ties.
        ranked = sorted(cands.items(), key=lambda kv: (kv[1]["_mase"], kv[0]))
        cfg = lambda k, r: {"max_context": k[0], "eval_mode": k[1],
                            "patch_size": k[2], "val_mase": r["_mase"]}
        (k1, r1) = ranked[0]
        top = [cfg(k, r) for k, r in ranked if k[1] == k1[1]][:top_k]
        tasks[task] = {"rank1": cfg(k1, r1), "top5": top,
                       "val_n_series": int(r1["n_series"]),
                       "val_n_kept": int(r1["n_eval_series"])}
        best_rows[task] = {k: v for k, v in r1.items() if k != "_mase"}

    selection = {
        "selected_on": "val",
        "metric": MASE_COL,
        "top_k": top_k,
        "source_csvs": val_csvs,
        "checkpoint": {"path": checkpoint_path, "sha256": shas[0]},
        "n_tasks": len(tasks),
        "n_expected": len(canonical),
        "tasks": tasks,
        "missing_tasks": sorted(t for t in canonical if t not in tasks),
        "ignored_tasks": sorted(t for t in by_task if t not in canonical),
    }
    return selection, best_rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Select per-task inference configs on val.")
    p.add_argument("--val-csv", nargs="+", required=True,
                   help="Val grid CSVs written by evaluate.py --split val.")
    p.add_argument("--out", required=True, help="Output path for selection.json.")
    p.add_argument("--val-best-csv", default=None, help="Output path for the rank-1 val rows.")
    p.add_argument("--seasonal-naive", default=None)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint path to record. Its sha256 must match the val rows.")
    args = p.parse_args(argv)

    selection, best_rows = select(args.val_csv, args.seasonal_naive, args.top_k,
                                  checkpoint_path=args.checkpoint)
    if args.checkpoint:
        import hashlib
        h = hashlib.sha256()
        with open(args.checkpoint, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != selection["checkpoint"]["sha256"]:
            print(f"ERROR: {args.checkpoint} sha256 differs from the val rows' checkpoint")
            return 3
    Path(args.out).write_text(json.dumps(selection, indent=2))
    if args.val_best_csv:
        header, _ = _read(args.val_csv)
        with open(args.val_best_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for task in sorted(best_rows):
                w.writerow(best_rows[task])

    short = [t for t, s in selection["tasks"].items() if len(s["top5"]) < args.top_k]
    print(f"Selected {selection['n_tasks']}/{selection['n_expected']} tasks -> {args.out}")
    if short:
        print(f"  {len(short)} tasks have fewer than {args.top_k} same-mode candidates, "
              f"and their ensembles use the candidates that exist: {short}")
    dropped = {t: (s["val_n_series"], s["val_n_kept"]) for t, s in selection["tasks"].items()
               if s["val_n_kept"] < s["val_n_series"]}
    for t, (n, k) in sorted(dropped.items()):
        print(f"  {t}: val uses {k}/{n} series, the rest are too short for a val window")
    if selection["ignored_tasks"]:
        print(f"  ignored {len(selection['ignored_tasks'])} non-canonical tasks")
    if selection["missing_tasks"]:
        print(f"  NO VALID VAL RESULT for {len(selection['missing_tasks'])} tasks:")
        for t in selection["missing_tasks"]:
            print(f"    {t}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
