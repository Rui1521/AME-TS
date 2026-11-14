"""GIFT-Eval scoring.

The score is the geometric mean over the 97 GIFT-Eval tasks of
``model MASE[0.5] / seasonal-naive MASE[0.5]``. The math follows
``evaluation/eval_utils.py::generate_comprehensive_comparison_table``: a ratio per
task, clipped at 1e-10, then the exp of the mean log. Row selection differs. The
input CSV must already hold exactly one selected row per task, and this module
picks nothing.

Guards that eval_utils.py does not apply:
  * A missing seasonal-naive file raises an error. There is no raw-mean fallback.
  * Any canonical task that is missing, duplicated, or non-finite makes the result
    partial, and a partial result carries no score.

Val scores use the same test-window seasonal-naive constants for normalization.
A per-task constant does not change the order of runs, so val scores can rank
runs against each other. A val score is not a test score, and its label says so.

CLI:
    python scripts/score.py --csv test_valbest.csv --split test
    python scripts/score.py --csv val_best.csv --split val --json-out s.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
MASE_COL = "eval_metrics/MASE[0.5]"
EXPECTED_N_TASKS = 97
# Rows each split may score: test only val-selected rows, val only grid rows.
ALLOWED_SELECTION = {"test": {"val_rank1", "val_top5_median"}, "val": {"grid"}}
SPLIT_LABELS = {
    "test": "test",
    "val": "val (normalized by test-window seasonal naive, for ranking only)",
}


def default_seasonal_naive_path() -> Path:
    """Return SEASONAL_NAIVE_RESULTS if set, else the gift-eval checkout's seasonal-naive CSV."""
    explicit = os.environ.get("SEASONAL_NAIVE_RESULTS")
    if explicit:
        return Path(explicit)
    repo = Path(os.environ.get("GIFT_EVAL_REPO", REPO_ROOT / "gift-eval"))
    return repo / "results" / "seasonal_naive" / "all_results.csv"


def _float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_seasonal_naive(path, expected_n_tasks: int = EXPECTED_N_TASKS) -> Dict[str, float]:
    """Map each canonical task to its seasonal-naive MASE. Raise on any malformed input."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Seasonal-naive results not found: {path}. Set SEASONAL_NAIVE_RESULTS or "
            "GIFT_EVAL_REPO. Scoring without it is not supported.")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    sn: Dict[str, float] = {}
    for r in rows:
        task = r["dataset"]
        if task in sn:
            raise ValueError(f"Duplicate task in seasonal-naive file: {task}")
        v = _float(r[MASE_COL])
        if not (math.isfinite(v) and v > 0):
            raise ValueError(f"Seasonal-naive MASE for {task} is not a positive number: {v}")
        sn[task] = v
    if expected_n_tasks is not None and len(sn) != expected_n_tasks:
        raise ValueError(
            f"Seasonal-naive file has {len(sn)} tasks, expected {expected_n_tasks}: {path}")
    return sn


def score(csv_path, seasonal_naive_csv=None, split: str = "test",
          expected_n_tasks: int = EXPECTED_N_TASKS) -> dict:
    """Score one results CSV holding one selected row per task.

    Returns a dict with these keys, among others:
      * ``score``: the geometric mean, or None when the result is partial.
      * ``partial``: True if any canonical task is missing, duplicated, or non-finite.
      * ``n_tasks``: the number of valid canonical tasks.
      * ``missing_tasks``, ``duplicate_tasks``, ``nonfinite_tasks``: task lists.
      * ``extra_tasks``: non-canonical tasks, which the score ignores.
      * ``per_task``: the ratio for each valid task.
    """
    if split not in SPLIT_LABELS:
        raise ValueError(f"split must be one of {sorted(SPLIT_LABELS)}, got {split!r}")
    sn_path = Path(seasonal_naive_csv) if seasonal_naive_csv else default_seasonal_naive_path()
    sn = load_seasonal_naive(sn_path, expected_n_tasks)

    with Path(csv_path).open(newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    for col in ("dataset", MASE_COL, "split", "selection"):
        if col not in header:
            raise ValueError(f"{csv_path} has no '{col}' column")
    wrong = sorted({r["split"] for r in rows} - {split})
    if wrong:
        raise ValueError(f"{csv_path} contains rows with split {wrong}, but the requested split is {split!r}")
    kinds = sorted({r["selection"] for r in rows})
    if len(kinds) > 1 or not set(kinds) <= ALLOWED_SELECTION[split]:
        raise ValueError(
            f"{csv_path}: selection values {kinds}. A {split} score needs rows of one kind "
            f"from {sorted(ALLOWED_SELECTION[split])}, and test rows must be val-selected.")

    values: Dict[str, List[float]] = {}
    for r in rows:
        values.setdefault(r["dataset"], []).append(_float(r[MASE_COL]))

    missing = sorted(t for t in sn if t not in values)
    duplicate = sorted(t for t in sn if len(values.get(t, [])) > 1)
    nonfinite = sorted(t for t in sn if len(values.get(t, [])) == 1
                       and not math.isfinite(values[t][0]))
    extra = sorted(t for t in values if t not in sn)

    per_task: Dict[str, float] = {}
    for t in sn:
        v = values.get(t, [])
        if len(v) == 1 and math.isfinite(v[0]):
            per_task[t] = v[0] / sn[t]

    partial = bool(missing or duplicate or nonfinite)
    value: Optional[float] = None
    if not partial:
        logs = [math.log(max(r, 1e-10)) for r in per_task.values()]
        value = math.exp(sum(logs) / len(logs))

    return {
        "split": split,
        "selection": kinds[0] if kinds else None,
        "label": SPLIT_LABELS[split] + (" [PARTIAL: no score]" if partial else ""),
        "score": value,
        "partial": partial,
        "n_tasks": len(per_task),
        "n_expected": len(sn),
        "missing_tasks": missing,
        "duplicate_tasks": duplicate,
        "nonfinite_tasks": nonfinite,
        "extra_tasks": extra,
        "per_task": per_task,
        "csv": str(csv_path),
        "seasonal_naive_csv": str(sn_path),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", required=True, help="Results CSV with one selected row per task.")
    p.add_argument("--split", required=True, choices=sorted(SPLIT_LABELS))
    p.add_argument("--seasonal-naive", default=None,
                   help="Seasonal-naive results CSV. Default: $SEASONAL_NAIVE_RESULTS or "
                        "<gift-eval>/results/seasonal_naive/all_results.csv.")
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    res = score(args.csv, args.seasonal_naive, args.split)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(res, indent=2))
    if res["partial"]:
        print(f"{res['label']}: {res['n_tasks']}/{res['n_expected']} tasks valid, "
              f"missing={len(res['missing_tasks'])} duplicate={len(res['duplicate_tasks'])} "
              f"nonfinite={len(res['nonfinite_tasks'])}")
        return 2
    print(f"{res['label']}: normalized MASE = {res['score']:.4f} over {res['n_tasks']} tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
