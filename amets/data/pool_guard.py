#  SPDX-License-Identifier: Apache-2.0

"""Refuse to train on GIFT-Eval benchmark data.

Does not import torch. ``GiftEvalDatasetBuilder.load_dataset`` and
``regime_predictor/train.py`` call it before loading any data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from amets.common.env import env  # noqa: F401  (sets GIFT_EVAL_DATA if unset)
from amets.data.gift_eval_split import CUT_RULE, GIFT_EVAL_55

TRAIN_SPLIT_POOL = "gift-eval-train"
TRAIN_SPLIT_MANIFEST = "gift_eval_train_split_manifest.json"
# name -> reason. m5_L* is matched by prefix.
BANNED_DATASETS = {
    "gifteval": "is built from GIFT-Eval test windows",
    "car_parts_with_missing": "is dropped from training and cannot be leak-checked",
    "m5_L*": "is one of the M5 aggregates. Train on item-level lotsa_m5 instead",
}

_FIX_ROOT = ("Point data.storage_path at $PRETRAIN_DATA and fill it with training "
             "pools only, built by scripts/build_train_split.py, "
             "scripts/build_synthetic_dataset.py, scripts/download_lotsa.py and "
             "scripts/download_chronos.py.")


def _read_marker(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"Unreadable pool marker {path}: {e}. Rebuild that pool "
                         f"into a clean directory.") from None


def check_training_root(storage_path: Path | str, datasets: Iterable[str]) -> None:
    """Raise ValueError unless ``storage_path`` is a leak-free training root.

    It refuses:

    - any dataset in ``BANNED_DATASETS``.
    - a root equal to, inside, or containing GIFT_EVAL_DATA, or a dataset
      directory that resolves into it.
    - any ``.pool_*.json`` marker whose role is not ``training``.
    - GIFT-Eval-55 datasets, unless the root has the ``gift-eval-train`` marker
      and a train-split manifest that lists them.
    """
    datasets = list(datasets)
    root = Path(storage_path).resolve()

    for d in datasets:
        key = "m5_L*" if d.startswith("m5_L") else d
        if key in BANNED_DATASETS:
            raise ValueError(f"{d!r} {BANNED_DATASETS[key]}. Remove it from data.datasets "
                             f"and weight_map.")

    benches = {var: Path(os.environ[var]).resolve()
               for var in ("GIFT_EVAL_DATA", "GIFT_EVAL") if os.environ.get(var)}
    for var, bench in benches.items():
        if root == bench or bench in root.parents or root in bench.parents:
            raise ValueError(f"storage_path {root} overlaps the GIFT-Eval benchmark "
                             f"root ${var}={bench} (val/test regions). {_FIX_ROOT}")
        for d in datasets:
            target = (root / d).resolve()
            if target == bench or bench in target.parents:
                raise ValueError(f"{root / d} resolves to {target}, inside the GIFT-Eval "
                                 f"benchmark root ${var}. Replace the link with the "
                                 f"{TRAIN_SPLIT_POOL} pool. {_FIX_ROOT}")

    markers = {p.name: _read_marker(p) for p in sorted(root.glob(".pool_*.json"))}
    wrong = {n: m.get("role") for n, m in markers.items() if m.get("role") != "training"}
    if wrong:
        raise ValueError(f"storage_path {root} holds non-training pools {wrong}. "
                         f"Delete them from this directory. {_FIX_ROOT}")

    gift = [d for d in datasets if d in GIFT_EVAL_55]
    if not gift:
        return
    marker = markers.get(f".pool_{TRAIN_SPLIT_POOL}.json")
    if not marker or marker.get("name") != TRAIN_SPLIT_POOL:
        raise ValueError(f"GIFT-Eval datasets requested but storage_path {root} has no "
                         f".pool_{TRAIN_SPLIT_POOL}.json marker. {_FIX_ROOT}")
    manifest_path = root / TRAIN_SPLIT_MANIFEST
    try:
        listed = json.loads(manifest_path.read_text())["datasets"]
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError(f"No readable {manifest_path}. Rebuild the {TRAIN_SPLIT_POOL} "
                         f"pool with scripts/build_train_split.py.") from None
    stale = [d for d in gift if d in listed and listed[d].get("cut_rule") != CUT_RULE]
    if stale:
        raise ValueError(f"{stale} in {manifest_path} were cut by a different rule (cut_rule "
                         f"is not {CUT_RULE!r}) and may overlap other frequency variants' "
                         f"val/test periods. Rebuild them with scripts/build_train_split.py.")
    missing = [d for d in gift if d not in listed]
    if missing:
        raise ValueError(f"{missing} are not in {manifest_path}, so they may still hold "
                         f"val/test points. Build them with scripts/build_train_split.py "
                         f"--datasets ..., or remove them from data.datasets.")
