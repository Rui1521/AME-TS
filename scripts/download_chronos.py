#!/usr/bin/env python3
"""Download Chronos training datasets from HuggingFace into $PRETRAIN_DATA.

The script converts each config of ``autogluon/chronos_datasets`` to the schema
that GiftEvalDatasetBuilder and HuggingFaceDatasetIndexer expect:
  * ``item_id``: string.
  * ``start``: timestamp[s].
  * ``freq``: string.
  * ``target``: sequence of float32.
It saves the result as ``$PRETRAIN_DATA/chronos_<config>``.

Chronos stores ``id``, a per-point ``timestamp`` sequence and ``target``. For a
config listed in PLACEHOLDER_TIME, the timestamps carry no meaning, and the
script replaces them with a fixed start and freq. KernelSynth is synthetic
Gaussian-process data and gets start 2000-01-01 and freq "H". In
DefaultPatchSizeConstraints, hourly data gets patch sizes 4..64 and a minimum
length of 8, which suits its 1024-point series. The descriptor predictor does
not read freq. For other configs, start is the first timestamp, and freq comes
from the first series with a regular timestamp sequence. The script retries HF
errors, such as 429 rate limits, with backoff. Set HF_TOKEN for authenticated
limits.

Usage (from the repository root):
    python scripts/download_chronos.py                                   # KernelSynth
    python scripts/download_chronos.py --only training_corpus_kernel_synth_1m
    python scripts/download_chronos.py --only dominick ercot exchange_rate mexico_city_bikes
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amets.common.env import env  # noqa: E402
from download_lotsa import with_retry  # noqa: E402

REPO = "autogluon/chronos_datasets"
DATASETS = ["training_corpus_kernel_synth_1m"]
PLACEHOLDER_TIME = {"training_corpus_kernel_synth_1m": ("2000-01-01", "H")}


def features():
    from datasets import Features, Sequence, Value

    return Features({"item_id": Value("string"), "start": Value("timestamp[s]"),
                     "freq": Value("string"), "target": Sequence(Value("float32"))})


def infer_freq(ds, max_rows: int = 20):
    """Return the freq of the first of ``max_rows`` series whose first 64 timestamps are regular."""
    import pandas as pd

    for i in range(min(len(ds), max_rows)):
        ts = pd.DatetimeIndex(ds[i]["timestamp"][:64])
        freq = pd.infer_freq(ts) if len(ts) >= 3 else None
        if freq is not None:
            return freq
    return None


def convert(ds, config: str, cache_file: str | None = None):
    """Convert a Chronos-format ``datasets.Dataset`` to the training schema in the module docstring."""
    import pandas as pd

    cols = set(ds.column_names)
    id_col = "id" if "id" in cols else "item_id" if "item_id" in cols else None
    if "target" not in cols:
        raise ValueError(f"{config}: no 'target' column (columns: {sorted(cols)})")
    if config in PLACEHOLDER_TIME:
        start, freq = pd.Timestamp(PLACEHOLDER_TIME[config][0]), PLACEHOLDER_TIME[config][1]
        use_ts = False
    else:
        if "timestamp" not in cols:
            raise ValueError(f"{config}: no 'timestamp' column and no placeholder time")
        freq = infer_freq(ds)
        if freq is None:
            raise ValueError(f"{config}: cannot infer freq from the first series")
        use_ts = True

    def fn(batch, idx):
        n = len(batch["target"])
        return {
            "item_id": [str(batch[id_col][k]) if id_col else f"{config}_{idx[k]}"
                        for k in range(n)],
            "start": ([pd.Timestamp(t[0]) for t in batch["timestamp"]] if use_ts
                      else [start] * n),
            "freq": [freq] * n,
            "target": [np.asarray(t, dtype=np.float32) for t in batch["target"]],
        }

    return ds.map(fn, batched=True, batch_size=1000, with_indices=True,
                  remove_columns=ds.column_names, features=features(),
                  cache_file_name=cache_file, desc=f"convert {config}")


def main(argv=None) -> int:
    from datasets import load_dataset

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--only", nargs="+", default=None,
                   help=f"Chronos configs to fetch. Default: {DATASETS}.")
    p.add_argument("--repo", default=REPO, help="Hugging Face dataset repository.")
    p.add_argument("--attempts", type=int, default=6,
                   help="Number of HF load attempts, with backoff between them.")
    args = p.parse_args(argv)

    out_root = Path(env.PRETRAIN_DATA)
    out_root.mkdir(parents=True, exist_ok=True)
    for config in args.only or DATASETS:
        out = out_root / f"chronos_{config}"
        if out.exists():
            print(f"SKIP {config}: {out} already exists")
            continue
        print(f"Downloading {args.repo} / {config} ...")
        ds = with_retry(lambda: load_dataset(args.repo, config, split="train"),
                        f"load_dataset({args.repo!r}, {config!r}, split='train')",
                        attempts=args.attempts)
        tmp = out_root / f".tmp-chronos-{config}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        try:
            conv = convert(ds, config, cache_file=str(tmp / "convert.arrow"))
            conv.save_to_disk(str(tmp / "data"))
            del conv
            os.replace(tmp / "data", out)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"  Saved {len(ds)} series to {out}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
