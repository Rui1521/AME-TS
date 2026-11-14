#!/usr/bin/env python3
"""Build the "synthetic" dataset that the AME-TS data configs
(configs/data/ame_*.yaml) list as a training source.

Writes an HF-Arrow dataset to $PRETRAIN_DATA/synthetic with the
{item_id, start, freq, target} schema that GiftEvalDatasetBuilder expects.
Each series comes from one of seven generators: pure sine, multi-seasonal sine,
linear trend, sine with trend, sparse, white noise, and random walk. Together
they cover all 4 structural descriptors.

Usage (from the repository root):
    python scripts/build_synthetic_dataset.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from datasets import Dataset, Features, Sequence, Value

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amets.common.env import env  # noqa: E402


def gen_pure_sine(n: int) -> np.ndarray:
    period = np.random.choice([7, 12, 14, 24, 30, 50])
    amp = np.random.uniform(0.5, 5.0)
    phase = np.random.uniform(0, 2 * np.pi)
    t = np.arange(n, dtype=np.float64)
    series = amp * np.sin(2 * np.pi * t / period + phase)
    return series + np.random.randn(n) * np.random.uniform(0.0, 0.1) * amp


def gen_multi_seasonal(n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    series = np.zeros(n)
    for _ in range(np.random.randint(2, 4)):
        period = np.random.choice([7, 12, 24, 30, 52])
        series += np.random.uniform(0.5, 3.0) * np.sin(
            2 * np.pi * t / period + np.random.uniform(0, 2 * np.pi))
    return series + np.random.randn(n) * 0.1


def gen_linear_trend(n: int) -> np.ndarray:
    t = np.linspace(0, 1, n)
    series = np.random.uniform(-5, 5) * t + np.random.uniform(-10, 10)
    return series + np.random.randn(n) * np.random.uniform(0.0, 0.5)


def gen_sine_with_trend(n: int) -> np.ndarray:
    period = np.random.choice([7, 12, 24, 30])
    t = np.arange(n, dtype=np.float64)
    series = np.random.uniform(1.0, 5.0) * np.sin(2 * np.pi * t / period)
    series += np.random.uniform(1, 10) * np.linspace(0, 1, n)
    return series + np.random.randn(n) * 0.1


def gen_sparse(n: int) -> np.ndarray:
    sparsity = np.random.uniform(0.7, 0.98)
    series = np.zeros(n)
    mask = np.random.rand(n) > sparsity
    series[mask] = np.random.randn(mask.sum()) * np.random.uniform(1, 10)
    return series


def gen_white_noise(n: int) -> np.ndarray:
    return np.random.randn(n) * np.random.uniform(0.1, 10.0)


def gen_random_walk(n: int) -> np.ndarray:
    steps = np.random.randn(n) * np.random.uniform(0.1, 2.0)
    return np.cumsum(steps)


GENERATORS = [
    gen_pure_sine, gen_multi_seasonal, gen_linear_trend,
    gen_sine_with_trend, gen_sparse, gen_white_noise, gen_random_walk,
]


def build(num_series: int, seq_lens: list[int], seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    rows = []
    for i in range(num_series):
        gen = GENERATORS[rng.integers(0, len(GENERATORS))]
        n = int(rng.choice(seq_lens))
        target = gen(n).astype(np.float32).tolist()
        rows.append({
            "item_id": f"synthetic_{i}",
            "start": pd.Timestamp("2020-01-01"),
            "freq": "D",
            "target": target,
        })
    features = Features({
        "item_id": Value("string"),
        "start": Value("timestamp[s]"),
        "freq": Value("string"),
        "target": Sequence(Value("float32")),
    })
    return Dataset.from_list(rows, features=features)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--num_series", type=int, default=87000,
                   help="Number of series to generate.")
    p.add_argument("--seq_lens", type=int, nargs="+",
                   default=[64, 96, 128, 192, 336, 512],
                   help="Series lengths to draw from.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = p.parse_args(argv)

    out_dir = os.path.join(str(env.PRETRAIN_DATA), "synthetic")
    ds = build(args.num_series, args.seq_lens, args.seed)
    ds.save_to_disk(out_dir)
    print(f"Saved synthetic dataset: {len(ds)} series -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
