#!/usr/bin/env python3
"""
Print all GIFT-Eval dataset names, one per line.

Reads dataset_properties.json from the gift-eval checkout, so the list matches
the benchmark you evaluate against. The repository README describes the setup.

Usage:
    python stats/get_all_datasets.py

    # Feed the full list to the evaluation script:
    python scripts/evaluate.py \
        --checkpoint path/to/checkpoint.ckpt \
        --split val \
        --datasets $(python stats/get_all_datasets.py) \
        --output_dir results/full_eval
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amets.common.env import dataset_properties_path  # noqa: E402


def main():
    try:
        properties_path = dataset_properties_path()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    with open(properties_path) as f:
        props = json.load(f)

    # One per line, for shell substitution
    for ds_name in sorted(props.keys()):
        print(ds_name)


if __name__ == "__main__":
    main()
