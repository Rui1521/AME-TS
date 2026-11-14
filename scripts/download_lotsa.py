#!/usr/bin/env python3
"""Download LOTSA datasets from HuggingFace into $PRETRAIN_DATA as ``lotsa_<config>``.

Each config directory of ``Salesforce/lotsa_data`` is already a ``save_to_disk``
layout with arrow shards, ``dataset_info.json`` and ``state.json``. The script
fetches the files as they are and checks them with ``datasets.load_from_disk``,
so there is one copy on disk and no conversion. A child process fetches each
file under a timeout that scales with file size, and the script kills a stalled
transfer. The script retries errors and timeouts, such as 429 rate limits, with
backoff, and the retries use plain HTTP instead of hf_xet. Set HF_TOKEN for
authenticated limits. Training pools never go into GIFT_EVAL_DATA.

Usage (from the repository root):
    python scripts/download_lotsa.py                       # every dataset listed below
    python scripts/download_lotsa.py --only m5 PEMS04      # only these configs
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amets.common.env import env  # noqa: E402

REPO = "Salesforce/lotsa_data"

# Every LOTSA dataset that configs/data/ame_*.yaml or regime_predictor/train.py reads.
DATASETS = [
    "traffic_hourly", "PEMS03", "PEMS04", "PEMS07", "PEMS08", "PEMS_BAY",
    "weather", "oikolab_weather", "subseasonal", "favorita_sales", "monash_m3_monthly",
    "tourism_monthly", "tourism_quarterly", "tourism_yearly", "fred_md",
    "nn5_daily_with_missing", "australian_electricity_demand", "wind_power",
    "alibaba_cluster_trace_2018", "london_smart_meters_with_missing",
    "extended_web_traffic_with_missing", "m1_monthly", "m1_quarterly", "m1_yearly",
    "buildings_900k", "era5_2015", "era5_2016", "era5_2017", "era5_2018",
    "uber_tlc_hourly", "pedestrian_counts", "solar_power", "wind_farms_with_missing",
    "m5",
]


def with_retry(fn, what: str, attempts: int = 6, base_s: float = 60.0):
    """Call ``fn()``. After an exception, wait base_s * 2**k seconds, capped at 15 min, and retry."""
    for k in range(attempts):
        try:
            return fn()
        except Exception as e:  # HF raises several types: HTTP 429/5xx, timeouts
            if k == attempts - 1:
                raise RuntimeError(f"{what} failed after {attempts} attempts: "
                                   f"{type(e).__name__}: {e}") from e
            wait = min(base_s * 2 ** k, 900.0)
            print(f"  {what}: {type(e).__name__}: {str(e)[:200]}. Retrying in {wait:.0f}s",
                  flush=True)
            time.sleep(wait)


_FETCH = ("import sys; from huggingface_hub import hf_hub_download; "
          "hf_hub_download(sys.argv[1], sys.argv[2], repo_type='dataset', local_dir=sys.argv[3])")


def list_files(repo: str, config: str) -> list:
    """Return (path, size) for every file under ``config/`` in the dataset repo."""
    from huggingface_hub import HfApi

    tree = with_retry(lambda: list(HfApi().list_repo_tree(repo, path_in_repo=config,
                                                          repo_type="dataset", recursive=True)),
                      f"list {config}")
    return [(f.path, getattr(f, "size", 0) or 0) for f in tree if hasattr(f, "size")]


def fetch_file(repo: str, path: str, size: int, local_dir: Path, attempts: int = 6,
               base_timeout_s: float = 300.0) -> None:
    """Download one file in a child process, killed after base_timeout_s + size / (2 MB/s)."""
    timeout = base_timeout_s + size / 2e6
    state = {"k": 0}

    def once():
        env = dict(os.environ)
        if state["k"] > 0:
            env["HF_HUB_DISABLE_XET"] = "1"
        state["k"] += 1
        with tempfile.TemporaryFile() as err:   # a file, not a pipe, so a killed child cannot block the parent
            p = subprocess.Popen([sys.executable, "-c", _FETCH, repo, path, str(local_dir)],
                                 env=env, stdout=subprocess.DEVNULL, stderr=err,
                                 start_new_session=True)
            try:
                rc = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                raise
            if rc != 0:
                err.seek(0)
                raise RuntimeError(err.read().decode(errors="replace").strip()[-500:])

    with_retry(once, f"download {path}", attempts=attempts)


def download(config: str, out_root: Path, repo: str = REPO, workers: int = 8) -> Path:
    """Fetch one LOTSA config to ``out_root/lotsa_<config>`` and return the path."""
    import datasets

    out = out_root / f"lotsa_{config}"
    if out.exists():
        print(f"SKIP {config}: {out} already exists")
        return out
    stage = out_root / ".tmp-lotsa"            # on the same filesystem, so the final move is a rename
    files = list_files(repo, config)
    print(f"Downloading {repo} / {config} ({len(files)} files, "
          f"{sum(s for _, s in files) / 1e9:.2f} GB) ...", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for f in [ex.submit(fetch_file, repo, path, size, stage) for path, size in files]:
            f.result()
    src = stage / config
    if not (src / "state.json").is_file():
        raise RuntimeError(f"{repo}/{config}: no state.json after download, so this is not a "
                           f"save_to_disk layout. Files: {sorted(p.name for p in src.glob('*'))}")
    ds = datasets.load_from_disk(str(src))
    info = json.loads((src / "dataset_info.json").read_text())
    expected = sum(s["num_examples"] for s in (info.get("splits") or {}).values())
    if expected and len(ds) != expected:
        raise RuntimeError(f"{config}: {len(ds)} rows, dataset_info says {expected}")
    del ds
    os.replace(src, out)
    print(f"  Saved {config} to {out} ({expected or '?'} rows)", flush=True)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--only", nargs="+", default=None,
                   help="Download only these LOTSA configs. Default: every config in DATASETS.")
    p.add_argument("--repo", default=REPO, help="Hugging Face dataset repository.")
    args = p.parse_args(argv)
    out_root = Path(env.PRETRAIN_DATA)
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        for config in args.only or DATASETS:
            download(config, out_root, args.repo)
    finally:
        shutil.rmtree(out_root / ".tmp-lotsa", ignore_errors=True)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
