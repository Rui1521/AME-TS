#!/usr/bin/env python3
"""
Train the structural-descriptor predictors (forecastability, seasonality, trend).

The script:
  * Crops windows from the training pool under PRETRAIN_DATA.
  * Computes ground-truth descriptors on each unpadded window with the feature pipeline.
  * Splits items per source group into train, early-stop and gate sets.
  * Quantile-normalizes targets against train-split references.
  * Trains one CNN per descriptor.

Model inputs take the same path as in AME-TS pre-training, ``stack_contexts``
followed by a masked batched forward pass, so padding has no effect.
Sparsity is analytic and not trained.

Usage:
    python regime_predictor/train.py                    # all three descriptors
    python regime_predictor/train.py --feature trend    # one descriptor, same cache

Outputs:
    $REGIME_CACHE_DIR/label_cache_seed{seed}.npz   windows, raw labels, item split
    <output_dir>/regime_predictor_{descriptor}.pt
    <output_dir>/regime_predictor_combined.pt      written once all three share one cache
--output_dir defaults to regime_predictor/trained_predictors/.
"""

import hashlib
import json
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Put the repository root and this directory on sys.path.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from feature_computation.feature_pipeline import TimeSeriesFeatureExtractor
# Predictor classes shared with the AMETS model.
from amets.model.combined_regime_predictor import (
    REGIME_MAX_LEN, CombinedRegimePredictor, SingleFeaturePredictor,
    series_regime_probs, stack_contexts,
)
from amets.common.env import env
from amets.data.pool_guard import check_training_root

# ============================================================
# Configuration
# ============================================================

# Read training pools only: the GIFT-Eval train split written by
# scripts/build_train_split.py, synthetic, LOTSA and Chronos. Never read
# GIFT_EVAL_DATA, which holds the val and test regions.
DATA_DIR = env.PRETRAIN_DATA
OUTPUT_DIR = Path(__file__).parent / "trained_predictors"
CODE_FILES = sorted((Path(__file__).parent / "feature_computation").glob("*.py")) + [
    Path(__file__),
    Path(__file__).parent.parent / "amets" / "model" / "combined_regime_predictor.py",
]

FEATURE_NAMES = ["forecastability", "seasonality", "trend", "sparsity"]
# Train CNN predictors for the first 3 descriptors only. Sparsity is analytic at inference.
TRAINABLE_FEATURES = ["forecastability", "seasonality", "trend"]

# Training hyperparameters
BATCH_SIZE = 256
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 15
MIN_WINDOW_LEN = 32  # lower bound of the uniform crop rule
MAX_WINDOW_LEN = REGIME_MAX_LEN  # stage 2 crops contexts to the same length
MIN_CONTEXT_LEN = 4  # shortest stage-2 context: one patch of size 4
PATCH_SIZES = (4, 8, 16, 32, 64, 128)  # model patch sizes (configs/model/*.yaml)
PATCH_CROP_FRAC = 0.5  # share of crops whose length is a patch-size multiple
SAMPLES_PER_DATASET = 3000  # random crops per dataset
# Item split per source group. The group is the name before '/', shared by its variants.
EARLY_STOP_RATIO = 0.15  # early stopping / checkpoint selection
GATE_RATIO = 0.15  # held out for gate metrics only
LENGTH_BUCKETS = [(4, 8), (8, 32), (32, 128), (128, 512), (512, 2049)]  # gate report only

# Training pool. amets/data/pool_guard.py checks it before loading.
DATASETS = [
    # GIFT-Eval train split
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
    "restaurant",
    "hierarchical_sales/D", "hierarchical_sales/W",
    "saugeenday/D", "saugeenday/M", "saugeenday/W",
    "kdd_cup_2018_with_missing/H", "kdd_cup_2018_with_missing/D",
    "bitbrains_fast_storage/5T", "bitbrains_fast_storage/H",
    "bitbrains_rnd/5T", "bitbrains_rnd/H",
    "bizitobs_application", "bizitobs_service",
    "bizitobs_l2c/5T", "bizitobs_l2c/H",
    "synthetic",
    # LOTSA
    "lotsa_traffic_hourly", "lotsa_PEMS04", "lotsa_weather",
    "lotsa_oikolab_weather", "lotsa_subseasonal",
    "lotsa_m5", "lotsa_favorita_sales",
    "lotsa_monash_m3_monthly", "lotsa_tourism_monthly",
    "lotsa_fred_md", "lotsa_nn5_daily_with_missing",
    "lotsa_australian_electricity_demand", "lotsa_wind_power",
    "lotsa_alibaba_cluster_trace_2018",
    "lotsa_PEMS03", "lotsa_PEMS07", "lotsa_PEMS08", "lotsa_PEMS_BAY",
    "lotsa_london_smart_meters_with_missing",
    "lotsa_extended_web_traffic_with_missing",
    "lotsa_tourism_quarterly", "lotsa_tourism_yearly",
    "lotsa_m1_monthly", "lotsa_m1_quarterly", "lotsa_m1_yearly",
    "lotsa_buildings_900k", "lotsa_era5_2015", "lotsa_era5_2016",
    "lotsa_era5_2017", "lotsa_era5_2018",
    "lotsa_uber_tlc_hourly", "lotsa_pedestrian_counts",
    "lotsa_solar_power", "lotsa_wind_farms_with_missing",
    # Chronos
    "chronos_training_corpus_kernel_synth_1m", "chronos_dominick",
    "chronos_ercot", "chronos_exchange_rate", "chronos_mexico_city_bikes",
]


def seed_everything(seed):
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def code_sha256():
    """sha256 over feature_computation/*.py, this file and the predictor module."""
    h = hashlib.sha256()
    for p in CODE_FILES:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


# ============================================================
# Data Generation
# ============================================================

def load_hf_dataset(ds_name):
    """Load a HF dataset from disk.

    Returns a list of (row_index, univariate float32 array, freq or None). The
    variates of a multivariate row share its row index. NaN becomes 0, and series
    shorter than MIN_CONTEXT_LEN are dropped.
    """
    dir_name = ds_name.replace("/", "_")
    for path in [DATA_DIR / dir_name, DATA_DIR / ds_name]:
        if path.exists():
            from datasets import Dataset as HFDataset
            data = HFDataset.load_from_disk(str(path))
            series_list = []
            freqs = data["freq"] if "freq" in data.column_names else [None] * len(data)
            for i in range(len(data)):
                target = data[i]["target"]
                is_multi = (isinstance(target, list) and len(target) > 0
                            and isinstance(target[0], (list, np.ndarray)))
                for variate in (target if is_multi else [target]):
                    arr = np.nan_to_num(np.array(variate, dtype=np.float32), nan=0.0)
                    if len(arr) >= MIN_CONTEXT_LEN:
                        series_list.append((i, arr, freqs[i]))
            return series_list
    return []


def compute_sparsity_unique(ts_window):
    """Compute sparsity as 1 - n_unique / len. A mostly-constant series is sparse."""
    valid = ts_window[~np.isnan(ts_window)]
    if len(valid) == 0:
        return 1.0
    n_unique = len(np.unique(valid))
    return 1.0 - n_unique / len(valid)


def compute_features_single(ts_window):
    """Compute the 4 regime features of a single univariate window."""
    ts_3d = ts_window.reshape(1, -1, 1)
    try:
        extractor = TimeSeriesFeatureExtractor(ts_3d)
        features = extractor.compute_all_features(mode='combined')
        result = np.array([
            float(features['forecastability'][0]),
            float(features['seasonality'][0]),
            float(features['trend'][0]),
            compute_sparsity_unique(ts_window),  # unique-values definition
        ], dtype=np.float32)
        # Replace NaN with 0.5 and clip to [0, 1]
        result = np.clip(np.nan_to_num(result, nan=0.5), 0.0, 1.0)
        return result
    except Exception:
        return None


def sample_crop_len(rng, series_len):
    """Crop length for a series of ``series_len`` >= MIN_CONTEXT_LEN points.

    With probability PATCH_CROP_FRAC, or always if the series is shorter than
    MIN_WINDOW_LEN, the length is k * patch_size, as in pre-training, down to 4
    points. Otherwise it is uniform in [MIN_WINDOW_LEN, min(MAX_WINDOW_LEN, series_len)].
    """
    max_len = min(MAX_WINDOW_LEN, series_len)
    if max_len >= MIN_WINDOW_LEN and rng.random() >= PATCH_CROP_FRAC:
        return int(rng.integers(MIN_WINDOW_LEN, max_len + 1))
    ps = int(rng.choice([p for p in PATCH_SIZES if p <= max_len]))
    return ps * int(rng.integers(1, max_len // ps + 1))


TRAIN, EARLY_STOP, GATE = 0, 1, 2
SPLIT_NAMES = {TRAIN: "train", EARLY_STOP: "early_stop", GATE: "gate"}


def source_groups(datasets):
    """Map each source to its dataset names. The source is the name before '/'."""
    groups = {}
    for d in datasets:
        groups.setdefault(d.split("/")[0], []).append(d)
    return groups


def item_ids(ds_name):
    """Return the item ids of a dataset's HF rows, or None if the dataset is missing.

    Uses the row index when there is no item_id column.
    """
    for path in [DATA_DIR / ds_name.replace("/", "_"), DATA_DIR / ds_name]:
        if path.exists():
            from datasets import Dataset as HFDataset
            data = HFDataset.load_from_disk(str(path))
            if "item_id" in data.column_names:
                return [str(i) for i in data["item_id"]]
            return [str(i) for i in range(len(data))]
    return None


def split_items(datasets, seed):
    """Three-way item split per source group, shared by all its variants.

    Returns ({dataset: {row: split}}, group meta). Variants of a source must
    list the same item ids in the same order, or the whole group goes to train.
    Groups with fewer than 3 items also go to train.
    """
    row_split, meta = {}, {}
    for group, names in source_groups(datasets).items():
        ids = {n: item_ids(n) for n in names}
        present = [n for n in names if ids[n] is not None]
        if not present:
            continue
        ref = ids[present[0]]
        aligned = all(ids[n] == ref for n in present)
        split = np.full(len(ref), TRAIN, dtype=np.int64)
        if aligned and len(ref) >= 3:
            rng = np.random.default_rng([seed, zlib.crc32(group.encode())])
            perm = rng.permutation(len(ref))
            n_gate = max(1, int(len(ref) * GATE_RATIO))
            n_es = max(1, int(len(ref) * EARLY_STOP_RATIO))
            split[perm[:n_gate]] = GATE
            split[perm[n_gate:n_gate + n_es]] = EARLY_STOP
        for n in present:
            row_split[n] = {r: int(split[r]) for r in range(len(ids[n]))} if aligned else \
                {r: TRAIN for r in range(len(ids[n]))}
        meta[group] = {
            "variants": present, "aligned": aligned, "n_items": len(ref),
            **{f"{SPLIT_NAMES[k]}_rows": np.flatnonzero(split == k).tolist() if aligned else []
               for k in (EARLY_STOP, GATE)},
        }
    return row_split, meta


def _crop_dataset(job):
    """Crop windows from one dataset. Labels are computed separately.

    ``job`` is (ds_name, seed, {row: split}). The crop RNG depends only on
    (seed, ds_name). Each window carries the split of its row and its freq.
    """
    ds_name, seed, row_split = job
    rng = np.random.default_rng([seed, zlib.crc32(ds_name.encode())])
    series_list = load_hf_dataset(ds_name)
    out = {"name": ds_name, "windows": [], "freqs": [], "rows": [], "split": [],
           "constant_dropped": 0}
    if not series_list:
        return out

    # 50 crops per series, between 500 and SAMPLES_PER_DATASET
    n_crops = min(SAMPLES_PER_DATASET, max(500, len(series_list) * 50))
    for _ in range(n_crops):
        row, ts, freq = series_list[rng.integers(len(series_list))]
        win_len = sample_crop_len(rng, len(ts))
        start = rng.integers(0, len(ts) - win_len + 1)
        window = ts[start:start + win_len].copy()

        if np.std(window) < 1e-8:
            out["constant_dropped"] += 1
            continue
        out["windows"].append(window)
        out["freqs"].append(freq)
        out["rows"].append(row)
        out["split"].append(row_split[row])
    return out


def generate_training_data(datasets, seed, num_workers=8):
    """Build the label cache for ``datasets``: windows, raw labels, item split and counts."""
    import time
    from multiprocessing import Pool

    row_split, groups = split_items(datasets, seed)
    print(f"  Processing {len(datasets)} datasets with {num_workers} workers...")
    jobs = [(name, seed, row_split.get(name, {})) for name in datasets]
    t0 = time.time()
    with Pool(num_workers) as pool:
        crops = pool.map(_crop_dataset, jobs)
        pairs = [w for r in crops for w in r["windows"]]
        print(f"  Cropped {len(pairs)} windows in {time.time() - t0:.0f} s, labelling...")
        t0 = time.time()
        flat_labels = pool.map(compute_features_single, pairs, chunksize=16)
    print(f"  Labelled {len(pairs)} windows in {time.time() - t0:.0f} s")

    windows, labels, sources, rows, split = [], [], [], [], []
    meta = {"seed": seed, "datasets_requested": list(datasets), "datasets": {},
            "groups": groups, "label_code_sha256": code_sha256()}
    for g, m in groups.items():
        if not m["aligned"]:
            print(f"  group {g}: item ids differ across {m['variants']}, all items go to train")
    pos = 0
    for r in crops:
        r_labels = flat_labels[pos:pos + len(r["windows"])]
        pos += len(r["windows"])
        keep = [i for i, lab in enumerate(r_labels) if lab is not None]
        if not keep:
            print(f"  {r['name']}: SKIP, not found or empty")
            continue
        r_split = [r["split"][i] for i in keep]
        windows += [r["windows"][i] for i in keep]
        labels += [r_labels[i] for i in keep]
        rows += [r["rows"][i] for i in keep]
        split += r_split
        sources += [r["name"]] * len(keep)
        counts = {f"{v}_windows": int(sum(x == k for x in r_split))
                  for k, v in SPLIT_NAMES.items()}
        meta["datasets"][r["name"]] = {**counts, "constant_dropped": r["constant_dropped"],
                                       "label_failed": len(r_labels) - len(keep),
                                       "freqs": sorted({str(f) for f in r["freqs"]})}
        print(f"  {r['name']}: " + " / ".join(f"{v} {k}" for k, v in counts.items())
              + f", {r['constant_dropped']} constant dropped, freq {meta['datasets'][r['name']]['freqs']}")
    print(f"\nTotal samples: {len(windows)}")
    return {
        "windows": windows,
        "labels": np.array(labels, dtype=np.float32).reshape(-1, 4),
        "sources": np.array(sources),
        "rows": np.array(rows, dtype=np.int64),
        "split": np.array(split, dtype=np.int8),
        "meta": meta,
    }


def save_cache(path, data):
    """Write the label cache as .npz without pickled objects."""
    lens = np.array([len(w) for w in data["windows"]], dtype=np.int64)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path,
             window_values=np.concatenate(data["windows"]) if len(lens) else np.zeros(0, np.float32),
             window_offsets=np.concatenate([[0], np.cumsum(lens)]),
             labels=data["labels"], sources=data["sources"], rows=data["rows"],
             split=data["split"], meta=np.array(json.dumps(data["meta"], sort_keys=True)))


def load_cache(path):
    """Inverse of ``save_cache``."""
    c = np.load(path)
    values, off = c["window_values"], c["window_offsets"]
    return {
        "windows": [values[off[i]:off[i + 1]] for i in range(len(off) - 1)],
        "labels": c["labels"], "sources": c["sources"], "rows": c["rows"],
        "split": c["split"], "meta": json.loads(str(c["meta"])),
    }


def cache_sha256(path):
    """sha256 of the cache file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantile_normalize(labels, is_train):
    """Map each label column to (0, 1] by rank among the train-split values.

    Returns normalized labels and the per-feature sorted train references.
    """
    normed = labels.copy()
    refs = {}
    for i, name in enumerate(FEATURE_NAMES):
        sorted_vals = np.sort(labels[is_train, i])
        refs[name] = sorted_vals
        rank = np.searchsorted(sorted_vals, labels[:, i], side='right')
        normed[:, i] = np.clip(rank / len(sorted_vals), 0.0, 1.0)
    return normed, refs


# ============================================================
# Dataset class for variable-length windows
# ============================================================

class RegimeDataset(Dataset):
    """(window, target) pairs. Windows stay unpadded until ``collate_windows``."""

    def __init__(self, windows, targets):
        self.windows = windows
        self.targets = np.asarray(targets, dtype=np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.targets[idx]


def collate_windows(batch):
    """Stack windows with the pre-training preprocessing. Returns (x, lengths, y)."""
    X, lengths = stack_contexts([w for w, _ in batch])
    y = torch.from_numpy(np.stack([t for _, t in batch]))
    return torch.from_numpy(X), torch.from_numpy(lengths), y


def gate_metrics(pred, target):
    """Spearman, MAE and pred-std / target-std for one descriptor."""
    from scipy.stats import spearmanr
    return {
        "spearman": float(spearmanr(pred, target)[0]),
        "mae": float(np.abs(pred - target).mean()),
        "std_ratio": float(pred.std() / target.std()) if target.std() > 1e-12 else float("nan"),
        "n": int(len(pred)),
    }


def train_single_predictor(
    feature_idx, feature_name, train_windows, train_targets, val_windows, val_targets,
    seed, epochs=NUM_EPOCHS, device="cuda", loader_workers=4,
):
    """Train one SingleFeaturePredictor on (window, normalized target) pairs."""
    print(f"\n{'='*60}")
    print(f"Training predictor for: {feature_name} (idx={feature_idx})")
    print(f"{'='*60}")
    torch.manual_seed(seed + feature_idx)

    loader_kw = dict(batch_size=BATCH_SIZE, collate_fn=collate_windows,
                     num_workers=loader_workers, pin_memory=device != "cpu")
    train_loader = DataLoader(RegimeDataset(train_windows, train_targets), shuffle=True,
                              drop_last=len(train_windows) >= BATCH_SIZE,
                              generator=torch.Generator().manual_seed(seed + feature_idx),
                              **loader_kw)
    val_loader = DataLoader(RegimeDataset(val_windows, val_targets), shuffle=False, **loader_kw)

    model = SingleFeaturePredictor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    print(f"  Label distribution: mean={train_targets.mean():.3f}, "
          f"std={train_targets.std():.3f}, "
          f"min={train_targets.min():.3f}, max={train_targets.max():.3f}")

    def predict(loader):
        model.eval()
        preds = []
        with torch.no_grad():
            for x, lengths, _ in loader:
                preds.append(torch.sigmoid(model(x.to(device), lengths.to(device))).cpu())
        return torch.cat(preds).numpy()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for x, lengths, y in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            pred = torch.sigmoid(model(x, lengths))
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss /= max(n_batches, 1)

        preds = predict(val_loader)
        val_loss = float(((preds - val_targets) ** 2).mean())
        val_mae = float(np.abs(preds - val_targets).mean())

        if epoch % 10 == 0 or epoch == epochs - 1:
            corr = np.corrcoef(preds, val_targets)[0, 1]
            print(f"  Epoch {epoch:3d}: train_mse={train_loss:.4f}, "
                  f"val_mse={val_loss:.4f}, val_mae={val_mae:.4f}, corr={corr:.4f}, "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best val_mse={best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    preds = predict(val_loader)
    m = gate_metrics(preds, val_targets)
    print(f"\n  Final: MAE={m['mae']:.4f}, Spearman={m['spearman']:.4f}, "
          f"std ratio={m['std_ratio']:.3f}")
    print(f"  Pred range: [{preds.min():.3f}, {preds.max():.3f}]")
    print(f"  Target range: [{val_targets.min():.3f}, {val_targets.max():.3f}]")
    return model.cpu(), m


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train the structural-descriptor predictors.")
    parser.add_argument("--feature", type=str, default=None, choices=TRAINABLE_FEATURES,
                        help="Train only this descriptor. Default: all three.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cache", type=str, default=None,
                        help="Label cache .npz. Loaded if present, else generated and saved. "
                             "Default: $REGIME_CACHE_DIR/label_cache_seed{seed}.npz")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help="Directory for the predictor checkpoints.")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Override the DATASETS list, for example for a smoke test.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Maximum training epochs.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device, for example cuda:0. Default: auto-detect.")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel workers for feature computation.")
    parser.add_argument("--loader_workers", type=int, default=4,
                        help="Number of DataLoader workers for training.")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache or env.REGIME_CACHE_DIR / f"label_cache_seed{args.seed}.npz")
    datasets = args.datasets or DATASETS
    check_training_root(DATA_DIR, datasets)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # Step 1: one label cache of windows, labels and item split, shared by all predictors
    print("\n" + "=" * 60)
    print("Step 1: Label cache")
    print("=" * 60)
    if cache_path.exists():
        print(f"  Loading cached data from {cache_path}...")
        data = load_cache(cache_path)
        m = data["meta"]
        if m["seed"] != args.seed or m["datasets_requested"] != list(datasets):
            raise ValueError(f"{cache_path} was built with seed {m['seed']} and another "
                             "dataset list. Pass a different --cache or matching args.")
        if m["label_code_sha256"] != code_sha256():
            raise ValueError(f"{cache_path} was built by other label code: cache "
                             f"{m['label_code_sha256'][:12]}, current {code_sha256()[:12]}. "
                             "Pass a different --cache.")
    else:
        data = generate_training_data(datasets, args.seed, num_workers=args.num_workers)
        save_cache(cache_path, data)
        print(f"  Cache saved to {cache_path}: {len(data['windows'])} samples")
    cache_hash = cache_sha256(cache_path)

    if len(data["windows"]) < 100:
        print("ERROR: Too few samples generated. Check data paths.")
        return

    # Step 2: read the item split stored in the cache and quantile-normalize targets
    split = data["split"]
    windows = data["windows"]
    train_idx, val_idx, gate_idx = (np.flatnonzero(split == k) for k in (TRAIN, EARLY_STOP, GATE))
    if len(val_idx) == 0 or len(gate_idx) == 0:
        print("ERROR: no early-stop or gate items. Every source has fewer than 3 items.")
        return
    print("\n" + "=" * 60)
    print("Step 2: Train / early-stop / gate split by source-group item")
    print("=" * 60)
    print(f"  Windows: train {len(train_idx)}, early-stop {len(val_idx)}, gate {len(gate_idx)}")

    raw = data["labels"]
    print("\n  Raw label distributions (train):")
    for i, name in enumerate(FEATURE_NAMES):
        vals = raw[train_idx, i]
        print(f"    {name:20s}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
              f"[{vals.min():.3f}, {vals.max():.3f}]")
    targets, quantile_refs = quantile_normalize(raw, split == TRAIN)

    train_windows = [windows[i] for i in train_idx]
    val_windows = [windows[i] for i in val_idx]
    gate_windows = [windows[i] for i in gate_idx]
    record = {
        'seed': args.seed,
        'code_sha256': code_sha256(),
        'label_code_sha256': data["meta"]["label_code_sha256"],
        'cache_sha256': cache_hash,
        'datasets': data["meta"]["datasets"],
        'split_windows': {"train": len(train_idx), "early_stop": len(val_idx),
                          "gate": len(gate_idx)},
        'split_items': {g: {"n_items": m["n_items"], "aligned": m["aligned"],
                            "early_stop": len(m["early_stop_rows"]), "gate": len(m["gate_rows"])}
                        for g, m in data["meta"]["groups"].items()},
        'input_path': 'stack_contexts+masked',
    }

    # Step 3: train predictors
    features_to_train = TRAINABLE_FEATURES if args.feature is None else [args.feature]
    print("\n" + "=" * 60)
    print(f"Step 3: Training {len(features_to_train)} predictor(s): {features_to_train}")
    print("=" * 60)
    for name in features_to_train:
        i = FEATURE_NAMES.index(name)
        model, metrics = train_single_predictor(
            i, name, train_windows, targets[train_idx, i], val_windows, targets[val_idx, i],
            args.seed, epochs=args.epochs, device=device, loader_workers=args.loader_workers,
        )
        save_path = output_dir / f"regime_predictor_{name}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_name': name,
            'feature_idx': i,
            'quantile_ref': quantile_refs[name].tolist(),
            'early_stop_metrics': metrics,
            **record,
        }, save_path)
        print(f"  Saved: {save_path}")

    # Step 4: combined predictor with 3 learned descriptors and analytic sparsity, from one cache
    paths = [output_dir / f"regime_predictor_{n}.pt" for n in TRAINABLE_FEATURES]
    missing = [n for n, p in zip(TRAINABLE_FEATURES, paths) if not p.exists()]
    if missing:
        print(f"\nStep 4 skipped. Waiting for: {missing}")
        return
    ckpts = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    stale = [n for n, c in zip(TRAINABLE_FEATURES, ckpts) if c.get('cache_sha256') != cache_hash]
    if stale:
        print(f"\nStep 4 skipped. {stale} trained on a different label cache. Retrain them.")
        return
    print("\n" + "=" * 60)
    print("Step 4: Combined predictor and gate metrics on held-out gate items")
    print("=" * 60)
    combined = CombinedRegimePredictor()
    for i, c in enumerate(ckpts):
        combined.predictors[i].load_state_dict(c['model_state_dict'])
    combined.eval().to(device)

    # Use gate items only, which early stopping does not see. The input path is the
    # same as in AMETSPretrain._compute_regime_probs.
    with torch.no_grad():
        probs = series_regime_probs(combined, gate_windows, device).cpu().numpy()
    gate = {name: gate_metrics(probs[:, i], targets[gate_idx, i])
            for i, name in enumerate(TRAINABLE_FEATURES)}
    gate["sparsity"] = gate_metrics(probs[:, 3], raw[gate_idx, 3])
    for name, m in gate.items():
        print(f"  {name:16s} spearman={m['spearman']:.4f} mae={m['mae']:.4f} "
              f"std_ratio={m['std_ratio']:.3f} (n={m['n']})")
    # Report the same metrics per window-length bucket.
    gate_len = np.array([len(w) for w in gate_windows])
    gate_by_length = {}
    for lo, hi in LENGTH_BUCKETS:
        sel = (gate_len >= lo) & (gate_len < hi)
        if sel.sum() < 3:
            continue
        key = f"{lo}-{hi - 1}"
        gate_by_length[key] = {name: gate_metrics(probs[sel, i], targets[gate_idx[sel], i])
                               for i, name in enumerate(TRAINABLE_FEATURES)}
        print(f"  len {key:9s} n={int(sel.sum()):5d} " + " ".join(
            f"{n[:6]}: rho={m['spearman']:.3f} mae={m['mae']:.3f} sr={m['std_ratio']:.2f}"
            for n, m in gate_by_length[key].items()))

    combined_path = output_dir / "regime_predictor_combined.pt"
    torch.save({
        'model_state_dict': combined.cpu().state_dict(),
        'feature_names': FEATURE_NAMES,
        'num_regimes': 4,
        'quantile_refs': {name: ref.tolist() for name, ref in quantile_refs.items()},
        'gate_metrics': gate,
        'gate_metrics_by_length': gate_by_length,
        **record,
    }, combined_path)
    CombinedRegimePredictor().load(str(combined_path))  # raises on any key mismatch
    print(f"Saved combined: {combined_path}")

    # Step 5: check predictions on synthetic series
    print("\n" + "=" * 60)
    print("Step 5: Verification on synthetic data")
    print("=" * 60)
    combined.to(device)
    rng = np.random.default_rng(args.seed)
    t = np.arange(192, dtype=np.float32)
    test_cases = {
        "pure_sine":    np.sin(2 * np.pi * t / 12),
        "linear_up":    t * 2.0,
        "white_noise":  rng.standard_normal(192).astype(np.float32) * 10,
        "mostly_zeros": np.where(rng.random(192) > 0.9, 5.0, 0.0).astype(np.float32),
        "trend+season": t * 0.5 + 10 * np.sin(2 * np.pi * t / 30),
        "constant_42":  np.ones(192, dtype=np.float32) * 42,
    }
    with torch.no_grad():
        probs = series_regime_probs(
            combined, [v.astype(np.float32) for v in test_cases.values()], device).cpu().numpy()
    print(f"{'Case':20s} {'forecast':>8s} {'season':>8s} {'trend':>8s} {'sparse':>8s}")
    print("-" * 55)
    for name, p in zip(test_cases, probs):
        print(f"{name:20s} {p[0]:8.4f} {p[1]:8.4f} {p[2]:8.4f} {p[3]:8.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
