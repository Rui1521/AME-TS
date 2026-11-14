#!/usr/bin/env python3
"""
AMETS evaluation on the GIFT-Eval benchmark.

Supports univariate and multivariate evaluation modes. In 'both' mode, a
multivariate dataset is evaluated twice (univariate and multivariate), so that
select_configs.py can pick the best config across context and eval_mode. The
results CSV carries an 'eval_mode' column.

Usage (from the repository root):
    # Both modes (only multivariate datasets get the multivariate pass)
    python scripts/evaluate.py --split val \
        --checkpoint outputs/pretrain/<model>/<run_name>/checkpoints/<ckpt>.ckpt \
        --datasets m4_weekly ett1/H \
        --eval_mode both

    # Univariate only
    python scripts/evaluate.py --split val --eval_mode univariate ...

    # Multivariate only
    python scripts/evaluate.py --split val --eval_mode multivariate ...

--split {val,test} is required. The val split has as many windows as test and
sits immediately before GIFT-Eval's test windows, so no val input contains a
test point. Test is evaluated only on configs chosen on val:

    # grid on val (one --max_context per call)
    python scripts/evaluate.py --split val --max_context 512 ...
    # rank-1 val config per task, on test
    python scripts/evaluate.py --split test --selection selection.json ...
    # median of the val top-5 configs' point forecasts, on test
    python scripts/evaluate.py --split test --selection selection.json --ensemble top5 ...

`--split test` without `--selection` is refused unless `--allow-unselected-test`.
"""

import os
import sys
import json
import csv
import math
import pickle
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

# Put the repository root on sys.path so `amets` imports without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Importing env exports AMETS_ROOT, GIFT_EVAL_DATA and GIFT_EVAL into os.environ
# with repo-relative defaults, so gift_eval.data can resolve dataset locations.
from amets.common.env import (  # noqa: E402
    add_gift_eval_to_path,
    dataset_properties_path,
    env,  # noqa: F401
)

# Make gift_eval importable from a local GIFT-Eval checkout.
add_gift_eval_to_path()

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from gift_eval.data import Dataset as GiftDataset
from gluonts.model import evaluate_model
from gluonts.model.forecast import SampleForecast
from gluonts.itertools import batcher
from gluonts.ev.metrics import (
    MSE, MAE, MASE, MAPE, SMAPE, MSIS, RMSE, NRMSE, ND,
    MeanWeightedSumQuantileLoss,
)

from amets.model.pretrain import AMETSPretrain
from amets.data.gift_eval_val import (  # noqa: F401  (re-exported)
    PRETTY_NAMES, DEFAULT_PATCH_RANGES, build_ds_config, build_eval_data,
    get_all_patch_sizes_for_freq, grid_patch_sizes, noise_seed, normalize_freq,
    val_min_input,
)


# CSV header with eval_mode and patch_size columns
CSV_HEADER = [
    "dataset", "model",
    "eval_metrics/MSE[mean]", "eval_metrics/MSE[0.5]",
    "eval_metrics/MAE[0.5]", "eval_metrics/MASE[0.5]",
    "eval_metrics/MAPE[0.5]", "eval_metrics/sMAPE[0.5]",
    "eval_metrics/MSIS",
    "eval_metrics/RMSE[mean]", "eval_metrics/NRMSE[mean]",
    "eval_metrics/ND[0.5]", "eval_metrics/mean_weighted_sum_quantile_loss",
    "domain", "num_variates", "eval_mode", "patch_size",
    # split is val or test. selection is grid, val_rank1 or val_top5_median.
    # members lists ensemble members as "max_context:patch_size;..." and is empty otherwise.
    "split", "max_context", "selection", "members",
    # n_series counts the series in the task for this eval_mode. n_eval_series counts
    # the series that give instances on this split, because val drops series too
    # short for a val window.
    "n_series", "n_eval_series", "checkpoint_sha256",
]


def get_context_length(target_len: int, patch_size: int, max_context: int) -> int:
    usable_len = min(target_len, max_context)
    context_len = (usable_len // patch_size) * patch_size
    if context_len < patch_size:
        context_len = patch_size
    return context_len


def _extract_metric(metrics, key):
    val = metrics[key]
    if hasattr(val, 'iloc'):
        return val.iloc[0]
    return val[0]


def load_amets_checkpoint(checkpoint_path, device):
    """Load an AMETSPretrain checkpoint for inference.

    The descriptor predictor's weights come from the checkpoint's state_dict through a
    strict load. The function does not read the regime_predictor_path saved in the
    hyperparameters, which is an absolute path on the training machine, so that file
    need not exist.
    """
    return AMETSPretrain.load_from_checkpoint(checkpoint_path, map_location=device,
                                              regime_predictor_path=None)


class AMETSPredictor:
    """Predictor for AMETS models compatible with gluonts evaluation."""

    def __init__(
        self,
        checkpoint_path: str,
        prediction_length: int,
        freq: str,
        device: str = "cuda",
        max_context: int = 4096,
        batch_size: int = 512,
        num_samples: int = 10,
        num_variates: int = 1,
        patch_size: Optional[int] = None,
        _preloaded_model: Optional["AMETSPretrain"] = None,
        noise_seed: Optional[int] = None,
    ):
        self.prediction_length = prediction_length
        self.freq = freq
        self.device = device
        self.max_context = max_context
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_variates = num_variates
        # Seed for sample noise. Each predict() call creates a new generator, so each
        # candidate evaluation starts from the same draws. None uses global np.random.
        self.noise_seed = noise_seed

        if _preloaded_model is not None:
            self.model = _preloaded_model
        else:
            self.model = load_amets_checkpoint(checkpoint_path, device)
            self.model.eval()
            self.model.to(device)

        self.max_patch_size = max(self.model.module.patch_sizes)
        if patch_size is not None:
            self.patch_size = patch_size
        else:
            self.patch_size = get_all_patch_sizes_for_freq(freq, self.model.module.patch_sizes)[0]

        print(f"  Predictor: patch_size={self.patch_size}, num_variates={num_variates}, "
              f"max_context={max_context}, pred_len={prediction_length}")

    def _compute_regime_probs(self, context_target, sample_id, variate_id, num_variates=1):
        regime_predictor = self.model.module.regime_predictor
        if regime_predictor is None:
            raise ValueError("Regime predictor is None.")
        device = context_target.device
        seq_len = sample_id.shape[0]
        num_regimes = self.model.module.num_regimes
        regime_predictor.eval()

        def _process_variate_context(var_context):
            var_context = torch.nan_to_num(var_context, nan=0.0)
            if len(var_context) > 192:
                var_context = var_context[-192:]
            if len(var_context) < 8:
                var_context = F.pad(var_context, (8 - len(var_context), 0), value=0.0)
            with torch.no_grad():
                probs = regime_predictor.get_probabilities(var_context.unsqueeze(0))
            if probs.shape[-1] < num_regimes:
                sparsity = (var_context == 0).float().mean().unsqueeze(0).unsqueeze(0).to(device)
                probs = torch.cat([probs, sparsity], dim=-1)
            if torch.isnan(probs).any():
                raise ValueError("Regime predictor returned NaN.")
            return probs.squeeze(0)

        if num_variates == 1:
            context_flat = context_target[:, :self.patch_size].flatten()
            probs = _process_variate_context(context_flat)
            return probs.unsqueeze(0).expand(seq_len, -1)
        else:
            num_tokens = seq_len // num_variates
            variate_probs = []
            for v in range(num_variates):
                start = v * num_tokens
                end = (v + 1) * num_tokens
                var_context = context_target[start:end, :self.patch_size].flatten()
                variate_probs.append(_process_variate_context(var_context))
            regime_probs = torch.zeros(seq_len, num_regimes, device=device)
            for i in range(seq_len):
                regime_probs[i] = variate_probs[variate_id[i].item()]
            return regime_probs

    def predict(self, test_data_input):
        test_data_input = list(test_data_input)
        rng = _noise_rng(self.noise_seed)
        samples = [self._to_samples(p, rng) for p in self.predict_points(test_data_input)]
        return _make_forecasts(samples, test_data_input)

    def predict_points(self, test_data_input) -> List[np.ndarray]:
        """Deterministic point forecast per input, before sample noise is added."""
        while True:
            try:
                all_points = []
                for batch in batcher(test_data_input, batch_size=self.batch_size):
                    past_targets = [entry["target"] for entry in batch]
                    all_points.extend(self._predict_batch(past_targets))
                return all_points
            except torch.cuda.OutOfMemoryError:
                print(f"OOM at batch_size {self.batch_size}, reducing to {self.batch_size // 2}")
                self.batch_size //= 2
                if self.batch_size < 1:
                    raise RuntimeError("Batch size reduced to 0")

    def _to_samples(self, predictions_np: np.ndarray, rng) -> np.ndarray:
        # Noise scale per variate, so a small-valued variate is not swamped when
        # variate scales differ by orders of magnitude.
        noise_scale = 0.01 * (np.abs(predictions_np).mean(axis=0) + 1e-6)
        noise = rng.normal(0, noise_scale, (self.num_samples,) + predictions_np.shape)
        samples = predictions_np[np.newaxis, :] + noise

        if self.num_variates == 1 and samples.ndim == 3:
            samples = samples.squeeze(-1)
        return samples

    def _predict_batch(self, past_targets: List[np.ndarray]) -> List[np.ndarray]:
        batch_forecasts = []
        with torch.no_grad():
            for idx, target in enumerate(past_targets):
                if idx % 10 == 0:
                    torch.cuda.empty_cache()

                if not isinstance(target, np.ndarray):
                    target = np.array(target)

                num_variates = self.num_variates

                if target.ndim == 1:
                    target = target.reshape(-1, 1)
                elif target.ndim == 2:
                    if num_variates > 1 and target.shape[0] == num_variates:
                        target = target.T
                else:
                    raise ValueError(f"Unexpected target shape: {target.shape}")

                seq_len = target.shape[0]
                target = np.nan_to_num(target, nan=0.0)
                context_len = get_context_length(seq_len, self.patch_size, self.max_context)
                target = target[-context_len:]

                num_context_tokens = context_len // self.patch_size
                num_pred_tokens = math.ceil(self.prediction_length / self.patch_size)
                total_tokens = num_context_tokens + num_pred_tokens

                if num_variates == 1:
                    predictions = self._forward_univariate(
                        target, num_context_tokens, num_pred_tokens, total_tokens
                    )
                else:
                    predictions = self._forward_multivariate(
                        target, num_variates, num_context_tokens, num_pred_tokens, total_tokens
                    )

                predictions_np = predictions.cpu().numpy()
                batch_forecasts.append(predictions_np)
        return batch_forecasts

    def _forward_univariate(self, target, num_context_tokens, num_pred_tokens, total_tokens):
        context_patched = target.reshape(num_context_tokens, self.patch_size)
        last_value = target[-1, 0]
        pred_patches = np.full((num_pred_tokens, self.patch_size), last_value)
        target_patched = np.concatenate([context_patched, pred_patches], axis=0)
        target_padded = np.pad(target_patched, ((0, 0), (0, self.max_patch_size - self.patch_size)), 'constant')

        target_tensor = torch.tensor(target_padded, dtype=torch.float32).to(self.device)
        sample_id = torch.ones(total_tokens, dtype=torch.long, device=self.device)
        time_id = torch.arange(total_tokens, dtype=torch.long, device=self.device)
        variate_id = torch.zeros(total_tokens, dtype=torch.long, device=self.device)
        observed_mask = torch.zeros(total_tokens, self.max_patch_size, dtype=torch.bool, device=self.device)
        observed_mask[:num_context_tokens, :self.patch_size] = True
        prediction_mask = torch.zeros(total_tokens, dtype=torch.bool, device=self.device)
        prediction_mask[num_context_tokens:] = True
        patch_size_tensor = torch.full((total_tokens,), self.patch_size, dtype=torch.long, device=self.device)

        regime_probs = None
        if self.model.module.routing_mode == 'additive':
            rp = self._compute_regime_probs(
                target_tensor[:num_context_tokens], sample_id[:num_context_tokens],
                variate_id[:num_context_tokens], num_variates=1,
            )
            regime_probs = rp[0:1].expand(total_tokens, -1)

        output, *_ = self.model.module(
            target=target_tensor, observed_mask=observed_mask, sample_id=sample_id,
            time_id=time_id, variate_id=variate_id, prediction_mask=prediction_mask,
            patch_size=patch_size_tensor, regime_probs=regime_probs,
        )
        pred_output = output[num_context_tokens:, :self.patch_size]
        return pred_output.reshape(-1)[:self.prediction_length]

    def _forward_multivariate(self, target, num_variates, num_context_tokens, num_pred_tokens, total_tokens):
        all_patched = []
        for v in range(num_variates):
            var_data = target[:, v]
            var_context = var_data.reshape(num_context_tokens, self.patch_size)
            var_pred = np.full((num_pred_tokens, self.patch_size), var_data[-1])
            var_patched = np.concatenate([var_context, var_pred], axis=0)
            var_padded = np.pad(var_patched, ((0, 0), (0, self.max_patch_size - self.patch_size)), 'constant')
            all_patched.append(var_padded)

        packed_target = np.concatenate(all_patched, axis=0)
        packed_len = num_variates * total_tokens

        target_tensor = torch.tensor(packed_target, dtype=torch.float32, device=self.device)
        sample_id = torch.ones(packed_len, dtype=torch.long, device=self.device)
        time_id = torch.arange(total_tokens, dtype=torch.long, device=self.device).repeat(num_variates)
        variate_id = torch.arange(num_variates, dtype=torch.long, device=self.device).repeat_interleave(total_tokens)

        observed_mask = torch.zeros(packed_len, self.max_patch_size, dtype=torch.bool, device=self.device)
        prediction_mask = torch.zeros(packed_len, dtype=torch.bool, device=self.device)
        for v in range(num_variates):
            s = v * total_tokens
            observed_mask[s:s + num_context_tokens, :self.patch_size] = True
            prediction_mask[s + num_context_tokens:(v + 1) * total_tokens] = True
        patch_size_tensor = torch.full((packed_len,), self.patch_size, dtype=torch.long, device=self.device)

        regime_probs = None
        if self.model.module.routing_mode == 'additive':
            ctx_idx = []
            for v in range(num_variates):
                ctx_idx.extend(range(v * total_tokens, v * total_tokens + num_context_tokens))
            ctx_tensor = target_tensor[ctx_idx]
            ctx_sid = torch.ones(len(ctx_idx), dtype=torch.long, device=self.device)
            ctx_vid = torch.arange(num_variates, dtype=torch.long, device=self.device).repeat_interleave(num_context_tokens)
            rp_ctx = self._compute_regime_probs(ctx_tensor, ctx_sid, ctx_vid, num_variates)
            regime_probs = torch.zeros(packed_len, self.model.module.num_regimes, device=self.device)
            for v in range(num_variates):
                regime_probs[v * total_tokens:(v + 1) * total_tokens] = rp_ctx[v * num_context_tokens]

        output, *_ = self.model.module(
            target=target_tensor, observed_mask=observed_mask, sample_id=sample_id,
            time_id=time_id, variate_id=variate_id, prediction_mask=prediction_mask,
            patch_size=patch_size_tensor, regime_probs=regime_probs,
        )

        output_unpacked = output.reshape(num_variates, total_tokens, self.max_patch_size)
        pred_output = output_unpacked[:, num_context_tokens:, :self.patch_size]
        pred_flat = pred_output.reshape(num_variates, -1)[:, :self.prediction_length]
        return pred_flat.T  # (prediction_length, num_variates)


def _noise_rng(seed: Optional[int]):
    return np.random if seed is None else np.random.default_rng(seed)


def _make_forecasts(samples_list, test_data_input) -> List[SampleForecast]:
    sample_forecasts = []
    for samples, ts in zip(samples_list, test_data_input):
        target_arr = np.array(ts["target"])
        seq_len = target_arr.shape[-1] if target_arr.ndim == 2 else len(target_arr)
        sample_forecasts.append(
            SampleForecast(
                samples=samples,
                start_date=ts["start"] + seq_len,
                item_id=ts.get("item_id"),
            )
        )
    return sample_forecasts


class EnsemblePredictor:
    """Median of several AMETSPredictors' point forecasts, for the val top-k ensemble.

    Each member produces its deterministic point forecast. The ensemble point is the
    element-wise median across members, and the samples add the same noise to it as a
    single AMETSPredictor does. MASE[0.5] therefore means the same thing as for one
    config: the MASE of the sample median around the ensemble point forecast.
    All members must share eval_mode, so they cover the same items with the same
    forecast shape.
    """

    def __init__(self, members: List[AMETSPredictor], noise_seed: Optional[int] = None):
        if not members:
            raise ValueError("EnsemblePredictor needs at least one member")
        if len({m.num_variates for m in members}) != 1:
            raise ValueError("Ensemble members must share eval_mode")
        self.members = members
        self.noise_seed = noise_seed

    def predict(self, test_data_input):
        test_data_input = list(test_data_input)
        member_points = [m.predict_points(test_data_input) for m in self.members]
        rng = _noise_rng(self.noise_seed)
        samples = []
        for per_item in zip(*member_points):
            point = np.median(np.stack(per_item, axis=0), axis=0)
            samples.append(self.members[0]._to_samples(point, rng))
        return _make_forecasts(samples, test_data_input)


# ===== Fast path =====
# The fast path computes the same results as AMETSPredictor with gluonts
# evaluate_model, organized differently:
#   * It builds the inputs, labels and seasonal errors of a task once and shares
#     them across all of the task's candidate configs.
#   * It runs the model on batches of same-length items, with a batch dimension and
#     no padding, instead of one forward pass per item.
#   * It builds noise samples, means and quantiles with array ops that reproduce the
#     per-item draws and reductions bit for bit, with the same rng stream and order.
#   * It uses gluonts' own evaluators on item batches of the same size as
#     evaluate_model uses, so masking, seasonal error, NaN checks and aggregation
#     follow gluonts.
# --per_item_eval selects the per-item path as a reference.

# A forward batch holds as many same-length items as fit in FWD_TOKENS tokens and
# ATTN_BUDGET attention-matrix elements (B * L^2). It halves on CUDA OOM. It does
# not depend on --batch_size, which sets the metric batches as in evaluate_model.
FWD_TOKENS = 2 ** 16
ATTN_BUDGET = 2 ** 26


def seasonal_error_masked(x, seasonality: int):
    """Compute gluonts.ev.ts_stats.seasonal_error(np.ma.masked_invalid(x), seasonality, -1).

    The result matches gluonts bit for bit in values and dtype, computed with plain
    ndarrays, which is faster on long series. numpy.ma takes one of two branches
    depending on whether masked_invalid returns nomask, and this function mirrors
    both. Edge cases call gluonts directly."""
    from gluonts.ev.ts_stats import seasonal_error
    x = np.asarray(x)
    L = x.shape[-1]
    m = 1 if seasonality > L else seasonality
    if x.dtype.kind != "f" or L - m <= 0:
        return seasonal_error(np.ma.masked_invalid(x), seasonality=seasonality, time_axis=-1)
    a, b = x[..., m:], x[..., :L - m]
    if np.ma.getmask(np.ma.masked_invalid(x)) is np.ma.nomask:
        return np.abs(a - b).mean(axis=-1, keepdims=True)
    finite = np.isfinite(x)
    valid = finite[..., m:] & finite[..., :L - m]
    cnt = valid.sum(axis=-1, keepdims=True)
    if (cnt == 0).any():
        return seasonal_error(np.ma.masked_invalid(x), seasonality=seasonality, time_axis=-1)
    with np.errstate(invalid="ignore", over="ignore"):
        d = np.where(valid, np.abs(a - b), 0)
    return d.sum(axis=-1, keepdims=True) * 1. / cnt


class TaskData:
    """One task's instances for one eval_mode, shared by all its candidate configs.

    Labels and seasonal errors are stored per item. metrics_from_points assembles them
    into the same per-batch arrays that gluonts' _get_data_batch builds: np.stack and
    masked_invalid per batch, and np.array of the batch's seasonal errors. The batch
    arrays' mask representation and dtype depend on which items share a batch, so
    the assembly must happen per batch.
    """

    def __init__(self, eval_data, seasonality: int):
        pairs = list(eval_data)
        self.inputs = [inp for inp, _ in pairs]
        self.labels = [lab["target"] for _, lab in pairs]
        self.label_ndim = self.labels[0].ndim
        self.seasonal_errors = [seasonal_error_masked(inp["target"], seasonality)
                                for inp in self.inputs]

    def __len__(self):
        return len(self.inputs)

    def batch(self, start: int, stop: int) -> dict:
        return {"label": np.ma.masked_invalid(np.stack(self.labels[start:stop], axis=0)),
                "seasonal_error": np.array(self.seasonal_errors[start:stop])}


def _prepare_target(target, num_variates, patch_size, max_context):
    """Apply AMETSPredictor._predict_batch's per-item preprocessing. Returns (context_len, nv)."""
    if not isinstance(target, np.ndarray):
        target = np.array(target)
    if target.ndim == 1:
        target = target.reshape(-1, 1)
    elif target.ndim == 2:
        if num_variates > 1 and target.shape[0] == num_variates:
            target = target.T
    else:
        raise ValueError(f"Unexpected target shape: {target.shape}")
    seq_len = target.shape[0]
    context_len = get_context_length(seq_len, patch_size, max_context)
    # Crop before nan_to_num, the reverse of the per-item path. nan_to_num is
    # element-wise and nothing downstream reads outside the last context_len rows,
    # so the result is identical and the whole history is not converted.
    target = np.nan_to_num(target[-context_len:], nan=0.0)
    if target.shape[0] != context_len:
        # The per-item reshape also fails here, for a series shorter than one patch.
        raise ValueError(f"cannot reshape series of length {seq_len} into patches of {patch_size}")
    return target


def _forward_group(module, ctx, num_variates, n_ctx, n_pred, patch_size, max_patch,
                   prediction_length, device):
    """Run _forward_univariate or _forward_multivariate on B items of equal length.

    ctx: (B, n_ctx * patch_size, nv). Returns (B, pl) or (B, pl, nv) float32.
    """
    B = ctx.shape[0]
    T = n_ctx + n_pred
    blocks = []
    for v in range(num_variates):
        var = ctx[:, :, v]
        var_ctx = var.reshape(B, n_ctx, patch_size)
        var_pred = np.broadcast_to(var[:, -1][:, None, None], (B, n_pred, patch_size))
        blocks.append(np.concatenate([var_ctx, var_pred], axis=1))
    packed = np.concatenate(blocks, axis=1)                       # (B, nv*T, ps)
    packed = np.pad(packed, ((0, 0), (0, 0), (0, max_patch - patch_size)), 'constant')
    L = num_variates * T

    target = torch.tensor(packed, dtype=torch.float32, device=device)
    sample_id = torch.ones(B, L, dtype=torch.long, device=device)
    time_id = torch.arange(T, dtype=torch.long, device=device).repeat(num_variates).expand(B, L)
    variate_id = (torch.arange(num_variates, dtype=torch.long, device=device)
                  .repeat_interleave(T).expand(B, L))
    observed_mask = torch.zeros(B, L, max_patch, dtype=torch.bool, device=device)
    prediction_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    for v in range(num_variates):
        s = v * T
        observed_mask[:, s:s + n_ctx, :patch_size] = True
        prediction_mask[:, s + n_ctx:(v + 1) * T] = True
    patch_size_t = torch.full((B, L), patch_size, dtype=torch.long, device=device)

    output, *_ = module(
        target=target, observed_mask=observed_mask, sample_id=sample_id,
        time_id=time_id.contiguous(), variate_id=variate_id.contiguous(),
        prediction_mask=prediction_mask, patch_size=patch_size_t, regime_probs=None,
    )
    out = output.reshape(B, num_variates, T, max_patch)[:, :, n_ctx:, :patch_size]
    out = out.reshape(B, num_variates, -1)[:, :, :prediction_length].cpu().numpy()
    return out[:, 0, :] if num_variates == 1 else out.transpose(0, 2, 1)


def predict_task_points(model, inputs, prediction_length, num_variates, patch_size,
                        max_context, device, batch_size=None) -> np.ndarray:
    """Return point forecasts of one config for all items, shaped (N, pl) or (N, pl, nv)."""
    module = model.module
    if module.routing_mode == 'additive':
        raise NotImplementedError("The fast path does not support additive routing, "
                                  "which needs per-item regime probs")
    max_patch = max(module.patch_sizes)
    n_pred = math.ceil(prediction_length / patch_size)
    ctxs = [_prepare_target(e["target"], num_variates, patch_size, max_context) for e in inputs]
    groups = {}
    for i, c in enumerate(ctxs):
        groups.setdefault(c.shape[0] // patch_size, []).append(i)
    out = [None] * len(ctxs)
    retried_at_one = False
    with torch.no_grad():
        for n_ctx, idx in groups.items():
            L = num_variates * (n_ctx + n_pred)
            B = max(1, min(FWD_TOKENS // L, ATTN_BUDGET // (L * L)))
            start = 0
            while start < len(idx):
                chunk = idx[start:start + B]
                try:
                    preds = _forward_group(
                        module, np.stack([ctxs[i] for i in chunk]), num_variates, n_ctx,
                        n_pred, patch_size, max_patch, prediction_length, device)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()   # cached blocks of earlier shapes can hold most of the GPU
                    if B == 1:
                        if retried_at_one:
                            raise
                        retried_at_one = True   # retry once with an empty cache, for example for a many-variate ensemble member at a long context
                        print("OOM at batch 1, retrying once with an empty cache")
                        continue
                    B //= 2
                    print(f"OOM, fast-path batch reduced to {B}")
                    continue
                for i, p in zip(chunk, preds):
                    out[i] = p
                start += len(chunk)
    torch.cuda.empty_cache()   # ensemble members use different shapes, so cached blocks would only fragment the GPU
    return np.stack(out)


class _BatchQuantiles:
    """Stand-in for gluonts' BatchForecast over SampleForecasts built from `samples`."""

    def __init__(self, samples: np.ndarray, allow_nan: bool = False):
        self.samples = samples            # (B, S, pl) or (B, S, pl, nv)
        self.allow_nan = allow_nan
        self._sorted = None

    def __getitem__(self, name):
        if name == "mean":
            res = np.mean(self.samples, axis=1)
        else:
            if self._sorted is None:
                self._sorted = np.sort(self.samples, axis=1)
            q = 0.5 if name == "median" else float(name)
            res = self._sorted[:, int(np.round((self.samples.shape[1] - 1) * q))]
        if res.ndim == 3:
            res = np.ascontiguousarray(res.transpose(0, 2, 1))  # forecast[name].T per item
        if np.isnan(res).any() and not self.allow_nan:
            raise ValueError("Forecast contains NaN values")
        return res


def metrics_from_points(points: np.ndarray, task: TaskData, gluonts_metrics, rng,
                        num_samples: int, batch_size: int, keep_samples: int = 0,
                        transposed_points: bool = False):
    """Add AMETSPredictor noise and run gluonts evaluate_forecasts(axis=None) per batch.

    Draws the same noise as AMETSPredictor._to_samples item by item, with one rng
    stream in item order, and feeds gluonts' evaluators batches of `batch_size`
    items, as evaluate_model does. Returns a dict from metric name to a 1-element
    array, and the samples of the first `keep_samples` items.

    Multivariate noise is scaled per variate. transposed_points is ignored.
    """
    from collections import ChainMap
    axis = tuple(range(task.label_ndim + 1))
    evaluators = {}
    for metric in gluonts_metrics:
        e = metric(axis=axis)
        evaluators[e.name] = e
    item_shape = points.shape[1:]
    kept = []
    for start in range(0, len(points), batch_size):
        P = points[start:start + batch_size]
        # Scale noise per variate (last axis) for multivariate items, as AMETSPredictor._to_samples does.
        if len(item_shape) == 2:
            scales = 0.01 * (np.abs(P).mean(axis=1) + 1e-6)          # (B, nv)
            scales = scales.astype(np.float64)[:, None, None, :]
        else:
            scales = np.array([0.01 * (np.abs(p).mean() + 1e-6) for p in P], dtype=np.float64)
            scales = scales.reshape((-1,) + (1,) * (len(item_shape) + 1))
        z = rng.standard_normal((len(P), num_samples) + item_shape)
        noise = 0.0 + scales * z
        samples = P[:, np.newaxis] + noise
        if sum(len(k) for k in kept) < keep_samples:
            kept.append(samples)
        batch = ChainMap(task.batch(start, start + batch_size), _BatchQuantiles(samples))
        for e in evaluators.values():
            e.update(batch)
    first = np.concatenate(kept)[:keep_samples] if kept else None
    return {name: np.ravel(e.get()) for name, e in evaluators.items()}, first


# ===== CSV helpers =====

def init_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)
        return
    with open(path, newline="") as f:
        existing = next(csv.reader(f), None)
    if existing != CSV_HEADER:
        raise ValueError(
            f"{path} exists with a different header. Use a fresh --output_dir.")


def append_csv(path, ds_config, model_name, metrics, domain, num_variates, eval_mode, patch_size,
               split, max_context, counts, ckpt_sha, selection="grid", members=""):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            ds_config, model_name,
            _extract_metric(metrics, "MSE[mean]"),
            _extract_metric(metrics, "MSE[0.5]"),
            _extract_metric(metrics, "MAE[0.5]"),
            _extract_metric(metrics, "MASE[0.5]"),
            _extract_metric(metrics, "MAPE[0.5]"),
            _extract_metric(metrics, "sMAPE[0.5]"),
            _extract_metric(metrics, "MSIS"),
            _extract_metric(metrics, "RMSE[mean]"),
            _extract_metric(metrics, "NRMSE[mean]"),
            _extract_metric(metrics, "ND[0.5]"),
            _extract_metric(metrics, "mean_weighted_sum_quantile_loss"),
            domain, num_variates, eval_mode, patch_size,
            split, max_context, selection, members,
            counts.n_series, counts.n_kept, ckpt_sha,
        ])


# ===== Main evaluation logic =====

def file_sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load_selection(path: str) -> dict:
    """Load a selection.json that lists the config chosen on val for each task."""
    with open(path) as f:
        selection = json.load(f)
    if selection.get("selected_on") != "val":
        raise ValueError(f"{path}: selected_on={selection.get('selected_on')!r}. "
                         "Only val-selected configs may be evaluated.")
    return selection


def evaluate_checkpoint(
    checkpoint_path: str,
    datasets: List[str],
    terms: List[str],
    output_dir: str,
    split: str,
    eval_mode: str = "both",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 512,
    max_context: Optional[int] = 512,
    num_samples: int = 10,
    selection: Optional[dict] = None,
    ensemble: Optional[str] = None,
    checkpoint_sha256: Optional[str] = None,
    fast: bool = True,
):
    if ensemble is not None and selection is None:
        raise ValueError("--ensemble requires --selection")
    ckpt_sha = checkpoint_sha256 or file_sha256(checkpoint_path)
    if selection is not None:
        sel_sha = selection.get("checkpoint", {}).get("sha256")
        if sel_sha != ckpt_sha:
            raise ValueError(
                f"Selection was made for checkpoint sha256={sel_sha}, but "
                f"{checkpoint_path} has sha256={ckpt_sha}. Re-run the val grid for this checkpoint.")
    print(f"\n{'='*80}")
    print(f"AMETS Unified Evaluation (split={split}, eval_mode={eval_mode}, "
          f"selection={'none' if selection is None else 'val'}, ensemble={ensemble}, "
          f"path={'fast' if fast else 'per-item'})")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}, Batch size: {batch_size}, Max context: "
          f"{max_context if selection is None else 'per selected config'}")
    print(f"{'='*80}\n")

    # Set up output directories
    experiment_name = Path(checkpoint_path).parent.parent.name
    out_dir = os.path.join(output_dir, experiment_name)
    metrics_dir = os.path.join(out_dir, "metrics")
    forecasts_dir = os.path.join(out_dir, "forecasts")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(forecasts_dir, exist_ok=True)
    csv_path = os.path.join(metrics_dir, "all_results.csv")
    init_csv(csv_path)

    # Load dataset properties from the gift-eval checkout
    with open(dataset_properties_path()) as f:
        dataset_properties = json.load(f)

    # Optionally restrict evaluation to the tasks listed in the `dataset` column of
    # the CSV at TASK_LIST_CSV.
    task_list_csv = os.environ.get("TASK_LIST_CSV")
    valid_combos = None
    if task_list_csv and Path(task_list_csv).exists():
        import pandas as pd
        valid_combos = set(pd.read_csv(task_list_csv)['dataset'].tolist())
        print(f"Loaded {len(valid_combos)} tasks from {task_list_csv}")

    gluonts_metrics = [
        MSE(forecast_type="mean"), MSE(forecast_type=0.5), MAE(), MASE(),
        MAPE(), SMAPE(), MSIS(), RMSE(), NRMSE(), ND(),
        MeanWeightedSumQuantileLoss(quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    ]

    # Load the model once and reuse it across all evaluations
    print(f"Loading model from {checkpoint_path}...")
    loaded_model = load_amets_checkpoint(checkpoint_path, device)
    loaded_model.eval()
    loaded_model.to(device)
    model_patch_sizes = loaded_model.module.patch_sizes
    print(f"Model loaded. patch_sizes={model_patch_sizes}")
    if fast and loaded_model.module.routing_mode == 'additive':
        print("routing_mode=additive: the fast path does not support it, using the per-item path")
        fast = False

    from gluonts.time_feature import get_seasonality

    for ds_name in tqdm(datasets, desc="Datasets"):
        ds_key = ds_name.split("/")[0].lower() if "/" in ds_name else ds_name.lower()
        ds_key = PRETTY_NAMES.get(ds_key, ds_key)
        if ds_key not in dataset_properties:
            print(f"Warning: {ds_key} not in properties. Skipping.")
            continue

        domain = dataset_properties[ds_key]["domain"]
        num_variates = dataset_properties[ds_key]["num_variates"]

        # Determine which eval modes to run for this dataset
        if eval_mode == "both" and num_variates > 1:
            modes = ["univariate", "multivariate"]
        elif eval_mode == "multivariate" and num_variates > 1:
            modes = ["multivariate"]
        else:
            modes = ["univariate"]

        for term in tqdm(terms, desc="Terms", leave=False):
            ds_config = build_ds_config(ds_name, term, dataset_properties)

            # Task-list filter (TASK_LIST_CSV)
            if valid_combos is not None and ds_config not in valid_combos:
                print(f"  Skipping {ds_config} (not in the task list)")
                continue

            task_modes = modes
            if selection is not None:
                task_sel = selection["tasks"].get(ds_config)
                if task_sel is None:
                    print(f"  Skipping {ds_config} (not in selection)")
                    continue
                task_modes = [task_sel["rank1"]["eval_mode"]]

            for mode in task_modes:
                to_univariate = (mode == "univariate" and num_variates > 1)
                effective_variates = 1 if mode == "univariate" else num_variates

                try:
                    dataset = GiftDataset(name=ds_name, term=term, to_univariate=to_univariate)
                    # Get all valid patch sizes for this frequency
                    valid_patch_sizes = get_all_patch_sizes_for_freq(dataset.freq, model_patch_sizes)
                    min_series_len = dataset._min_series_length
                    # Every candidate of the task uses the same threshold. See val_min_input.
                    counts = build_eval_data(
                        dataset, split, min_input=val_min_input(valid_patch_sizes, min_series_len),
                        to_univariate=to_univariate)
                    eval_data = counts.data
                except Exception as e:
                    print(f"  Skipping {ds_name} ({term}, {mode}, split={split}): {e}")
                    continue

                season_length = get_seasonality(dataset.freq)
                seed = noise_seed(ds_config, split, mode)

                def make_predictor(ctx, ps):
                    return AMETSPredictor(
                        checkpoint_path=checkpoint_path,
                        prediction_length=dataset.prediction_length,
                        freq=dataset.freq, device=device,
                        max_context=ctx, batch_size=batch_size,
                        num_samples=num_samples, num_variates=effective_variates,
                        patch_size=ps, _preloaded_model=loaded_model, noise_seed=seed,
                    )

                task_cache = {}

                def run_fast(cfgs, is_ensemble, keep_samples=0):
                    if "task" not in task_cache:
                        task_cache["task"] = TaskData(eval_data, season_length)
                    task = task_cache["task"]
                    pts = [predict_task_points(
                        loaded_model, task.inputs, dataset.prediction_length,
                        effective_variates, c["patch_size"], c["max_context"], device,
                        batch_size) for c in cfgs]
                    points = np.median(np.stack(pts, axis=0), axis=0) if is_ensemble else pts[0]
                    res, first = metrics_from_points(
                        points, task, gluonts_metrics, np.random.default_rng(seed),
                        num_samples, batch_size, keep_samples=keep_samples,
                        transposed_points=(not is_ensemble and effective_variates > 1))
                    print(f"  MSE={_extract_metric(res, 'MSE[mean]'):.4f}  "
                          f"MASE={_extract_metric(res, 'MASE[0.5]'):.4f}  "
                          f"sMAPE={_extract_metric(res, 'sMAPE[0.5]'):.4f}")
                    return res, first

                def run(predictor):
                    res = evaluate_model(
                        predictor, test_data=eval_data,
                        metrics=gluonts_metrics, batch_size=batch_size,
                        axis=None, mask_invalid_label=True,
                        allow_nan_forecast=False, seasonality=season_length,
                    )
                    print(f"  MSE={_extract_metric(res, 'MSE[mean]'):.4f}  "
                          f"MASE={_extract_metric(res, 'MASE[0.5]'):.4f}  "
                          f"sMAPE={_extract_metric(res, 'sMAPE[0.5]'):.4f}")
                    return res

                if selection is not None:
                    # Evaluate only the val-selected configs, without grid heuristics.
                    cfgs = task_sel["top5"] if ensemble == "top5" else [task_sel["rank1"]]
                    if any(c["eval_mode"] != mode for c in cfgs):
                        print(f"  Error: {ds_config}: selected configs mix eval_modes, skipping")
                        continue
                    members = ";".join(f"{c['max_context']}:{c['patch_size']}" for c in cfgs)
                    print(f"\nEvaluating: {ds_name} ({term}) [{mode}] split={split} "
                          f"{'top5 median of' if ensemble else 'val rank-1'} ctx:ps={members}")
                    try:
                        if ensemble == "top5":
                            if fast:
                                res, _ = run_fast(cfgs, is_ensemble=True)
                            else:
                                res = run(EnsemblePredictor(
                                    [make_predictor(c["max_context"], c["patch_size"]) for c in cfgs],
                                    noise_seed=seed))
                            append_csv(csv_path, ds_config, "amets_top5_median", res, domain,
                                       num_variates, mode, "", split, "", counts, ckpt_sha,
                                       selection="val_top5_median", members=members)
                        else:
                            c = cfgs[0]
                            if fast:
                                res, _ = run_fast([c], is_ensemble=False)
                            else:
                                res = run(make_predictor(c["max_context"], c["patch_size"]))
                            append_csv(csv_path, ds_config, "amets", res, domain, num_variates,
                                       mode, c["patch_size"], split, c["max_context"], counts,
                                       ckpt_sha, selection="val_rank1", members=members)
                    except Exception as e:
                        print(f"  Error: {ds_config} ({mode}, split={split}): {e}")
                        import traceback
                        traceback.print_exc()
                    continue

                # Skip a max_context well above the dataset's minimum series length.
                # get_context_length caps the context at the series length, so the
                # result would match a smaller context.
                # The smallest context (64) always runs. For very short series, such
                # as m4_yearly, every context exceeds the length, and the smallest one
                # runs with the internal cap to cover them.
                SMALLEST_GRID_CONTEXT = 64
                if (min_series_len is not None and max_context > min_series_len * 1.1
                        and max_context > SMALLEST_GRID_CONTEXT):
                    print(f"  Skipping {ds_name} ({term}, {mode}): max_context={max_context} > series_len={min_series_len}")
                    continue

                run_patch_sizes = grid_patch_sizes(valid_patch_sizes, min_series_len)
                for ps in valid_patch_sizes:
                    # Skip this patch_size if even the smallest useful context exceeds the series length
                    if ps not in run_patch_sizes:
                        print(f"  Skipping patch_size={ps} for {ds_name} ({term}): series too short, {min_series_len} < {ps*2}")
                        continue

                    print(f"\nEvaluating: {ds_name} ({term}) [{mode}] split={split} patch_size={ps}")
                    print(f"  freq={dataset.freq}, pred_len={dataset.prediction_length}, variates={effective_variates}")

                    try:
                        if fast:
                            res, first_samples = run_fast(
                                [{"max_context": max_context, "patch_size": ps}], is_ensemble=False,
                                keep_samples=50 if ps == valid_patch_sizes[0] else 0)
                        else:
                            predictor = make_predictor(max_context, ps)
                            res = run(predictor)

                        # Save to CSV
                        append_csv(csv_path, ds_config, "amets", res, domain, num_variates, mode, ps,
                                   split, max_context, counts, ckpt_sha)

                        # Save forecasts for the first patch size only
                        if ps == valid_patch_sizes[0]:
                            try:
                                if fast:
                                    # These match the per-item prediction of the first 50
                                    # items: same seed, same draws, SampleForecast.mean.
                                    forecasts = list(np.mean(first_samples, axis=1))
                                    test_label = [{"target": t} for t in task_cache["task"].labels[:50]]
                                else:
                                    test_input = list(eval_data.input)[:50]
                                    test_label = list(eval_data.label)[:50]
                                    sample_forecasts = predictor.predict(test_input)
                                    forecasts = [f.mean for f in sample_forecasts]
                                actuals = []
                                for label_entry in test_label:
                                    actual = np.array(label_entry["target"])
                                    if actual.ndim == 2 and effective_variates > 1:
                                        actual = actual.T
                                    actuals.append(actual)

                                fc_dir = os.path.join(forecasts_dir, split, ds_name.replace("/", "_"), term, mode)
                                os.makedirs(fc_dir, exist_ok=True)
                                with open(os.path.join(fc_dir, "forecasts.pkl"), "wb") as f:
                                    pickle.dump(np.array(forecasts), f)
                                with open(os.path.join(fc_dir, "actuals.pkl"), "wb") as f:
                                    pickle.dump(np.array(actuals), f)
                                print(f"  Saved forecasts ({mode}): {np.array(forecasts).shape}")
                            except Exception as e:
                                print(f"  Warning: Could not save forecasts: {e}")

                    except Exception as e:
                        print(f"  Error: {ds_name} ({term}, {mode}, ps={ps}): {e}")
                        import traceback
                        traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"Evaluation complete. Results: {csv_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="AMETS Unified Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--datasets", type=str, nargs="+", default=["m4_weekly"],
                        help="GIFT-Eval dataset configs to evaluate.")
    parser.add_argument("--terms", type=str, nargs="+", default=["short", "medium", "long"],
                        help="Forecast terms to evaluate.")
    parser.add_argument("--output_dir", type=str, default="results/amets_eval",
                        help="Root directory for metrics and forecasts.")
    parser.add_argument("--eval_mode", type=str, default="both",
                        choices=["univariate", "multivariate", "both"],
                        help="univariate, multivariate, or both. Default: both.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_context", type=int, default=None,
                        help="Context length for grid mode. Default: 512. With --selection, "
                             "each selected config carries its own max_context.")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--split", type=str, required=True, choices=["val", "test"],
                        help="val: the windows just before GIFT-Eval's test windows. "
                             "test: GIFT-Eval test windows, for final reporting only.")
    parser.add_argument("--selection", type=str, default=None,
                        help="selection.json from select_configs.py. Evaluates only the "
                             "val-selected (task, max_context, eval_mode, patch_size).")
    parser.add_argument("--ensemble", type=str, default=None, choices=["top5"],
                        help="With --selection, use the median of the val top-5 configs' point forecasts.")
    parser.add_argument("--checkpoint_sha256", type=str, default=None,
                        help="Precomputed sha256 of --checkpoint. evaluate_protocol.sh hashes "
                             "it once. Computed here when omitted.")
    parser.add_argument("--per_item_eval", action="store_true",
                        help="Use the per-item predictor with gluonts evaluate_model. This slow "
                             "reference path computes the same numbers as the default fast path.")
    parser.add_argument("--allow-unselected-test", action="store_true",
                        help="Permit --split test without --selection. Never use it for selection.")
    args = parser.parse_args()

    if args.split == "test" and args.selection is None and not args.allow_unselected_test:
        parser.error("--split test needs --selection, with configs chosen on val. A test grid "
                     "must never be used for selection. Pass --allow-unselected-test only "
                     "for diagnostics that feed no decision.")
    if args.ensemble is not None and args.selection is None:
        parser.error("--ensemble requires --selection")
    if args.selection is not None and args.max_context is not None:
        parser.error("--max_context conflicts with --selection, whose configs carry their own")
    selection = load_selection(args.selection) if args.selection else None

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        datasets=args.datasets,
        terms=args.terms,
        output_dir=args.output_dir,
        split=args.split,
        eval_mode=args.eval_mode,
        device=args.device,
        batch_size=args.batch_size,
        max_context=args.max_context if args.max_context is not None else 512,
        num_samples=args.num_samples,
        selection=selection,
        ensemble=args.ensemble,
        checkpoint_sha256=args.checkpoint_sha256,
        fast=not args.per_item_eval,
    )


if __name__ == "__main__":
    main()
