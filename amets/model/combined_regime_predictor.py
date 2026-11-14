"""
CombinedRegimePredictor: per-descriptor predictors wrapped as one module.

The module holds three learned CNNs for forecastability, seasonality, and trend,
and an analytic sparsity measure. This file defines the architecture and
checkpoint loading only. It does not depend on the feature pipeline.
"""

import os

import numpy as np
import torch
import torch.nn as nn


class SingleFeaturePredictor(nn.Module):
    """Predict one regime feature from a raw time series."""

    def __init__(self, hidden_dim=128):
        super().__init__()

        self.scale1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.MaxPool1d(2, 2),
        )
        self.scale2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, padding=5),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.MaxPool1d(2, 2),
        )
        self.scale3 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=21, padding=10),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 128, kernel_size=21, padding=10),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.MaxPool1d(2, 2),
        )

        self.attn_norm = nn.LayerNorm(128)
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True, dropout=0.1)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(128 * 3 + 128, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, lengths=None):
        """Return logits of shape (batch,) for x of shape (batch, T).

        With ``lengths`` of shape (batch,), the method reads row i as the
        right-padded series ``x[i, :lengths[i]]``. The result equals the unpadded
        forward of that row up to float rounding. Lengths must be >= 4.
        """
        if lengths is not None:
            return self._forward_masked(x, lengths)
        x = x.unsqueeze(1)
        xd = x.to(_norm_dtype(x))
        xd = xd - xd[..., :1]  # Shifting by the first value keeps the z-score and improves conditioning.
        mean = xd.mean(dim=-1, keepdim=True)
        std = xd.std(dim=-1, keepdim=True).clamp(min=1e-6)
        x = ((xd - mean) / std).to(x.dtype)

        c1 = self.scale1(x)
        c2 = self.scale2(x)
        c3 = self.scale3(x)

        f1 = self.pool(c1).squeeze(-1)
        f2 = self.pool(c2).squeeze(-1)
        f3 = self.pool(c3).squeeze(-1)

        attn_in = c1.transpose(1, 2)
        attn_in = self.attn_norm(attn_in)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        f_attn = attn_out.mean(dim=1)

        feat = torch.cat([f1, f2, f3, f_attn], dim=1)
        return self.head(feat).squeeze(-1)

    def _forward_masked(self, x, lengths):
        """Run the masked forward over right-padded rows. See ``forward``."""
        lengths = lengths.to(device=x.device, dtype=torch.long)
        mask = _length_mask(lengths, x.shape[-1], x.dtype)
        dt = _norm_dtype(x)
        md = mask.to(dt)
        xd = x.unsqueeze(1).to(dt)
        xd = torch.where(mask > 0, xd - xd[..., :1], torch.zeros_like(md))
        n = lengths.to(dt)[:, None, None]
        mean = xd.sum(dim=-1, keepdim=True) / n
        var = (((xd - mean) * md) ** 2).sum(dim=-1, keepdim=True) / (n - 1)
        x = ((xd - mean) / var.sqrt().clamp(min=1e-6) * md).to(x.dtype)

        # Disable NNPACK, which is slow for these batched convolutions on CPU.
        with torch.backends.nnpack.flags(enabled=False):
            branches = [_masked_branch(s, x, lengths)
                        for s in (self.scale1, self.scale2, self.scale3)]
        out_len = branches[0][1]
        n_out = out_len.to(x.dtype)[:, None]
        pooled = [h.sum(dim=-1) / n_out for h, _ in branches]

        c1 = branches[0][0]
        valid = _length_mask(out_len, c1.shape[-1], c1.dtype)[:, 0]  # (batch, T')
        attn_in = self.attn_norm(c1.transpose(1, 2))
        attn_out, _ = self.attn(attn_in, attn_in, attn_in,
                                key_padding_mask=valid == 0, need_weights=False)
        f_attn = (attn_out * valid[..., None]).sum(dim=1) / n_out

        feat = torch.cat(pooled + [f_attn], dim=1)
        return self.head(feat).squeeze(-1)


def _norm_dtype(x):
    """Return the dtype for input z-normalization: float64, or float32 on mps, which lacks float64."""
    return torch.float32 if x.device.type == "mps" else torch.float64


def _length_mask(lengths, T, dtype):
    """Return a (batch, 1, T) mask that is 1 where the position is less than the length."""
    return (torch.arange(T, device=lengths.device)[None] < lengths[:, None]).unsqueeze(1).to(dtype)


def _masked_group_norm(gn, h, mask):
    """Apply GroupNorm with statistics computed over valid positions only."""
    B, C, T = h.shape
    G = gn.num_groups
    hg = h.view(B, G, C // G, T)
    m = mask.view(B, 1, 1, T)
    n = m.sum(dim=(2, 3), keepdim=True) * (C // G)
    mu = (hg * m).sum(dim=(2, 3), keepdim=True) / n
    var = (((hg - mu) * m) ** 2).sum(dim=(2, 3), keepdim=True) / n
    y = ((hg - mu) / torch.sqrt(var + gn.eps)).view(B, C, T)
    return y * gn.weight[None, :, None] + gn.bias[None, :, None]


def _masked_branch(seq, x, lengths):
    """Run one conv branch with padding masked.

    The branch applies conv, GroupNorm, GELU, and pool twice.

    Returns the output with invalid positions zeroed and the output lengths.
    """
    conv1, gn1, act1, pool1, conv2, gn2, act2, pool2 = seq
    for conv, gn, act, pool in ((conv1, gn1, act1, pool1), (conv2, gn2, act2, pool2)):
        mask = _length_mask(lengths, x.shape[-1], x.dtype)
        h = conv(x * mask)
        x = pool(act(_masked_group_norm(gn, h, mask)) * mask)
        lengths = lengths // 2
    return x * _length_mask(lengths, x.shape[-1], x.dtype), lengths


class SparsityPredictor(nn.Module):
    """Compute sparsity analytically with no learned parameters.

    Sparsity = 1 - n_unique_values / length. A mostly constant series has high
    sparsity.
    """

    def forward(self, x, lengths=None):
        """Return sparsity of shape (batch,) in [0, 1] for x of shape (batch, T).

        The computation ignores NaNs. With ``lengths``, it reads only
        ``x[i, :lengths[i]]``. Rows with no valid value get 0.
        """
        T = x.shape[-1]
        valid = ~torch.isnan(x)
        if lengths is not None:
            valid &= torch.arange(T, device=x.device)[None] < lengths.to(x.device)[:, None]
        s = torch.where(valid, x, torch.full_like(x, float("inf"))).sort(dim=-1).values
        n = valid.sum(dim=-1)
        pos = torch.arange(1, T, device=x.device)[None]
        n_unique = ((s[:, 1:] != s[:, :-1]) & (pos < n[:, None])).sum(dim=-1) + (n > 0).long()
        dt = torch.float32 if x.device.type == "mps" else torch.float64  # mps has no float64
        sparsity = 1.0 - n_unique.to(dt) / n.clamp(min=1).to(dt)
        # Sparsity is already in [0, 1], so forward returns it as is and not as a logit.
        return torch.where(n > 0, sparsity, torch.zeros_like(sparsity)).float()


class CombinedRegimePredictor(nn.Module):
    """Three learned CNN predictors and one analytic sparsity computation.

    Output order: [forecastability, seasonality, trend, sparsity].

    - Features 0 to 2 come from the learned CNNs.
    - Feature 3 comes from the analytic computation, which has no parameters.
    """

    def __init__(self):
        super().__init__()
        # forecastability, seasonality, trend
        self.predictors = nn.ModuleList([SingleFeaturePredictor() for _ in range(3)])
        self.sparsity = SparsityPredictor()
        self.num_regimes = 4

    def forward(self, x, lengths=None):
        """Return logits of shape (batch, 4).

        The first three are CNN logits and the last is raw sparsity. The optional
        ``lengths`` marks rows of x as right-padded. See
        ``SingleFeaturePredictor.forward``.
        """
        logits = [p(x, lengths) for p in self.predictors]
        logits.append(self.sparsity(x, lengths))
        return torch.stack(logits, dim=-1)

    def get_probabilities(self, x, lengths=None):
        """Return probabilities of shape (batch, 4).

        The first three are sigmoids of the CNN logits. The last is raw sparsity,
        which is already in [0, 1].
        """
        logits = [p(x, lengths) for p in self.predictors]
        probs_3 = torch.sigmoid(torch.stack(logits, dim=-1))  # (batch, 3)
        sparsity = self.sparsity(x, lengths).unsqueeze(-1)  # (batch, 1)
        return torch.cat([probs_3, sparsity], dim=-1)  # (batch, 4)

    def load(self, filepath, device=None):
        """Load a combined checkpoint.

        The method accepts checkpoints with three or four learned predictors.
        This class computes sparsity analytically, so the method ignores a fourth
        CNN. It raises ValueError on any other missing or unexpected key.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Regime predictor checkpoint not found: {filepath}\n"
                "This checkpoint is not distributed with the repository. Either "
                "train one with `python regime_predictor/train.py`, or set "
                "model.regime_predictor_path=null to start from a randomly "
                "initialized predictor. See "
                "regime_predictor/trained_predictors/README.md for details."
            )
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        # Keep only the three learned predictors. Index 3 holds a sparsity CNN
        # that some checkpoints include. This class uses the analytic measure.
        filtered = {}
        for k, v in state_dict.items():
            if k.startswith('predictors.3.'):
                continue
            filtered[k] = v
        result = self.load_state_dict(filtered, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(
                f"Regime predictor checkpoint {filepath} does not match "
                f"CombinedRegimePredictor: missing {result.missing_keys}, "
                f"unexpected {result.unexpected_keys}."
            )
        print(f"Loaded regime predictor from {filepath} (3 CNN + analytic sparsity)")


REGIME_MAX_LEN = 2048  # Each context keeps only its last REGIME_MAX_LEN points.
REGIME_MIN_LEN = 8  # Shorter contexts get left zero padding up to this length.
REGIME_CHUNK_ELEMS = 2 ** 18  # Maximum padded points per batched predictor call.
REGIME_MIN_FILL = 0.5  # A chunk's shortest row is at least this share of its longest.


def stack_contexts(series):
    """Build regime predictor input for a list of 1D contexts.

    Both training stages use this function. It crops each context to its last
    REGIME_MAX_LEN points, left-pads it with zeros to REGIME_MIN_LEN, and replaces
    NaN with 0. It right-pads every row with zeros to the longest row.

    Returns:
        X (n, T) float32 and lengths (n,) int64 numpy arrays, for
        ``predictor(X, lengths)``.
    """
    lengths = np.array([max(min(len(s), REGIME_MAX_LEN), REGIME_MIN_LEN) for s in series],
                       dtype=np.int64)
    X = np.zeros((len(series), lengths.max() if len(series) else 0), dtype=np.float32)
    for row, (s, n) in enumerate(zip(series, lengths)):
        s = np.asarray(s)[-REGIME_MAX_LEN:]
        X[row, n - len(s):n] = s
    return np.nan_to_num(X, nan=0.0), lengths


def _series_probs_loop(predictor, series, device):
    """Run one forward with batch size 1 per context as the reference path."""
    out = []
    for ctx in series:
        x = torch.nan_to_num(torch.from_numpy(ctx).float().to(device), nan=0.0)
        x = x[-REGIME_MAX_LEN:]
        if len(x) < REGIME_MIN_LEN:
            x = torch.nn.functional.pad(x, (REGIME_MIN_LEN - len(x), 0), value=0.0)
        out.append(predictor.get_probabilities(x.unsqueeze(0)).squeeze(0))
    return torch.stack(out)


def series_regime_probs(predictor, series, device, batched=True,
                        chunk_elems=REGIME_CHUNK_ELEMS, min_fill=REGIME_MIN_FILL):
    """Return regime probabilities of shape (n, 4) for a list of 1D numpy contexts.

    With ``batched=True`` the function sorts contexts by length and runs the
    masked forward in chunks of at most ``chunk_elems`` padded points. It makes
    one host-to-device copy each for data and lengths. With ``batched=False`` it
    runs one forward per context. Both give the same outputs up to float rounding.
    """
    if not batched:
        return _series_probs_loop(predictor, series, device)
    order = np.argsort([-min(len(s), REGIME_MAX_LEN) for s in series], kind="stable")
    X, lengths = stack_contexts([series[j] for j in order])
    X = torch.from_numpy(X).to(device)
    L = torch.from_numpy(lengths).to(device)
    probs = []
    start = 0
    while start < len(series):
        T = int(lengths[start])
        stop = min(len(series), start + max(1, chunk_elems // T))
        stop = start + max(1, int((lengths[start:stop] >= min_fill * T).sum()))
        probs.append(predictor.get_probabilities(X[start:stop, :T], L[start:stop]))
        start = stop
    inverse = torch.from_numpy(np.argsort(order)).to(device)
    return torch.cat(probs)[inverse]


def context_regime_probs(predictor, context_target_list, sample_id, variate_id,
                         num_regimes, batched=True):
    """Return token-level regime probabilities for a packed batch.

    Args:
        predictor: CombinedRegimePredictor. The caller sets eval mode.
        context_target_list: One dict per packed sample, {variate_id: 1D float array}.
        sample_id, variate_id: (batch, seq_len) packing ids. sample_id 0 marks padding.
        num_regimes: Number of regime probabilities per token.
        batched: Masked batched forward if True, per-series loop if False.

    Returns:
        Tensor of shape (batch, seq_len, num_regimes). Every token of a
        (sample, variate) gets that context's probabilities. Padding tokens get
        zeros.
    """
    batch_size, seq_len = sample_id.shape
    device = sample_id.device
    keys = [(i, v) for i, d in enumerate(context_target_list) for v in d]
    series = [context_target_list[i][v] for i, v in keys]
    max_variates = max(v for _, v in keys) + 1

    with torch.no_grad():
        probs = torch.nan_to_num(series_regime_probs(predictor, series, device, batched), nan=0.0)

    idx = torch.tensor(keys, device=device).T  # (2, n_series): sample, variate
    lookup = torch.zeros(len(context_target_list), max_variates, num_regimes, device=device)
    lookup[idx[0], idx[1]] = probs

    # Global sample index of each token: samples in earlier bins + (sample_id - 1).
    per_bin = sample_id.max(dim=1).values
    offset = (torch.cumsum(per_bin, 0) - per_bin)[:, None]
    valid = sample_id > 0
    global_idx = (offset + sample_id - 1)[valid]

    regime_probs = torch.zeros(batch_size, seq_len, num_regimes, device=device)
    regime_probs[valid] = lookup[global_idx, variate_id[valid]]
    return regime_probs
