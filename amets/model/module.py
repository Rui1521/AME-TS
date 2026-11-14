#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
AMETS network module.

This module implements the AMETS network: a transformer with structure-aware
expert routing. It outputs point predictions through output_proj and supports
an L1 or L2 loss.
"""

from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Bool, Float, Int
from torch import nn

from amets.common.torch_util import packed_causal_attention_mask, mask_fill
from amets.module.norm import RMSNorm
from amets.module.packed_scaler import PackedNOPScaler, PackedStdScaler
from amets.module.position import (
    BinaryAttentionBias,
    QueryKeyProjection,
    RecencyAttentionBias,
    RotaryProjection,
)
from amets.module.transformer_moe import TransformerEncoderAnchored
from amets.module.ts_embed import FeatLinear, MultiInSizeLinear, MultiOutSizeLinear

from .combined_regime_predictor import CombinedRegimePredictor


class AMETSModule(nn.Module, PyTorchModelHubMixin):
    """
    AMETS: a time series foundation model with structure-aware expert routing.
    
    Components:
    1. Regime predictor that infers structural descriptors of each series
    2. Mapping of descriptors to an expert prior
    3. Transformer whose MoE layers use that prior to guide expert routing
    4. Point prediction output
    5. L1 or L2 loss
    """
    
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        patch_sizes: tuple[int, ...],
        max_seq_len: int,
        attn_dropout_p: float,
        dropout_p: float,
        d_ff: Optional[int] = None,
        scaling: bool = True,
        pre_norm: bool = True,
        time_recency_bias: bool = False,
        # MoE parameters
        num_experts: int = 32,
        num_experts_per_tok: int = 2,
        # Routing parameters
        lambda_max: float = 1.0,
        l0: int = 0,
        lambda_schedule: str = "linear_increase",
        shared_expert_ratio: float = 0.25,
        num_regimes: int = 4,
        regime_predictor_hidden_dim: int = 64,
        regime_predictor_dropout: float = 0.2,
        regime_predictor_path: Optional[str] = None,
        freeze_regime_predictor: bool = False,
        # Loss configuration
        loss_type: str = "l1",  # "l1" or "l2"
        # Routing mode configuration
        routing_mode: str = "additive",
        kl_loss_coef: float = 0.1,
        kl_divergence_type: str = "js",
        # Shared gate parameters
        shared_gate_mode: str = "trainable",
        shared_gate_alpha: float = 1.0,
        shared_gate_beta: float = 0.5,
        # Expert diversity
        ortho_loss_coef: float = 0.0,
    ):
        """
        Initialize AMETS module.

        Args:
            d_model: Model hidden dimensions
            num_layers: Number of transformer layers
            patch_sizes: Sequence of patch sizes
            max_seq_len: Maximum sequence length for inputs
            attn_dropout_p: Dropout probability for attention layers
            dropout_p: Dropout probability for all other layers
            d_ff: Feed-forward hidden dimensions
            scaling: Whether to standardize inputs
            pre_norm: Whether transformer blocks are pre-norm (True) or post-norm (False)
            time_recency_bias: Add RecencyAttentionBias on time ids alongside RoPE
            num_experts: Number of experts in MoE layers
            num_experts_per_tok: Number of experts per token
            lambda_max: Maximum prior weight (default: 1.0)
            l0: Layer to start prior injection from (default: 0)
            lambda_schedule: Lambda schedule type
            shared_expert_ratio: Sizes only the placeholder routing stats.
                MoEAnchoredFeedForward sets the shared and specialized split.
            num_regimes: Number of regime predictions
            regime_predictor_hidden_dim: Unused. CombinedRegimePredictor defines its own architecture.
            regime_predictor_dropout: Unused. CombinedRegimePredictor defines its own architecture.
            regime_predictor_path: Path to pre-trained regime predictor
            freeze_regime_predictor: Whether to freeze regime predictor
            loss_type: Loss type, "l1" or "l2"
            routing_mode: Routing mode, "additive" or "kl_divergence"
            kl_loss_coef: Coefficient for KL divergence loss
            kl_divergence_type: Type of divergence, "forward", "reverse", or "js"
        """
        super().__init__()

        # Store configuration
        self.d_model = d_model
        self.num_layers = num_layers
        self.patch_sizes = patch_sizes
        self.max_seq_len = max_seq_len
        self.scaling = scaling
        self.num_experts = num_experts
        self.loss_type = loss_type

        # Routing parameters
        self.lambda_max = lambda_max
        self.l0 = l0
        self.lambda_schedule = lambda_schedule
        self.shared_expert_ratio = shared_expert_ratio
        self.num_regimes = num_regimes
        self.routing_mode = routing_mode
        self.kl_loss_coef = kl_loss_coef
        self.kl_divergence_type = kl_divergence_type
        
        # Scaler for input normalization
        self.scaler = PackedStdScaler(minimum_scale=1e-6) if scaling else PackedNOPScaler()

        # Learned mask embedding that hides the ground truth at forecast positions
        self.mask_encoding = nn.Embedding(num_embeddings=1, embedding_dim=d_model)
        
        # Input projections
        self.in_proj = MultiInSizeLinear(
            in_features_ls=patch_sizes,
            out_features=d_model,
        )
        self.res_proj = MultiInSizeLinear(
            in_features_ls=patch_sizes,
            out_features=d_model,
        )
        self.feat_proj = FeatLinear(
            in_features_ls=patch_sizes,
            out_features=d_model,
        )
        
        # Transformer encoder with anchored MoE layers
        self.encoder = TransformerEncoderAnchored(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=None,  # Computed as d_model // 64
            pre_norm=pre_norm,
            attn_dropout_p=attn_dropout_p,
            dropout_p=dropout_p,
            norm_layer=RMSNorm,
            activation=F.silu,
            use_moe=True,
            use_glu=True,
            use_qk_norm=True,
            var_attn_bias_layer=partial(BinaryAttentionBias),
            time_attn_bias_layer=partial(RecencyAttentionBias) if time_recency_bias else None,
            time_qk_proj_layer=partial(
                QueryKeyProjection,
                proj_layer=RotaryProjection,
                kwargs=dict(max_len=max_seq_len),
                partial_factor=(0.0, 0.5),
            ),
            shared_var_attn_bias=False,
            shared_time_qk_proj=True,
            d_ff=d_ff,
            # MoE parameters
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            # Routing parameters
            lambda_max=lambda_max,
            l0=l0,
            lambda_schedule=lambda_schedule,
            routing_mode=routing_mode,
            kl_loss_coef=kl_loss_coef,
            kl_divergence_type=kl_divergence_type,
            num_regimes=num_regimes,
            shared_gate_mode=shared_gate_mode,
            shared_gate_alpha=shared_gate_alpha,
            shared_gate_beta=shared_gate_beta,
            ortho_loss_coef=ortho_loss_coef,
        )
        
        # Output projection from d_model to patch_size for point prediction
        self.output_proj = MultiOutSizeLinear(
            in_features=d_model,
            out_features_ls=patch_sizes,
            dim=1,  # One value per patch element
        )
        
        # Structural descriptor predictor with one model per descriptor
        self.regime_predictor = CombinedRegimePredictor()
        if regime_predictor_path is not None:
            print(f"Loading regime predictor from {regime_predictor_path}")
            self.regime_predictor.load(regime_predictor_path, device="cpu")
        
        # Optionally freeze regime predictor
        if freeze_regime_predictor:
            for param in self.regime_predictor.parameters():
                param.requires_grad = False
            print("Regime predictor frozen (not trainable)")
        
        print(f"Initialized AMETSModule with:")
        print(f"  - d_model: {d_model}")
        print(f"  - num_layers: {num_layers}")
        print(f"  - patch_sizes: {patch_sizes}")
        print(f"  - Loss type: {loss_type}")
        print(f"  - Lambda schedule: {lambda_schedule}")
        print(f"  - Lambda max: {lambda_max}")
        print(f"  - Num regimes: {num_regimes}")
    
    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],
        regime_probs: Optional[torch.Tensor] = None,
    ) -> Float[torch.Tensor, "*batch seq_len max_patch"]:
        """
        Forward pass with prior-guided routing and point prediction.
        
        Args:
            target: Patched input data from the transforms
            observed_mask: Binary mask for missing values (1 if observed, 0 otherwise)
            sample_id: Sample index of each token, used for packing
            time_id: Indices indicating the time index
            variate_id: Indices indicating the variate index
            prediction_mask: Binary mask for prediction horizon
            patch_size: Patch size for each token
            regime_probs: Precomputed regime probabilities
            
        Returns:
            Tuple of point predictions with the same shape as target, load balance
            loss, KL loss, and orthogonality loss.
        """
        # 1. Scaling
        loc, scale = self.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),
            sample_id,
            variate_id,
        )
        scaled_target = (target - loc) / scale
        
        # Zero-variance series can produce NaN or Inf after scaling.
        if torch.isnan(scaled_target).any() or torch.isinf(scaled_target).any():
            scaled_target = torch.nan_to_num(scaled_target, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. Patching and projection
        in_reprs = self.in_proj(scaled_target, patch_size)
        in_reprs = F.silu(in_reprs)
        in_reprs = self.feat_proj(in_reprs, patch_size)
        
        res_reprs = self.res_proj(scaled_target, patch_size)
        
        reprs = in_reprs + res_reprs

        # 3. Replace forecast position representations with the learned mask
        # embedding so that they do not carry the ground truth.
        reprs = mask_fill(reprs, prediction_mask, self.mask_encoding.weight)

        # 4. Transformer with anchored MoE layers. The MoE layers in ffn_moe.py
        # map regime_probs to an expert prior.
        reprs, total_lb_loss, total_kl_loss, total_ortho_loss = self.encoder(
            reprs,
            packed_causal_attention_mask(sample_id, time_id),
            var_id=variate_id,
            time_id=time_id,
            regime_prior=regime_probs,  # (batch, seq_len, num_regimes)
        )

        # 5. Output projection for point prediction
        output = self.output_proj(reprs, patch_size)

        # 6. Unscale predictions
        output = output * scale + loc

        return output, total_lb_loss, total_kl_loss, total_ortho_loss
    
    def compute_loss(
        self,
        pred: Float[torch.Tensor, "*batch seq_len max_patch"],
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        patch_size: Optional[Int[torch.Tensor, "*batch seq_len"]] = None,
    ) -> Float[torch.Tensor, ""]:
        """
        Compute loss (L1 or L2) on prediction positions.

        Args:
            pred: Model predictions
            target: Ground truth values
            prediction_mask: Binary mask for prediction positions
            observed_mask: Binary mask for observed values
            patch_size: Per-token patch size, used to exclude zero padding from the loss

        Returns:
            Scalar loss value
        """
        # Combine the masks to select the positions that enter the loss.
        mask = prediction_mask.unsqueeze(-1) * ~observed_mask

        # Exclude zero-padding positions beyond the actual patch size.
        if patch_size is not None:
            max_patch = pred.shape[-1]
            # Compare (batch, seq_len) patch sizes against (batch, seq_len, max_patch) positions.
            pos_indices = torch.arange(max_patch, device=pred.device).expand_as(pred)
            patch_mask = pos_indices < patch_size.unsqueeze(-1)
            mask = mask * patch_mask

        if self.loss_type == "l1":
            loss = (pred - target).abs() * mask
        elif self.loss_type == "l2":
            loss = (pred - target).pow(2) * mask
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}. Must be 'l1' or 'l2'")
        
        loss = torch.clamp(loss, max=1e5)
        # Average over valid positions
        num_valid = mask.sum()
        if num_valid > 0:
            loss = loss.sum() / num_valid
        else:
            loss = loss.sum()  # The sum is 0 here.
        return loss

    def compute_recon_loss(
        self,
        pred: Float[torch.Tensor, "*batch seq_len max_patch"],
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
    ) -> Float[torch.Tensor, ""]:
        """
        Compute the reconstruction loss on context positions.

        The MaskedPrediction transform sets observed_mask to False at every forecast
        position, so observed_mask is True only at observed context positions.
        """
        mask = observed_mask

        if self.loss_type == "l1":
            loss = (pred - target).abs() * mask
        elif self.loss_type == "l2":
            loss = (pred - target).pow(2) * mask
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        loss = torch.clamp(loss, max=1e5)
        num_valid = mask.sum()
        if num_valid > 0:
            loss = loss.sum() / num_valid
        else:
            loss = loss.sum()
        return loss

    @torch.no_grad()
    def get_routing_info(self, batch: dict) -> Optional[dict]:
        """
        Get routing statistics for logging.
        
        Args:
            batch: Training batch with target, observed_mask, etc.
            
        Returns:
            Dictionary with routing statistics or None if not available
        """
        try:
            # Get routing stats from encoder
            routing_stats = self.encoder.get_routing_stats()
            
            if routing_stats is None:
                return self._get_basic_routing_info()
            
            return routing_stats
            
        except Exception as e:
            print(f"Warning: Could not get routing info: {e}")
            return self._get_basic_routing_info()
    
    def _get_basic_routing_info(self) -> dict:
        """Return placeholder routing info when detailed stats are not available."""
        # Compute lambda values per layer
        lambda_per_layer = []
        for layer_idx in range(self.num_layers):
            if self.lambda_schedule == "linear_increase":
                if layer_idx < self.l0:
                    lambda_val = 0.0
                else:
                    lambda_val = self.lambda_max * (layer_idx - self.l0) / max(1, self.num_layers - self.l0 - 1)
            elif self.lambda_schedule == "linear_decrease":
                if layer_idx < self.l0:
                    lambda_val = 0.0
                else:
                    lambda_val = self.lambda_max * (1 - (layer_idx - self.l0) / max(1, self.num_layers - self.l0 - 1))
            else:  # constant
                lambda_val = self.lambda_max if layer_idx >= self.l0 else 0.0
            lambda_per_layer.append(lambda_val)
        
        # Build placeholder stats
        num_shared = int(self.num_experts * self.shared_expert_ratio)
        num_specialized = self.num_experts - num_shared
        
        return {
            'regime_probs': None,  # Not available without forward pass
            'expert_usage_per_layer': [
                [1.0 / self.num_experts] * self.num_experts 
                for _ in range(self.num_layers)
            ],
            'shared_mass_gate_per_layer': [0.5] * self.num_layers,
            'entropy_per_layer': [1.0] * self.num_layers,
            'lambda_per_layer': lambda_per_layer,
            'alpha': 1.0,
            'beta': 0.0,
            'num_shared_experts': num_shared,
            'num_specialized_experts': num_specialized,
        }

