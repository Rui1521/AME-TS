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
Transformer encoder with anchored MoE layers.

This module extends the standard Transformer with prior-guided routing in the
MoE layers. It passes regime distributions through the network to guide expert
selection at each layer.
"""

from collections.abc import Callable
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import nn

from .attention import GroupedQueryAttention
from .ffn import FeedForward, GatedLinearUnitFeedForward
from .ffn_moe import MoEAnchoredFeedForward
from .position import AttentionBias, QueryKeyProjection


class TransformerEncoderLayerAnchored(nn.Module):
    """
    Transformer encoder layer with an anchored MoE feed-forward block.
    
    The layer matches a standard TransformerEncoderLayer but passes
    regime_prior and lambda_anchor to the FFN for prior-guided routing.
    """
    
    def __init__(
        self,
        self_attn: GroupedQueryAttention,
        ffn: FeedForward,
        norm1: Optional[nn.Module],
        norm2: Optional[nn.Module],
        post_attn_dropout_p: float = 0.0,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.dropout_p = post_attn_dropout_p
        
        self.self_attn = self_attn
        self.ffn = ffn
        self.norm1 = norm1 or nn.Identity()
        self.norm2 = norm2 or nn.Identity()
        self.dropout = nn.Dropout(post_attn_dropout_p)
    
    def forward(
        self,
        x: Float[torch.Tensor, "*batch time_len dim"],
        attn_mask: Optional[Bool[torch.Tensor, "*batch time_len time_len"]] = None,
        var_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
        time_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
        regime_prior: Optional[Float[torch.Tensor, "*batch time_len expert"]] = None,
        lambda_anchor: float = 0.0,
    ) -> tuple[Float[torch.Tensor, "*batch time_len dim"], float, float, float]:
        """
        Forward pass with prior-guided routing.

        Args:
            x: Input tensor
            attn_mask: Attention mask (causal)
            var_id: Variate IDs
            time_id: Time IDs
            regime_prior: Regime-based expert preferences
            lambda_anchor: Prior interpolation weight

        Returns:
            Tuple of (output, load_balance_loss, kl_loss, ortho_loss)
        """
        lb_loss = 0.0
        kl_loss = 0.0
        ortho_loss = 0.0

        if self.pre_norm:
            x = x + self._sa_block(
                self.norm1(x), attn_mask, var_id=var_id, time_id=time_id
            )
            # Pass regime_prior and lambda_anchor to the FFN.
            if isinstance(self.ffn, MoEAnchoredFeedForward):
                ffn_out, ffn_lb_loss, ffn_kl_loss, ffn_ortho_loss = self.ffn(
                    self.norm2(x),
                    regime_prior=regime_prior,
                    lambda_anchor=lambda_anchor,
                )
                x = x + ffn_out
                lb_loss = ffn_lb_loss
                kl_loss = ffn_kl_loss
                ortho_loss = ffn_ortho_loss
            else:
                x = x + self.ffn(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, attn_mask, var_id=var_id, time_id=time_id)
            )
            if isinstance(self.ffn, MoEAnchoredFeedForward):
                ffn_out, ffn_lb_loss, ffn_kl_loss, ffn_ortho_loss = self.ffn(
                    x,
                    regime_prior=regime_prior,
                    lambda_anchor=lambda_anchor,
                )
                x = self.norm2(x + ffn_out)
                lb_loss = ffn_lb_loss
                kl_loss = ffn_kl_loss
                ortho_loss = ffn_ortho_loss
            else:
                x = self.norm2(x + self.ffn(x))

        return x, lb_loss, kl_loss, ortho_loss
    
    def _sa_block(
        self,
        x: Float[torch.Tensor, "*batch time_len dim"],
        attn_mask: Optional[Bool[torch.Tensor, "*batch time_len time_len"]],
        var_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
        time_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
    ) -> Float[torch.Tensor, "*batch time_len dim"]:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            query_var_id=var_id,
            kv_var_id=var_id,
            query_time_id=time_id,
            kv_time_id=time_id,
        )
        return self.dropout(x)


class TransformerEncoderAnchored(nn.Module):
    """
    Transformer encoder with anchored MoE layers.
    
    This encoder uses MoEAnchoredFeedForward layers and implements depth-aware
    lambda scheduling for prior-guided routing.
    """
    
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: Optional[int] = None,
        num_groups: Optional[int] = None,
        pre_norm: bool = True,
        attn_dropout_p: float = 0.0,
        dropout_p: float = 0.0,
        norm_layer: Optional[Callable[[int], nn.Module]] = nn.LayerNorm,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        use_moe: bool = True,
        use_glu: bool = True,
        use_qk_norm: bool = True,
        var_attn_bias_layer: Optional[Callable[[int, int, int], AttentionBias]] = None,
        time_attn_bias_layer: Optional[Callable[[int, int, int], AttentionBias]] = None,
        var_qk_proj_layer: Optional[
            Callable[[int, int, int], QueryKeyProjection]
        ] = None,
        time_qk_proj_layer: Optional[
            Callable[[int, int, int], QueryKeyProjection]
        ] = None,
        shared_var_attn_bias: bool = False,
        shared_time_attn_bias: bool = False,
        shared_var_qk_proj: bool = False,
        shared_time_qk_proj: bool = False,
        d_ff: Optional[int] = None,
        # MoE parameters
        num_experts: int = 32,
        num_experts_per_tok: int = 2,
        # Routing parameters
        lambda_max: float = 1.0,
        l0: int = 0,
        lambda_schedule: str = "linear_increase",
        routing_mode: str = "additive",
        kl_loss_coef: float = 0.1,
        kl_divergence_type: str = "js",
        num_regimes: int = 4,
        shared_gate_mode: str = "trainable",
        shared_gate_alpha: float = 1.0,
        shared_gate_beta: float = 0.5,
        ortho_loss_coef: float = 0.0,
    ):
        super().__init__()
        self.use_moe = use_moe
        self.num_layers = num_layers
        self.lambda_max = lambda_max
        self.l0 = l0
        self.lambda_schedule = lambda_schedule
        self.routing_mode = routing_mode
        self.kl_loss_coef = kl_loss_coef
        self.kl_divergence_type = kl_divergence_type
        self.num_regimes = num_regimes
        
        num_heads = num_heads or d_model // 64
        num_groups = num_groups or num_heads
        
        var_attn_bias = self.get_layer(
            d_model,
            num_heads,
            num_groups,
            var_attn_bias_layer,
            shared_var_attn_bias,
        )
        time_attn_bias = self.get_layer(
            d_model,
            num_heads,
            num_groups,
            time_attn_bias_layer,
            shared_time_attn_bias,
        )
        var_qk_proj = self.get_layer(
            d_model, num_heads, num_groups, var_qk_proj_layer, shared_var_qk_proj
        )
        time_qk_proj = self.get_layer(
            d_model, num_heads, num_groups, time_qk_proj_layer, shared_time_qk_proj
        )
        
        get_self_attn = partial(
            GroupedQueryAttention,
            dim=d_model,
            num_heads=num_heads,
            num_groups=num_groups,
            bias=False,
            norm_layer=norm_layer if use_qk_norm else None,
            softmax_scale=None,
            attn_dropout_p=attn_dropout_p,
            var_attn_bias=var_attn_bias,
            time_attn_bias=time_attn_bias,
            var_qk_proj=var_qk_proj,
            time_qk_proj=time_qk_proj,
        )
        
        if not use_moe:
            get_ffn = partial(
                GatedLinearUnitFeedForward if use_glu else FeedForward,
                in_dim=d_model,
                hidden_dim=d_ff,
                out_dim=None,
                activation=activation,
                bias=False,
                ffn_dropout_p=dropout_p,
            )
        else:
            # Anchored MoE with learned routing
            get_ffn = partial(
                MoEAnchoredFeedForward,
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_tok,
                in_dim=d_model,
                hidden_dim=d_ff,
                out_dim=None,
                activation=activation,
                bias=False,
                ffn_dropout_p=dropout_p,
                routing_mode=routing_mode,
                kl_loss_coef=kl_loss_coef,
                kl_divergence_type=kl_divergence_type,
                num_regimes=num_regimes,
                shared_gate_mode=shared_gate_mode,
                shared_gate_alpha=shared_gate_alpha,
                shared_gate_beta=shared_gate_beta,
                ortho_loss_coef=ortho_loss_coef,
            )
        
        get_encoder_layer_norm = partial(norm_layer, d_model)
        
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerAnchored(
                    self_attn=get_self_attn(),
                    ffn=get_ffn(),
                    norm1=get_encoder_layer_norm(),
                    norm2=get_encoder_layer_norm(),
                    pre_norm=pre_norm,
                    post_attn_dropout_p=dropout_p,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = norm_layer(d_model)
    
    @staticmethod
    def get_layer(
        dim: int,
        num_heads: int,
        num_groups: int,
        layer: Callable,
        shared_layer: bool,
    ) -> Optional[Callable[[], nn.Module]]:
        if layer is None:
            return None
        if shared_layer:
            module = layer(dim=dim, num_heads=num_heads, num_groups=num_groups)
            return lambda: module
        return partial(layer, dim=dim, num_heads=num_heads, num_groups=num_groups)
    
    def compute_lambda(self, layer_idx: int) -> float:
        """
        Compute depth-aware lambda for prior-guided routing.
        
        Supports three schedules:
        - "linear_increase": λ increases from 0 to λ_max (default)
        - "linear_decrease": λ decreases from λ_max to 0
        - "constant": λ = λ_max at all layers
        
        Args:
            layer_idx: Current layer index (0-indexed)
            
        Returns:
            Lambda value for this layer
        """
        if self.lambda_schedule == "constant":
            return self.lambda_max
        
        elif self.lambda_schedule == "linear_increase":
            if layer_idx < self.l0:
                return 0.0
            return self.lambda_max * (layer_idx - self.l0) / max(1, self.num_layers - self.l0 - 1)
        
        elif self.lambda_schedule == "linear_decrease":
            if layer_idx < self.l0:
                return self.lambda_max
            return self.lambda_max * (1 - (layer_idx - self.l0) / max(1, self.num_layers - self.l0 - 1))
        
        else:
            raise ValueError(
                f"Unknown lambda_schedule: {self.lambda_schedule}. "
                f"Must be one of: 'linear_increase', 'linear_decrease', 'constant'"
            )
    
    def forward(
        self,
        x: Float[torch.Tensor, "*batch time_len dim"],
        attn_mask: Optional[Bool[torch.Tensor, "*batch time_len time_len"]] = None,
        var_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
        time_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
        regime_prior: Optional[Float[torch.Tensor, "*batch time_len expert"]] = None,
    ) -> tuple[Float[torch.Tensor, "*batch time_len dim"], float, float, float]:
        """
        Forward pass with prior-guided routing.

        Args:
            x: Input tensor
            attn_mask: Attention mask (causal)
            var_id: Variate IDs
            time_id: Time IDs
            regime_prior: Regime-based expert preferences (from regime predictor)

        Returns:
            Tuple of (output, total_load_balance_loss, total_kl_loss, total_ortho_loss)
        """
        total_lb_loss = 0.0
        total_kl_loss = 0.0
        total_ortho_loss = 0.0

        if self.use_moe:
            for idx, layer in enumerate(self.layers):
                # Compute depth-aware lambda
                lambda_anchor = self.compute_lambda(idx)

                x, lb_loss, kl_loss, ortho_loss = layer(
                    x,
                    attn_mask,
                    var_id=var_id,
                    time_id=time_id,
                    regime_prior=regime_prior,
                    lambda_anchor=lambda_anchor,
                )
                total_lb_loss += lb_loss
                total_kl_loss += kl_loss
                total_ortho_loss += ortho_loss
        else:
            for layer in self.layers:
                x, _, _, _ = layer(x, attn_mask, var_id=var_id, time_id=time_id)

        return self.norm(x), total_lb_loss, total_kl_loss, total_ortho_loss
    
    def get_routing_stats(self) -> Optional[dict]:
        """
        Get routing statistics from all MoE layers.
        
        Returns:
            Dictionary with per-layer routing stats or None if not available
        """
        if not self.use_moe:
            return None
        
        expert_usage_per_layer = []
        shared_mass_gate_per_layer = []
        entropy_per_layer = []
        lambda_per_layer = []
        alpha = None
        beta = None
        
        for idx, layer in enumerate(self.layers):
            # Get FFN from layer
            ffn = layer.ffn
            
            if hasattr(ffn, 'get_routing_stats'):
                stats = ffn.get_routing_stats()
                
                if stats['expert_usage'] is not None:
                    expert_usage_per_layer.append(stats['expert_usage'])
                else:
                    # Use a placeholder until the layer records stats.
                    expert_usage_per_layer.append([1.0 / ffn.num_experts] * ffn.num_experts)
                
                shared_mass_gate_per_layer.append(stats['pi_sh'] if stats['pi_sh'] is not None else 0.5)
                entropy_per_layer.append(stats['entropy'] if stats['entropy'] is not None else 1.0)
                lambda_per_layer.append(self.compute_lambda(idx))
                
                # Read alpha and beta from the first layer with lambda > 0,
                # because a layer with lambda 0 does not train its gate.
                if alpha is None and self.compute_lambda(idx) > 0:
                    alpha = stats['alpha']
                    beta = stats['beta']
            else:
                # Placeholder values for a dense FFN
                expert_usage_per_layer.append([1.0 / 32] * 32)
                shared_mass_gate_per_layer.append(0.5)
                entropy_per_layer.append(1.0)
                lambda_per_layer.append(self.compute_lambda(idx))
        
        return {
            'regime_probs': None,  # The encoder does not track regime_probs.
            'expert_usage_per_layer': expert_usage_per_layer,
            'shared_mass_gate_per_layer': shared_mass_gate_per_layer,
            'entropy_per_layer': entropy_per_layer,
            'lambda_per_layer': lambda_per_layer,
            'alpha': alpha if alpha is not None else 1.0,
            'beta': beta if beta is not None else 0.0,
        }

