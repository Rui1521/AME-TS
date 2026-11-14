"""
Anchored MoE feed-forward layer.

This module extends the standard MoE FFN with regime-aware priors. A two-part
prior guides expert selection and treats shared and specialized experts
separately.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import nn

from .ffn import GatedLinearUnitFeedForward


class MoEAnchoredFeedForward(nn.Module):
    """
    Mixture-of-Experts with structure-aware routing.
    
    routing_mode="additive" mixes the prior into the gate logits:
        gate_logits = (1 + lambda) * router_logits + lambda * log(regime_prior)
    
    routing_mode="kl_divergence" keeps gate_logits = router_logits and adds
    lambda * D(prior, router) to the auxiliary loss, where kl_divergence_type
    selects D (forward KL, reverse KL, or Jensen-Shannon).
    
    Where:
        - router_logits: Learned routing scores from linear layer (base routing)
        - regime_prior: Expert preferences from regime predictor (structural prior)
        - lambda: Interpolation weight (depth-aware)
    
    The router is a standard learned linear layer.
    
    Args:
        num_experts: Number of expert networks
        num_experts_per_token: Top-k experts to route to (default: 2)
        in_dim: Input dimension
        hidden_dim: Hidden dimension for experts
        out_dim: Output dimension
        activation: Activation function
        bias: Whether to use bias in linear layers
        ffn_dropout_p: Dropout probability
    """
    
    def __init__(
        self,
        num_experts: int,
        num_experts_per_token: int,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        activation=F.silu,
        bias: bool = True,
        ffn_dropout_p: float = 0.0,
        shared_gate_mode: str = "trainable",
        shared_gate_alpha: float = 1.0,
        shared_gate_beta: float = 0.5,
        num_regimes: int = 4,
        routing_mode: str = "additive",
        kl_loss_coef: float = 0.1,
        kl_divergence_type: str = "js",
        ortho_loss_coef: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.num_regimes = num_regimes
        self.shared_gate_mode = shared_gate_mode
        self.routing_mode = routing_mode
        self.kl_loss_coef = kl_loss_coef
        self.kl_divergence_type = kl_divergence_type
        self.ortho_loss_coef = ortho_loss_coef

        # Learned router (standard MoE routing)
        self.router = nn.Linear(in_dim, num_experts, bias=False)

        # Shared-mass gate parameters (α and β)
        if shared_gate_mode == "trainable":
            # Make α and β learnable parameters
            self.alpha = nn.Parameter(torch.tensor(shared_gate_alpha))
            self.beta = nn.Parameter(torch.tensor(shared_gate_beta))
        else:  # "fixed"
            # Register as buffers (not trainable)
            self.register_buffer('alpha', torch.tensor(shared_gate_alpha))
            self.register_buffer('beta', torch.tensor(shared_gate_beta))

        # Create expert networks (same as standard MoE)
        self.experts = nn.ModuleList(
            [
                GatedLinearUnitFeedForward(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    out_dim=out_dim,
                    activation=activation,
                    bias=bias,
                    ffn_dropout_p=ffn_dropout_p,
                )
                for _ in range(num_experts)
            ]
        )

        # Build regime groups for the orthogonality loss. The groups exist only
        # when ortho_loss_coef > 0 and num_experts >= 8.
        self.regime_groups = {}  # {regime_idx: [expert_indices]}
        self._experts_needing_cache = set()
        if ortho_loss_coef > 0 and num_experts >= 8:
            self._build_regime_groups()

        # Routing stats for logging
        self._last_expert_usage = None
        self._last_pi_sh = None
        self._last_entropy = None
    
    def _build_regime_groups(self):
        """
        Build the mapping from regime index to expert indices.

        The allocation matches _build_two_part_prior and _compute_specialized_prior.
        The method stores only groups with two or more experts, because a
        single-expert group needs no orthogonality loss.
        """
        R = self.num_regimes
        # Same allocation as _build_two_part_prior
        if self.num_experts <= R:
            return  # No multi-expert groups possible

        if self.num_experts % R == 0:
            if self.num_experts == R:
                return
            n_specialized = self.num_experts - R
        else:
            n_specialized = (self.num_experts // R) * R

        experts_per_regime = n_specialized // R
        remainder = n_specialized % R

        expert_idx = 0
        for regime_idx in range(R):
            count = experts_per_regime + (1 if regime_idx < remainder else 0)
            if count >= 2:
                group = list(range(expert_idx, expert_idx + count))
                self.regime_groups[regime_idx] = group
                self._experts_needing_cache.update(group)
            expert_idx += count

    def _compute_ortho_loss(self, selected_experts, expert_outputs_cache):
        """
        Compute the orthogonality loss from inner products between co-selected
        experts in the same regime group.

        For each pair of experts (e_i, e_j) in the same regime, the method finds
        the tokens where both are in the top-k and penalizes the mean of
        |<output_i, output_j>|.

        Args:
            selected_experts: [N, num_experts_per_token] selected expert indices
            expert_outputs_cache: {expert_idx: (batch_indices, expert_outputs)}

        Returns:
            Scalar orthogonality loss
        """
        loss = torch.tensor(0.0, device=selected_experts.device)
        n_pairs = 0

        for regime_idx, expert_list in self.regime_groups.items():
            for i in range(len(expert_list)):
                for j in range(i + 1, len(expert_list)):
                    ei, ej = expert_list[i], expert_list[j]
                    if ei not in expert_outputs_cache or ej not in expert_outputs_cache:
                        continue

                    idx_i, out_i = expert_outputs_cache[ei]
                    idx_j, out_j = expert_outputs_cache[ej]

                    # Find the tokens where both experts are selected.
                    set_i = set(idx_i.cpu().tolist())
                    set_j = set(idx_j.cpu().tolist())
                    common = sorted(set_i & set_j)

                    if len(common) == 0:
                        continue

                    # Map common token indices back to positions in the cached
                    # outputs. idx_i holds the token index of each output of expert i.
                    common_t = torch.tensor(common, device=selected_experts.device)

                    # Find the positions in out_i and out_j of the common tokens.
                    mask_i = (idx_i.unsqueeze(1) == common_t.unsqueeze(0)).any(dim=1)
                    mask_j = (idx_j.unsqueeze(1) == common_t.unsqueeze(0)).any(dim=1)

                    oi = out_i[mask_i]  # [M, dim]
                    oj = out_j[mask_j]  # [M, dim]

                    # Penalize the inner product |<o_i, o_j>|.
                    inner = (oi * oj).sum(dim=-1)  # [M]
                    loss = loss + inner.abs().mean()
                    n_pairs += 1

        if n_pairs > 0:
            loss = loss / n_pairs
        return loss

    def _compute_shared_mass_gate(
        self, 
        regime_probs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute π_sh, the fraction of prior mass for shared experts.
        
        The gate has two stages:
        1. Strength gate (1 - S_max), a hard cutoff for strong signals.
        2. Entropy gate sigmoid(α·H - β), a soft threshold for uncertainty.
        
        Formula: π_sh = (1 - S_max) · sigmoid(α·H - β)
        
        Behavior:
        - If any feature is strong, with S near 1, π_sh is near 0 and the
          specialized experts get the mass.
        - If all features are weak and uncertain, π_sh is high and the shared
          experts get the mass.
        
        Args:
            regime_probs: [batch_size, num_regimes] independent regime feature scores
            
        Returns:
            pi_sh: [batch_size] shared mass gate values in [0, 1]
            H: [batch_size] total entropy across regimes
        """
        # Clamp to avoid log(0).
        p = regime_probs.clamp(1e-6, 1 - 1e-6)
        
        # Compute per-feature Bernoulli entropy
        # H_i = -p * log(p) - (1-p) * log(1-p)
        H_i = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
        H_i = H_i / np.log(2)  # Normalize each feature to [0, 1]
        
        # Sum entropy across all 4 independent features
        H = H_i.sum(dim=-1)  # [batch_size]
        
        # S is the strength of the strongest feature, so one strong feature
        # makes S high.
        S = regime_probs.max(dim=-1)[0]  # [batch_size]
        
        # Stage 1: strength gate, a hard cutoff for strong signals.
        strength_gate = 1 - S  # S=1 gives gate=0, S=0 gives gate=1
        
        # Stage 2: entropy gate, a soft threshold for uncertainty. Clamping
        # alpha*H - beta prevents overflow in the sigmoid.
        entropy_gate = torch.sigmoid(torch.clamp(self.alpha * H - self.beta, min=-10.0, max=10.0))
        
        # Combined: π_sh = (1 - S_max) · sigmoid(α·H - β)
        pi_sh = strength_gate * entropy_gate
        
        return pi_sh, H
    
    def _compute_specialized_prior(
        self,
        regime_probs: torch.Tensor,
        n_specialized: int
    ) -> torch.Tensor:
        """
        Compute the prior over specialized experts.
        
        The method spreads experts evenly across regimes and gives the remainder
        to the first regimes.
        
        Args:
            regime_probs: [batch_size, num_regimes] regime probabilities
            n_specialized: Number of specialized experts
            
        Returns:
            q_sp: [batch_size, n_specialized] normalized prior
        """
        batch_size, num_regimes = regime_probs.shape
        q_sp = torch.zeros(
            batch_size, n_specialized,
            device=regime_probs.device
        )
        
        if n_specialized <= num_regimes:
            # Simple 1:1 mapping
            q_sp[:, :n_specialized] = regime_probs[:, :n_specialized]
        else:
            # Several experts per regime. Spread them evenly and give the
            # remainder to the first regimes.
            experts_per_regime = n_specialized // num_regimes
            remainder = n_specialized % num_regimes
            
            expert_idx = 0
            for regime_idx in range(num_regimes):
                # Base allocation
                count = experts_per_regime
                # Add one extra expert to first 'remainder' regimes
                if regime_idx < remainder:
                    count += 1
                
                # Assign this regime's probability to all its experts
                q_sp[:, expert_idx:expert_idx+count] = regime_probs[:, regime_idx:regime_idx+1].expand(
                    -1, count
                )
                expert_idx += count
        
        # Normalize to sum to 1
        q_sp = q_sp / (q_sp.sum(dim=-1, keepdim=True) + 1e-6)
        
        return q_sp
    
    def _build_two_part_prior(
        self,
        regime_probs: torch.Tensor,
        num_experts: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build the combined prior q(e|X) with separate shared and specialized parts.
        
        Steps:
        - Compute shared-mass gate π_sh from regime uncertainty and strength
        - Build specialized prior q_sp over regime-specific experts
        - Distribute shared mass uniformly over shared experts
        - Combine: q[shared] = π_sh/n_shared, q[specialized] = (1-π_sh)*q_sp
        
        Args:
            regime_probs: [batch_size, num_regimes] regime probabilities
            num_experts: Total number of experts
            
        Returns:
            q: [batch_size, num_experts] combined prior distribution
            pi_sh: [batch_size] shared mass gate values
            H: [batch_size] regime entropy
        """
        batch_size, num_regimes = regime_probs.shape
        
        # Check that regime_probs has the expected num_regimes.
        if num_regimes != self.num_regimes:
            raise ValueError(
                f"Regime predictor output has {num_regimes} features, "
                f"but router expects {self.num_regimes}. "
                f"Please check regime predictor configuration."
            )
        
        # Allocate experts. The specialized count is the largest multiple of R
        # that leaves at least one shared expert when num_experts > R.
        R = self.num_regimes
        
        if num_experts <= R:
            # R or fewer experts: all specialized, none shared
            n_specialized = num_experts
            n_shared = 0
        elif num_experts % R == 0:
            # num_experts is a multiple of R
            if num_experts == R:
                # Exactly R experts: all specialized
                n_specialized = R
                n_shared = 0
            else:
                # Reserve R experts as shared and specialize the rest.
                # For example, 12 experts give 8 specialized and 4 shared.
                n_specialized = num_experts - R
                n_shared = R
        else:
            # Not a multiple of R: specialize the largest multiple of R.
            # For example, 13 experts give 12 specialized and 1 shared.
            n_specialized = (num_experts // R) * R
            n_shared = num_experts % R
        
        # Compute shared-mass gate π_sh and entropy H
        pi_sh, H = self._compute_shared_mass_gate(regime_probs)  # [batch_size], [batch_size]
        
        # Build specialized prior
        q_sp = self._compute_specialized_prior(regime_probs, n_specialized)
        
        # Combine into full prior
        q = torch.zeros(batch_size, num_experts, device=regime_probs.device)
        
        # Shared experts split the π_sh mass uniformly.
        if n_shared > 0:
            q[:, n_specialized:] = pi_sh.unsqueeze(-1) / n_shared
        
        # Specialized experts get (1 - π_sh) * q_sp.
        q[:, :n_specialized] = (1 - pi_sh).unsqueeze(-1) * q_sp
        return q, pi_sh, H

    def _compute_kl_divergence(
        self,
        router_logits: torch.Tensor,
        anchor_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the divergence between the anchor (regime prior) and router distributions.

        The method supports three types:
        - forward: KL(anchor||router), which penalizes the router for missing the anchor's modes.
        - reverse: KL(router||anchor), which penalizes the router for extra modes.
        - js: Jensen-Shannon divergence, a symmetric penalty.

        Args:
            router_logits: [N, num_experts] learned routing logits
            anchor_logits: [N, num_experts] regime-based prior logits

        Returns:
            Scalar KL divergence loss (mean over batch)
        """
        # Convert logits to probabilities in fp32 to avoid fp16 underflow.
        p_router = F.softmax(router_logits.float(), dim=-1).clamp(min=1e-8)
        p_anchor = F.softmax(anchor_logits.float(), dim=-1).clamp(min=1e-8)

        if self.kl_divergence_type == 'forward':
            # KL(anchor||router): anchor is P, router is Q
            kl = (p_anchor * (p_anchor.log() - p_router.log())).sum(-1)
        elif self.kl_divergence_type == 'reverse':
            # KL(router||anchor): router is P, anchor is Q
            kl = (p_router * (p_router.log() - p_anchor.log())).sum(-1)
        elif self.kl_divergence_type == 'js':
            # Jensen-Shannon divergence: symmetric
            m = (0.5 * (p_anchor + p_router)).clamp(min=1e-8)
            kl_am = (p_anchor * (p_anchor.log() - m.log())).sum(-1)
            kl_rm = (p_router * (p_router.log() - m.log())).sum(-1)
            kl = 0.5 * (kl_am + kl_rm)
        else:
            raise ValueError(f"Unknown kl_divergence_type: {self.kl_divergence_type}")

        return kl.mean()

    def _compute_load_balance_loss(
        self,
        gate_logits: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the load balance loss.

        The loss pushes the router toward equal expert use, which prevents expert
        collapse.

        Formula: lb_loss = num_experts * sum_i(f_i * p_i)
        where:
          f_i = fraction of tokens routed to expert i
          p_i = mean routing probability for expert i across all tokens

        Perfect balance gives lb_loss = 1.0. Imbalance gives a larger value.

        Args:
            gate_logits: [N, num_experts] routing logits (before top-k)
            selected_experts: [N, num_experts_per_token] selected expert indices

        Returns:
            Scalar load balance loss
        """
        num_tokens = gate_logits.shape[0]

        # f_i is the fraction of tokens dispatched to each expert. Sum the
        # one-hot encodings of the selected experts across the top-k slots.
        expert_mask = F.one_hot(selected_experts, self.num_experts).float()  # [N, k, E]
        expert_mask = expert_mask.sum(dim=1)  # [N, E], selection count per token
        f = expert_mask.mean(dim=0)  # [E], fraction of tokens per expert

        # p_i is the mean routing probability of each expert across all tokens.
        routing_probs = F.softmax(gate_logits, dim=-1)  # [N, E]
        p = routing_probs.mean(dim=0)  # [E]

        # Load balance loss
        lb_loss = self.num_experts * (f * p).sum()

        return lb_loss

    def forward(
        self,
        x: Float[torch.Tensor, "... in_dim"],
        regime_prior: Optional[Float[torch.Tensor, "... expert"]] = None,
        lambda_anchor: float = 0.0,
    ) -> tuple[Float[torch.Tensor, "... dim"], float, float, float]:
        """
        Forward pass with prior-guided routing.

        Args:
            x: Input tensor
            regime_prior: Regime-based expert preferences (from regime predictor)
            lambda_anchor: Prior interpolation weight (0 = pure router, higher = more prior influence)

        Returns:
            Tuple of (output, load_balance_loss, kl_loss, ortho_loss)
        """
        # Flatten the input to 2D: [batch*seq, dim]
        x_squashed = x.view(-1, x.shape[-1])

        # 1. Compute base routing logits using learned router
        router_logits = self.router(x_squashed)  # [N, num_experts]
        gate_logits = router_logits.clone()  # Additive mode overwrites this copy.

        # 2. Combine router logits with two-part prior
        pi_sh_for_logging = None
        regime_entropy_for_logging = None
        kl_loss = 0.0

        if regime_prior is not None and lambda_anchor > 0:
            # Flatten regime_prior to [N, num_regimes].
            regime_prior_flat = regime_prior.view(-1, regime_prior.shape[-1])

            # Build the two-part prior with the shared-mass gate. Also return pi_sh and entropy.
            anchor_probs, pi_sh, regime_entropy = self._build_two_part_prior(
                regime_prior_flat, self.num_experts
            )

            # Save for logging
            pi_sh_for_logging = pi_sh
            regime_entropy_for_logging = regime_entropy

            # Convert to logits
            anchor_logits = torch.log(anchor_probs + 1e-6)

            # Apply routing mode
            if self.routing_mode == 'additive':
                # Additive routing: gate_logits = (1 + lambda) * router_logits + lambda * log(prior)
                gate_logits = (1 + lambda_anchor) * router_logits + lambda_anchor * anchor_logits
            elif self.routing_mode == 'kl_divergence':
                # The divergence enters as a regularization loss.
                kl_loss = self._compute_kl_divergence(router_logits, anchor_logits)
                # Scale the loss by this layer's lambda.
                kl_loss = lambda_anchor * kl_loss
                # gate_logits stay unchanged, so routing is purely learned.
            else:
                raise ValueError(f"Unknown routing_mode: {self.routing_mode}")

        # 3. Select top-k experts with noisy gating. Training adds noise to
        # break expert symmetry.
        if self.training:
            noise = torch.randn_like(gate_logits) * (1.0 / self.num_experts)
            gate_logits_for_topk = gate_logits + noise
        else:
            gate_logits_for_topk = gate_logits

        weights, selected_experts = torch.topk(
            gate_logits_for_topk, self.num_experts_per_token, dim=-1
        )
        # Normalize the top-k weights with a softmax.
        weights = nn.functional.softmax(weights, dim=-1, dtype=torch.float).type_as(x)

        # 4. Compute the load balance loss, which prevents expert collapse.
        lb_loss = self._compute_load_balance_loss(gate_logits, selected_experts)

        # Track stats for logging
        with torch.no_grad():
            # Count how often each expert is selected.
            expert_counts = torch.zeros(self.num_experts, device=x.device)
            for i in range(self.num_experts):
                expert_counts[i] = (selected_experts == i).sum().float()
            total_selections = selected_experts.numel()
            self._last_expert_usage = (expert_counts / total_selections).cpu().tolist()
            self._last_selected_experts = selected_experts.detach().cpu()

            # Record regime entropy and pi_sh from _build_two_part_prior.
            if pi_sh_for_logging is not None and regime_entropy_for_logging is not None:
                self._last_pi_sh = pi_sh_for_logging.mean().item()
                self._last_entropy = regime_entropy_for_logging.mean().item()
            else:
                # No prior is active.
                self._last_pi_sh = None
                self._last_entropy = None

        # 5. Route tokens to the selected experts as in a standard MoE. Cache
        # expert outputs when the orthogonality loss needs them.
        need_ortho = (self.training and self.ortho_loss_coef > 0
                      and len(self.regime_groups) > 0)
        expert_outputs_cache = {} if need_ortho else None

        results = torch.zeros_like(x_squashed)
        for i, expert in enumerate(self.experts):
            mask = (selected_experts == i)
            batch_idx, nth_expert = torch.where(mask)

            if len(batch_idx) > 0:
                expert_out = expert(x_squashed[batch_idx])
                results[batch_idx] += weights[batch_idx, nth_expert, None] * expert_out

                # Cache outputs only for experts in multi-expert regime groups.
                if need_ortho and i in self._experts_needing_cache:
                    expert_outputs_cache[i] = (batch_idx, expert_out)

        # 6. Compute the orthogonality loss.
        ortho_loss = 0.0
        if need_ortho:
            ortho_loss = self._compute_ortho_loss(selected_experts, expert_outputs_cache)

        results = results.view_as(x)

        return results, lb_loss, kl_loss, ortho_loss
    
    def get_routing_stats(self) -> dict:
        """Return the latest routing statistics for logging."""
        return {
            'expert_usage': self._last_expert_usage,
            'pi_sh': self._last_pi_sh,
            'entropy': self._last_entropy,
            'alpha': self.alpha.item() if hasattr(self.alpha, 'item') else self.alpha,
            'beta': self.beta.item() if hasattr(self.beta, 'item') else self.beta,
        }

