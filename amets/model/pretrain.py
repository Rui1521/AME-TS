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
AMETS pretraining module.

Lightning module for training AMETS, a time-series foundation model with
structure-aware expert routing. It predicts points and trains with a
configurable L1 or L2 loss. The regime predictor runs inside forward() on the
GPU, not in a data transform.
"""

from collections import defaultdict
from collections.abc import Callable
from typing import Any, Optional

import lightning as L
import numpy as np
import torch
from torch import nn

from amets.module.ts_embed import (
    FeatLinear,
    MultiInSizeLinear,
    MultiOutSizeLinear,
)
from amets.module.position.additive import LearnedEmbedding
from amets.module.position.attn_bias import BinaryAttentionBias
from amets.module.position.attn_projection import LearnedProjection
from amets.module.norm import RMSNorm
from amets.optim import SchedulerType, get_scheduler
from amets.transform import (
    AddObservedMask,
    AddTimeIndex,
    AddVariateIndex,
    DefaultPatchSizeConstraints,
    DummyValueImputation,
    FlatPackCollection,
    FlatPackFields,
    GetPatchSize,
    ImputeTimeSeries,
    MaskedPrediction,
    PackFields,
    PatchCrop,
    Patchify,
    SampleDimension,
    SaveContextTarget,
    SelectFields,
    SequencifyField,
    Transformation,
)

from .combined_regime_predictor import context_regime_probs
from .module import AMETSModule


class AMETSPretrain(L.LightningModule):
    """
    Lightning module for pretraining AMETS models.
    
    The module predicts points and trains with an L1 or L2 loss. forward()
    computes regime probabilities on the GPU from the raw context.
    """
    
    # context_target carries the raw context to forward(), which computes
    # regime_probs from it on the GPU. regime_probs is not in seq_fields.
    seq_fields: tuple[str, ...] = (
        "target",
        "observed_mask",
        "time_id",
        "variate_id",
        "prediction_mask",
        "patch_size",
        "context_target",  # Raw context that forward() uses for regime prediction
    )
    
    pad_func_map: dict[str, Callable[[tuple[int, ...], np.dtype], np.ndarray]] = {
        "target": np.zeros,
        "observed_mask": np.zeros,
        "time_id": np.zeros,
        "variate_id": np.zeros,
        "prediction_mask": np.zeros,
        "patch_size": np.zeros,
        "context_target": np.zeros,  # The collate function handles this field separately
    }
    
    def __init__(
        self,
        min_patches: int,
        min_mask_ratio: float,
        max_mask_ratio: float,
        max_dim: int,
        num_training_steps: int,
        num_warmup_steps: int,
        module_kwargs: Optional[dict[str, Any]] = None,
        module: Optional[AMETSModule] = None,
        beta1: float = 0.9,
        beta2: float = 0.98,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        log_on_step: bool = False,
        # Routing parameters. __init__ passes these to the module it builds from module_kwargs.
        regime_predictor_path: Optional[str] = None,
        freeze_regime_predictor: bool = True,
        lambda_max: float = 1.0,
        l0: int = 0,
        lambda_schedule: str = "linear_increase",
        shared_expert_ratio: float = 0.25,
        num_regimes: int = 4,
        pre_norm: bool = True,
        time_recency_bias: bool = False,
        # Loss configuration
        loss_type: str = "l1",  # "l1" or "l2"
        recon_loss_coef: float = 0.0,  # Coefficient for context reconstruction loss
        # Routing mode configuration
        routing_mode: str = "additive",
        kl_loss_coef: float = 0.1,
        lb_loss_coef: float = 0.01,
        kl_divergence_type: str = "js",
        # Shared gate parameters
        shared_gate_mode: str = "trainable",
        shared_gate_alpha: float = 1.0,
        shared_gate_beta: float = 0.5,
        # Expert diversity
        ortho_loss_coef: float = 0.0,
        # Scheduler configuration
        scheduler_type: str = "cosine_with_restarts",
        num_cycles: int = 1,
        regime_batched: bool = True,
    ):
        """
        Initialize AMETS pretrain module.

        Args:
            min_patches: Minimum number of patches
            min_mask_ratio: Minimum mask ratio for masked prediction
            max_mask_ratio: Maximum mask ratio for masked prediction
            max_dim: Maximum dimension for sampling
            num_training_steps: Total number of training steps
            num_warmup_steps: Number of warmup steps
            module_kwargs: Keyword arguments for AMETSModule
            module: Pre-initialized AMETSModule (optional)
            beta1: Adam beta1
            beta2: Adam beta2
            lr: Learning rate
            weight_decay: Weight decay
            log_on_step: Whether to log on step
            regime_predictor_path: Path to pre-trained regime predictor
            freeze_regime_predictor: Whether to freeze regime predictor
            lambda_max: Maximum lambda for prior injection
            l0: Layer index to start prior injection
            lambda_schedule: Lambda schedule type
            shared_expert_ratio: Sizes only the placeholder routing stats.
                MoEAnchoredFeedForward sets the shared and specialized split.
            num_regimes: Number of regimes
            pre_norm: Whether transformer blocks are pre-norm (True) or post-norm (False)
            time_recency_bias: Add RecencyAttentionBias on time ids alongside RoPE
            loss_type: Loss type, "l1" or "l2"
            routing_mode: Routing mode, "additive" or "kl_divergence"
            kl_loss_coef: Coefficient for KL divergence loss
            kl_divergence_type: Type of divergence, "forward", "reverse", or "js"
            scheduler_type: LR scheduler type, one of "linear", "cosine",
                "cosine_with_restarts", "constant_with_warmup", "polynomial", or
                "inverse_sqrt"
            regime_batched: Run the regime predictor as one masked batched forward
                if True, or as one forward per (sample, variate) if False. The
                outputs match within float rounding.
        """
        super().__init__()

        assert (module is not None) or (
            module_kwargs is not None
        ), "if module is not provided, module_kwargs is required"
        assert (
            num_warmup_steps <= num_training_steps
        ), f"num_warmup_steps ({num_warmup_steps}) should be <= num_training_steps ({num_training_steps})."

        # Store hyperparameters
        self.save_hyperparameters(ignore=["module"])

        # Store training parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.log_on_step = log_on_step
        self.loss_type = loss_type
        self.routing_mode = routing_mode
        self.kl_loss_coef = kl_loss_coef
        self.lb_loss_coef = lb_loss_coef
        self.recon_loss_coef = recon_loss_coef
        self.ortho_loss_coef = ortho_loss_coef
        self.scheduler_type = scheduler_type
        self.num_cycles = num_cycles
        self.regime_batched = regime_batched

        # Create or use provided module
        if module is not None:
            self.module = module
        else:
            # Add routing parameters to module_kwargs
            module_kwargs_resolved = module_kwargs.copy()
            module_kwargs_resolved.update({
                'regime_predictor_path': regime_predictor_path,
                'freeze_regime_predictor': freeze_regime_predictor,
                'lambda_max': lambda_max,
                'l0': l0,
                'lambda_schedule': lambda_schedule,
                'shared_expert_ratio': shared_expert_ratio,
                'num_regimes': num_regimes,
                'pre_norm': pre_norm,
                'time_recency_bias': time_recency_bias,
                'loss_type': loss_type,
                'routing_mode': routing_mode,
                'kl_loss_coef': kl_loss_coef,
                'kl_divergence_type': kl_divergence_type,
                'shared_gate_mode': shared_gate_mode,
                'shared_gate_alpha': shared_gate_alpha,
                'shared_gate_beta': shared_gate_beta,
                'ortho_loss_coef': ortho_loss_coef,
            })
            self.module = AMETSModule(**module_kwargs_resolved)
    
    def forward(
        self,
        target: torch.Tensor,
        observed_mask: torch.Tensor,
        sample_id: torch.Tensor,
        time_id: torch.Tensor,
        variate_id: torch.Tensor,
        prediction_mask: torch.Tensor,
        patch_size: torch.Tensor,
        context_target: Optional[torch.Tensor] = None,
        context_target_list: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Forward pass through AMETSModule.
        
        Args:
            target: Input target tensor
            observed_mask: Mask for observed values
            sample_id: Sample IDs for packing
            time_id: Time indices
            variate_id: Variate indices
            prediction_mask: Mask for prediction positions
            patch_size: Patch sizes
            context_target: Unused. forward() reads the contexts from context_target_list.
            context_target_list: Per-sample raw contexts for regime prediction on the GPU
        
        Returns:
            Tuple of point predictions, load balance loss, KL loss, and
            orthogonality loss.
        """
        # Compute regime_probs from context_target_list on GPU (per-variate)
        regime_probs = None
        if context_target_list is not None and self.module.regime_predictor is not None:
            # Additive mode needs regime_probs at every step. KL mode needs
            # them only during training.
            if self.routing_mode == 'additive' or self.training:
                regime_probs = self._compute_regime_probs(context_target_list, sample_id, variate_id)

        pred, lb_loss, kl_loss, ortho_loss = self.module(
            target=target,
            observed_mask=observed_mask,
            sample_id=sample_id,
            time_id=time_id,
            variate_id=variate_id,
            prediction_mask=prediction_mask,
            patch_size=patch_size,
            regime_probs=regime_probs,
        )
        return pred, lb_loss, kl_loss, ortho_loss
    

    def _compute_regime_probs(
        self,
        context_target_list: list,
        sample_id: torch.Tensor,
        variate_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return token-level regime probabilities of shape (batch, seq_len,
        num_regimes). ``context_regime_probs`` computes them from the
        per-(sample, variate) contexts. The regime predictor runs in eval mode.
        """
        predictor = self.module.regime_predictor
        was_training = predictor.training
        predictor.eval()
        try:
            return context_regime_probs(
                predictor, context_target_list, sample_id, variate_id,
                self.module.num_regimes, batched=self.regime_batched,
            )
        finally:
            if was_training:
                predictor.train()

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step with point prediction loss.
        """
        # Extract context_target_list from batch
        context_target_list = batch.pop("context_target_list", None)
        
        # Handle NaN values in input tensors (some datasets have missing values)
        if "target" in batch and torch.isnan(batch["target"]).any():
            batch["target"] = torch.nan_to_num(batch["target"], nan=0.0)
        
        # Forward pass
        forward_kwargs = {field: batch[field] for field in list(self.seq_fields) + ["sample_id"] if field in batch and field != "context_target"}
        forward_kwargs["context_target_list"] = context_target_list

        pred, lb_loss, kl_loss, ortho_loss = self(**forward_kwargs)

        if torch.isnan(pred).any():
            print(f"[WARNING] Predictions contain NaN at batch {batch_idx}, masking them out")
            nan_mask = torch.isnan(pred)

            # Clone masks to avoid inplace modification of tensors in the computation graph
            observed_mask = batch["observed_mask"].clone()
            prediction_mask = batch["prediction_mask"].clone()

            # Set observed_mask to False where predictions are NaN
            # This excludes NaN positions from loss calculation
            observed_mask[nan_mask] = False

            nan_mask_2d = nan_mask.any(dim=-1)  # Collapse feature dim
            prediction_mask[nan_mask_2d] = False

            batch["observed_mask"] = observed_mask
            batch["prediction_mask"] = prediction_mask

            # Replace NaN with 0. The masks exclude these positions from the loss.
            pred = torch.nan_to_num(pred, nan=0.0)

        # Compute task loss using module's compute_loss method
        task_loss = self.module.compute_loss(
            pred=pred,
            target=batch["target"],
            prediction_mask=batch["prediction_mask"],
            observed_mask=batch["observed_mask"],
            patch_size=batch.get("patch_size", None),
        )

        # Compute reconstruction loss on context positions
        recon_loss = torch.tensor(0.0, device=task_loss.device)
        if self.recon_loss_coef > 0:
            recon_loss = self.module.compute_recon_loss(
                pred=pred,
                target=batch["target"],
                prediction_mask=batch["prediction_mask"],
                observed_mask=batch["observed_mask"],
            )

        # Add auxiliary losses
        total_loss = (task_loss + self.recon_loss_coef * recon_loss
                      + self.lb_loss_coef * lb_loss + self.kl_loss_coef * kl_loss
                      + self.ortho_loss_coef * ortho_loss)

        # Store per-batch losses for callback access
        self._last_task_loss = task_loss.detach()
        self._last_kl_loss = kl_loss if isinstance(kl_loss, (int, float)) else kl_loss.detach()
        self._last_lb_loss = lb_loss if isinstance(lb_loss, (int, float)) else lb_loss.detach()
        self._last_total_loss = total_loss.detach()

        # Log losses (EarlyStopping with check_finite=True will handle NaN/Inf)
        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        loss_name = "L1Loss" if self.loss_type == "l1" else "L2Loss"
        self.log(
            f"train/{loss_name}",
            task_loss,
            on_step=self.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self.log(
            "train/load_balance_loss",
            lb_loss,
            on_step=self.log_on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self.log(
            "train/kl_loss",
            kl_loss,
            on_step=self.log_on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self.log(
            "train/ortho_loss",
            ortho_loss,
            on_step=self.log_on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self.log(
            "train/recon_loss",
            recon_loss,
            on_step=self.log_on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self.log(
            "train/total_loss",
            total_loss,
            on_step=self.log_on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )

        return total_loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """
        Validation step with point prediction loss.
        """
        # Extract context_target_list from batch
        context_target_list = batch.pop("context_target_list", None)
        
        # Handle NaN values in input tensors (some datasets have missing values)
        if "target" in batch and torch.isnan(batch["target"]).any():
            batch["target"] = torch.nan_to_num(batch["target"], nan=0.0)
        
        # Forward pass
        forward_kwargs = {field: batch[field] for field in list(self.seq_fields) + ["sample_id"] if field in batch and field != "context_target"}
        forward_kwargs["context_target_list"] = context_target_list

        pred, lb_loss, kl_loss, ortho_loss = self(**forward_kwargs)

        if torch.isnan(pred).any():
            print(f"[WARNING] Predictions contain NaN at batch {batch_idx}, masking them out")

            nan_mask = torch.isnan(pred)

            # Set observed_mask to False where predictions are NaN
            # This excludes NaN positions from loss calculation
            batch["observed_mask"][nan_mask] = False

            nan_mask_2d = nan_mask.any(dim=-1)  # Collapse feature dim
            batch["prediction_mask"][nan_mask_2d] = False

            # Replace NaN with 0. The masks exclude these positions from the loss.
            pred = torch.nan_to_num(pred, nan=0.0)

        # Compute task loss
        task_loss = self.module.compute_loss(
            pred=pred,
            target=batch["target"],
            prediction_mask=batch["prediction_mask"],
            observed_mask=batch["observed_mask"],
            patch_size=batch.get("patch_size", None),
        )

        # Compute reconstruction loss on context positions
        recon_loss = torch.tensor(0.0, device=task_loss.device)
        if self.recon_loss_coef > 0:
            recon_loss = self.module.compute_recon_loss(
                pred=pred,
                target=batch["target"],
                prediction_mask=batch["prediction_mask"],
                observed_mask=batch["observed_mask"],
            )

        # Add auxiliary losses
        total_loss = (task_loss + self.recon_loss_coef * recon_loss
                      + self.lb_loss_coef * lb_loss + self.kl_loss_coef * kl_loss
                      + self.ortho_loss_coef * ortho_loss)

        # Log losses
        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        loss_name = "L1Loss" if self.loss_type == "l1" else "L2Loss"
        self.log(
            f"val/{loss_name}",
            task_loss,
            on_step=self.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self.log(
            "val/load_balance_loss",
            lb_loss,
            on_step=self.log_on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self.log(
            "val/kl_loss",
            kl_loss,
            on_step=self.log_on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self.log(
            "val/ortho_loss",
            ortho_loss,
            on_step=self.log_on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self.log(
            "val/recon_loss",
            recon_loss,
            on_step=self.log_on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self.log(
            "val/total_loss",
            total_loss,
            on_step=self.log_on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )

        # Also compute the other loss for comparison
        other_loss_type = "l2" if self.loss_type == "l1" else "l1"
        other_loss_name = "L2Loss" if self.loss_type == "l1" else "L1Loss"

        # Compute other loss manually (use ~observed_mask to match training loss logic)
        mask = batch["prediction_mask"].unsqueeze(-1) * ~batch["observed_mask"]
        if other_loss_type == "l1":
            other_loss = ((pred - batch["target"]).abs() * mask).sum() / mask.sum()
        else:
            other_loss = ((pred - batch["target"]).pow(2) * mask).sum() / mask.sum()

        self.log(
            f"val/{other_loss_name}",
            other_loss,
            on_step=self.log_on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )

        return total_loss
    
    def configure_optimizers(self) -> dict:
        """
        Configure optimizer and learning rate scheduler.
        """
        decay = set()
        no_decay = set()

        whitelist_params = (
            LearnedProjection,
            MultiInSizeLinear,
            MultiOutSizeLinear,
            FeatLinear,
            nn.Linear,
        )
        blacklist_params = (
            BinaryAttentionBias,
            LearnedEmbedding,
            RMSNorm,
            nn.Embedding,
            nn.LayerNorm,
        )

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue

                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_params):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_params):
                    no_decay.add(fpn)
                elif pn in ("alpha", "beta"):
                    no_decay.add(fpn)

        # Validate parameter separation
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {inter_params} in both sets!"
        assert len(param_dict.keys() - union_params) == 0, f"parameters {param_dict.keys() - union_params} not separated!"

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay)) if param_dict[pn].requires_grad],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay)) if param_dict[pn].requires_grad],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            eps=1e-6,
        )

        # Map scheduler type string to SchedulerType enum
        scheduler_type_map = {
            "linear": SchedulerType.LINEAR,
            "cosine": SchedulerType.COSINE,
            "cosine_with_restarts": SchedulerType.COSINE_WITH_RESTARTS,
            "constant_with_warmup": SchedulerType.CONSTANT_WITH_WARMUP,
            "polynomial": SchedulerType.POLYNOMIAL,
            "inverse_sqrt": SchedulerType.INVERSE_SQRT,
        }
        scheduler_type = scheduler_type_map.get(
            self.scheduler_type, SchedulerType.COSINE_WITH_RESTARTS
        )

        # Only pass num_cycles to schedulers that support it
        scheduler_kwargs = {}
        if scheduler_type in (SchedulerType.COSINE, SchedulerType.COSINE_WITH_RESTARTS):
            scheduler_kwargs["num_cycles"] = self.num_cycles
        if scheduler_type == SchedulerType.LINEAR:
            scheduler_kwargs["min_lr_ratio"] = 1e-6 / self.lr  # floor at 1e-6 absolute LR

        scheduler = get_scheduler(
            scheduler_type,
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
            scheduler_specific_kwargs=scheduler_kwargs if scheduler_kwargs else None,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    
    def on_before_optimizer_step(self, optimizer):
        """
        Log the gradient norm before optimizer.step().

        Under mixed precision, GradScaler skips the optimizer step when gradients
        contain NaN or Inf. This hook must not modify gradients, because that
        would break the scaler's state tracking.

        Do not clip gradients here. Lightning clips them through the trainer's
        gradient_clip_val before this hook runs.
        """
        # Compute and log gradient norm
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # Log the gradient norm every 100 steps.
        if self.global_step % 100 == 0:
            self.log("train/grad_norm", total_norm, on_step=True, prog_bar=False, logger=True)
    
    def on_train_start(self):
        """Print the training and routing configuration at training start."""
        self.print(f"AMETS Training Configuration:")
        self.print(f"  - Loss type: {self.loss_type}")
        self.print(f"  - Learning rate: {self.lr}")
        self.print(f"  - LR scheduler: {self.scheduler_type}")
        self.print(f"  - Weight decay: {self.weight_decay}")

        if hasattr(self.module, 'lambda_max'):
            self.print(f"  - Lambda max: {self.module.lambda_max}")
            self.print(f"  - Lambda schedule: {self.module.lambda_schedule}")
            self.print(f"  - Num regimes: {self.module.num_regimes}")
    
    @property
    def train_transform_map(self) -> dict[str, Callable[..., Transformation]]:
        """
        Get a dictionary of Transforms for training data preprocessing.
        
        Returns:
            defaultdict with default Transform
        """

        def default_train_transform():
            return (
                SampleDimension(
                    max_dim=self.hparams.max_dim,
                    fields=("target",),
                )
                + GetPatchSize(
                    min_time_patches=self.hparams.min_patches,
                    target_field="target",
                    patch_sizes=self.module.patch_sizes,
                    patch_size_constraints=DefaultPatchSizeConstraints(),
                    offset=True,
                )
                + PatchCrop(
                    min_time_patches=self.hparams.min_patches,
                    max_patches=self.module.max_seq_len,
                    will_flatten=True,
                    offset=True,
                    fields=("target",),
                )
                + PackFields(
                    output_field="target",
                    fields=("target",),
                    feat=False,
                )
                + AddObservedMask(
                    fields=("target",),
                    observed_mask_field="observed_mask",
                    collection_type=dict,
                )
                + ImputeTimeSeries(
                    fields=("target",),
                    imputation_method=DummyValueImputation(value=0.0),
                )
                + Patchify(
                    max_patch_size=max(self.module.patch_sizes),
                    fields=("target", "observed_mask"),
                )
                + AddVariateIndex(
                    fields=("target",),
                    optional_fields=(),
                    variate_id_field="variate_id",
                    expected_ndim=3,
                    max_dim=self.hparams.max_dim,
                    randomize=False,
                    collection_type=dict,
                )
                + AddTimeIndex(
                    fields=("target",),
                    optional_fields=(),
                    time_id_field="time_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                + MaskedPrediction(
                    min_mask_ratio=self.hparams.min_mask_ratio,
                    max_mask_ratio=self.hparams.max_mask_ratio,
                    target_field="target",
                    truncate_fields=("variate_id", "time_id", "observed_mask"),
                    prediction_mask_field="prediction_mask",
                    expected_ndim=3,
                )
                # SaveContextTarget runs after MaskedPrediction. forward() feeds
                # the context portion to the descriptor predictor.
                + SaveContextTarget(
                    target_field="target",
                    prediction_mask_field="prediction_mask",
                    output_field="context_target",
                )
                + FlatPackCollection(
                    field="variate_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="time_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="prediction_mask",
                    feat=False,
                )
                + FlatPackCollection(
                    field="observed_mask",
                    feat=True,
                )
                + FlatPackFields(
                    output_field="target",
                    fields=("target",),
                    feat=True,
                )
                + SequencifyField(field="patch_size", target_field="target")
                + SelectFields(fields=list(self.seq_fields))
            )

        return defaultdict(lambda: default_train_transform)

