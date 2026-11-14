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
Router logging callback for AMETS.

The callback prints routing statistics during training, covering expert usage
and the shared-mass gate.
"""

import torch
from lightning.pytorch.callbacks import Callback


class RouterLoggingCallback(Callback):
    """
    Lightning callback that logs routing statistics.
    
    Logs:
    - Regime probabilities (mean across batch)
    - Expert usage per layer
    - Shared-mass gate values (π_sh) per layer
    - Routing entropy per layer
    - Lambda values per layer
    - Alpha and beta parameters
    """
    
    def __init__(
        self,
        log_interval: int = 100,
        log_on_epoch_end: bool = True,
    ):
        """
        Initialize the router logging callback.
        
        Args:
            log_interval: Log every N training batches
            log_on_epoch_end: Whether to also log at the end of each epoch
        """
        super().__init__()
        self.log_interval = log_interval
        self.log_on_epoch_end = log_on_epoch_end
        self.batch_count = 0
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log routing stats after training batch."""
        self.batch_count += 1

        # Only log every N batches
        if self.batch_count % self.log_interval != 0:
            return

        self._log_routing_stats(
            pl_module,
            batch,
            trainer.current_epoch,
            batch_idx,
            outputs=outputs
        )
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Log routing stats at end of epoch if configured."""
        if not self.log_on_epoch_end:
            return

        # Get a sample batch from train dataloader
        try:
            batch = next(iter(trainer.train_dataloader))
            self._log_routing_stats(
                pl_module,
                batch,
                trainer.current_epoch,
                batch_idx=None,
                epoch_end=True,
                outputs=None  # No outputs available at epoch end
            )
        except Exception as e:
            print(f"Warning: Could not log routing stats at epoch end: {e}")
    
    def _log_routing_stats(
        self,
        pl_module,
        batch,
        epoch,
        batch_idx=None,
        epoch_end=False,
        outputs=None
    ):
        """
        Extract and log routing statistics.

        Args:
            pl_module: Lightning module (AMETSPretrain)
            batch: Training batch
            epoch: Current epoch
            batch_idx: Current batch index (None if epoch end)
            epoch_end: Whether this is end-of-epoch logging
            outputs: Training outputs (for extracting losses)
        """
        try:
            # Extract routing info from model
            routing_info = pl_module.module.get_routing_info(batch)

            if routing_info is None:
                return

            # Extract loss information from module's stored per-batch values
            task_loss = getattr(pl_module, '_last_task_loss', None)
            kl_loss = getattr(pl_module, '_last_kl_loss', None)
            lb_loss = getattr(pl_module, '_last_lb_loss', None)
            total_loss = getattr(pl_module, '_last_total_loss', None)

            # Print header
            print(f"\n{'='*100}")
            if epoch_end:
                print(f"Router Statistics - Epoch {epoch+1} (End)")
            else:
                print(f"Router Statistics - Epoch {epoch+1}, Batch {batch_idx}")
            print(f"{'='*100}")

            # Print loss information
            if task_loss is not None:
                print(f"\nLoss Information:")
                loss_val = task_loss.item() if torch.is_tensor(task_loss) else task_loss
                print(f"  Task Loss: {loss_val:.6f}")
                if kl_loss is not None:
                    kl_val = kl_loss.item() if torch.is_tensor(kl_loss) else kl_loss
                    print(f"  KL Loss:   {kl_val:.6f}")
                if lb_loss is not None:
                    lb_val = lb_loss.item() if torch.is_tensor(lb_loss) else lb_loss
                    print(f"  LB Loss:   {lb_val:.6f}")
                if total_loss is not None:
                    total_val = total_loss.item() if torch.is_tensor(total_loss) else total_loss
                    print(f"  Total Loss: {total_val:.6f}")
                print()
            
            # Print regime probabilities
            if 'regime_probs' in routing_info and routing_info['regime_probs'] is not None:
                regime_probs = routing_info['regime_probs']
                print(f"\nRegime Probabilities (mean across batch):")
                feature_names = ['Forecastability', 'Seasonality', 'Trend', 'Sparsity']
                for i, name in enumerate(feature_names):
                    if i < len(regime_probs):
                        print(f"  {name:20s}: {regime_probs[i]:.4f}")
                print()
            
            # Get number of experts and layers
            expert_usage = routing_info['expert_usage_per_layer']
            num_layers = len(expert_usage)
            num_experts = len(expert_usage[0]) if num_layers > 0 else 0
            
            # Build header
            header = f"{'Layer':<6} "
            for i in range(num_experts):
                header += f"{'E'+str(i):<7} "
            header += f"{'π_sh':<7} {'Entropy':<8} {'λ':<6}"
            print(header)
            
            # Print separator
            sep = f"{'-'*6} "
            for i in range(num_experts):
                sep += f"{'-'*7} "
            sep += f"{'-'*7} {'-'*8} {'-'*6}"
            print(sep)
            
            # Print stats for each layer
            for layer_idx in range(num_layers):
                usage = expert_usage[layer_idx]
                pi_sh = routing_info['shared_mass_gate_per_layer'][layer_idx]
                entropy = routing_info['entropy_per_layer'][layer_idx]
                lambda_val = routing_info['lambda_per_layer'][layer_idx]
                
                row = f"{layer_idx:<6} "
                for u in usage:
                    row += f"{u:.4f}  "
                row += f"{pi_sh:.4f}  "
                row += f"{entropy:.4f}   "
                row += f"{lambda_val:.3f}"
                print(row)
            
            # Print shared-mass gate parameters
            alpha = routing_info.get('alpha', 0.0)
            beta = routing_info.get('beta', 0.0)
            print(f"\nShared-Mass Gate: α={alpha:.4f}, β={beta:.4f}")
            print(f"{'='*100}\n")
            
        except Exception as e:
            print(f"Warning: Could not log routing stats: {e}")
            import traceback
            traceback.print_exc()
