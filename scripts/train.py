#!/usr/bin/env python3
"""
Training script for AME-TS, driven by Hydra.

Trains a structure-aware Mixture-of-Experts model with PyTorch Lightning on the
pre-training pools under $PRETRAIN_DATA. Run from the repository root.

Usage:
    # Train a model+data pair
    python scripts/train.py --config-name ame_base

    # Point at a specific descriptor predictor
    python scripts/train.py --config-name ame_base \
        model.regime_predictor_path=regime_predictor/trained_predictors/regime_predictor_combined.pt

    # Override hyperparameters
    python scripts/train.py --config-name ame_base \
        model.lambda_max=1.5 \
        model.kl_loss_coef=50.0 \
        trainer.max_epochs=100

    # Resume from the last checkpoint
    python scripts/train.py --config-name ame_base ckpt_path=last
"""

import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import lightning as L

import warnings
warnings.filterwarnings('ignore')

import torch
torch.set_warn_always(False)


# Put the repository root on sys.path.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Importing env exports AMETS_ROOT, PRETRAIN_DATA and GIFT_EVAL_DATA into
# os.environ, with repo-relative defaults for any variable not already set or
# present in .env. The import must run before Hydra composes, because the
# configs read these variables through ${oc.env:...}.
from amets.common.env import env  # noqa: E402,F401

# Register custom Hydra resolvers
OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
OmegaConf.register_new_resolver("floordiv", lambda x, y: x // y)
OmegaConf.register_new_resolver("as_tuple", lambda x: tuple(x))

def cls_getattr_resolver(cls_path: str, attr: str):
    """Return attribute ``attr`` of the class at dotted path ``cls_path``."""
    import importlib
    module_path, class_name = cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return getattr(cls, attr)

OmegaConf.register_new_resolver("cls_getattr", cls_getattr_resolver)


@hydra.main(version_base="1.3", config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Train the model described by the Hydra config ``cfg``."""
    # Print configuration
    print("=" * 80)
    print("AME-TS Training Configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Set random seed
    if cfg.seed is not None:
        L.seed_everything(cfg.seed, workers=True)
    
    # Enable TF32 matmuls on Ampere and newer GPUs.
    if cfg.tf32:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled")
    
    # Instantiate model
    print("\nInstantiating model...")
    model = hydra.utils.instantiate(cfg.model)

    # Load model weights only, without optimizer or scheduler state, if requested.
    weights_only_ckpt = cfg.get("weights_only_ckpt", None)
    if weights_only_ckpt:
        import torch
        ckpt_path = str(weights_only_ckpt).strip("'\"")
        print(f"Loading weights only from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print("  Loaded model weights. Training starts with a new optimizer and scheduler.")

    print(f"Model: {model.__class__.__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Instantiate data module and load dataset with transforms
    print("\nInstantiating data module...")
    data_module = hydra.utils.instantiate(cfg.data)
    print(f"Data module: {data_module.__class__.__name__}")
    
    # Load the dataset with the model's transform map, which includes descriptor prediction.
    print("\nLoading dataset with transforms...")
    train_dataset = data_module.load_dataset(model.train_transform_map)
    print(f"Train dataset loaded with transforms")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_dataloader = hydra.utils.instantiate(
        cfg.train_dataloader,
        dataset=train_dataset,
    )
    print(f"Train dataloader: {len(train_dataloader.dataloader.dataset)} samples")
    
    # Instantiate the trainer, which also instantiates the callbacks in the config.
    print("\nInstantiating trainer...")
    trainer = hydra.utils.instantiate(cfg.trainer)
    
    # Add router logging callback if enabled
    if cfg.get('router_logging', {}).get('enabled', False):
        print("\nRouter logging enabled")
        from amets.callbacks import RouterLoggingCallback
        router_callback = RouterLoggingCallback(
            log_interval=cfg.router_logging.get('log_interval', 100),
            log_on_epoch_end=cfg.router_logging.get('log_on_epoch_end', True),
        )
        trainer.callbacks.append(router_callback)
        print(f"  - Log interval: {cfg.router_logging.get('log_interval', 100)} batches")
        print(f"  - Log on epoch end: {cfg.router_logging.get('log_on_epoch_end', True)}")
    
    print(f"Trainer: {trainer.__class__.__name__}")
    print(f"Max epochs: {trainer.max_epochs}")
    print(f"Devices: {trainer.num_devices}")
    print(f"Precision: {trainer.precision}")
    print(f"Callbacks: {len(trainer.callbacks)}")
    
    # Compile model if requested
    if cfg.compile:
        print(f"\nCompiling model with mode: {cfg.compile}")
        import torch
        model = torch.compile(model, mode=cfg.compile)
    
    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        ckpt_path=cfg.ckpt_path,
    )
    
    print("\n" + "=" * 80)
    print("Training complete.")
    print("=" * 80)
    print(f"Checkpoints saved to: {HydraConfig.get().runtime.output_dir}/checkpoints")
    print(f"Logs saved to: {trainer.log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
