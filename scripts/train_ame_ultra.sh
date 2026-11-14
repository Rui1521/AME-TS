#!/bin/bash
# AME-Ultra training script (8 GPUs, d1024 L6 E10 k2)
#
# Usage:
#   bash scripts/train_ame_ultra.sh                    # Start fresh training
#   bash scripts/train_ame_ultra.sh resume             # Resume from last checkpoint
#   bash scripts/train_ame_ultra.sh test               # Quick test run (2 epochs, 20 batches)
#
# Configuration:
#   - configs/ame_ultra.yaml       (top-level trainer settings)
#   - configs/model/ame_ultra.yaml (model architecture: d1024 L6 E10 k2)
#   - configs/data/ame_ultra.yaml  (pretraining data pool + weights)

set -e

# Run from the repository root, resolved from this script's location, so the
# script works from any checkout and any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

# Activate the conda env if conda is present. Otherwise assume the caller
# has already activated a suitable environment.
CONDA_ENV="${CONDA_ENV:-amets}"
for _c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    if [ -f "$_c/etc/profile.d/conda.sh" ]; then
        source "$_c/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
        break
    fi
done

# Multi-GPU env
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

MODE=${1:-"train"}

case $MODE in
    "resume")
        echo "=========================================="
        echo "Resuming AME-Ultra from checkpoint"
        echo "=========================================="
        HYDRA_FULL_ERROR=1 python scripts/train.py \
            --config-name ame_ultra \
            ckpt_path=last
        ;;
    "test")
        echo "=========================================="
        echo "AME-Ultra test run (2 epochs, 20 batches)"
        echo "=========================================="
        HYDRA_FULL_ERROR=1 python scripts/train.py \
            --config-name ame_ultra \
            run_name=ame_ultra_test \
            trainer.max_epochs=2 \
            train_dataloader.num_batches_per_epoch=20
        ;;
    *)
        echo "=========================================="
        echo "Starting AME-Ultra full training"
        echo "=========================================="
        HYDRA_FULL_ERROR=1 python scripts/train.py \
            --config-name ame_ultra
        ;;
esac

echo "=========================================="
echo "Training complete."
echo "=========================================="
