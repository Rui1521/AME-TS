#!/bin/bash
# AME-Tiny training script (8 GPUs, d256 L4 E5 k2)
#
# Usage:
#   bash scripts/train_ame_tiny.sh                    # Start fresh training
#   bash scripts/train_ame_tiny.sh resume             # Resume from last checkpoint
#   bash scripts/train_ame_tiny.sh test               # Quick test run (2 epochs, 20 batches)

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

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

MODE=${1:-"train"}

case $MODE in
    "resume")
        echo "=========================================="
        echo "Resuming AME-Tiny from checkpoint"
        echo "=========================================="
        HYDRA_FULL_ERROR=1 python scripts/train.py \
            --config-name ame_tiny \
            ckpt_path=last
        ;;
    "test")
        echo "=========================================="
        echo "AME-Tiny test run (2 epochs, 20 batches)"
        echo "=========================================="
        HYDRA_FULL_ERROR=1 python scripts/train.py \
            --config-name ame_tiny \
            run_name=ame_tiny_test \
            trainer.max_epochs=2 \
            train_dataloader.num_batches_per_epoch=20
        ;;
    *)
        echo "=========================================="
        echo "Starting AME-Tiny full training"
        echo "=========================================="
        HYDRA_FULL_ERROR=1 python scripts/train.py \
            --config-name ame_tiny
        ;;
esac

echo "=========================================="
echo "Training complete."
echo "=========================================="