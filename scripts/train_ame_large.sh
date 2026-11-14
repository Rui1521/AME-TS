#!/bin/bash
# AME-Large training script (8 GPUs)
#
# Usage:
#   bash scripts/train_ame_large.sh                    # Start fresh training
#   bash scripts/train_ame_large.sh resume             # Resume from last checkpoint
#   bash scripts/train_ame_large.sh test               # Quick test run (2 epochs, 20 batches)

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
        echo "Resuming AME-Large from checkpoint"
        HYDRA_FULL_ERROR=1 python scripts/train.py \
            --config-name ame_large \
            ckpt_path=last
        ;;
    "test")
        echo "AME-Large test run (2 epochs, 20 batches)"
        HYDRA_FULL_ERROR=1 python scripts/train.py \
            --config-name ame_large \
            run_name=ame_large_test \
            trainer.max_epochs=2 \
            train_dataloader.num_batches_per_epoch=20
        ;;
    *)
        echo "Starting AME-Large full training"
        HYDRA_FULL_ERROR=1 python scripts/train.py \
            --config-name ame_large
        ;;
esac

echo "Training complete."
