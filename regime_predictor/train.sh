#!/bin/bash
# Train the structural-descriptor predictor for AME-TS.
#
# Learns to map a raw time series window to four descriptors: forecastability,
# seasonality, trend, and sparsity. Writes regime_predictor_combined.pt into
# regime_predictor/trained_predictors/, which is where the model configs expect it.
#
# Usage:
#   bash regime_predictor/train.sh                  # train all three learned descriptors
#   bash regime_predictor/train.sh seasonality      # train a single descriptor
#
# Reads training pools from $PRETRAIN_DATA (never the GIFT-Eval benchmark).

set -e

# Change to the repository root, resolved from this script's location.
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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

FEATURE="$1"

# The label cache (windows, labels, item split) is reused by repeat runs. It is
# written to $REGIME_CACHE_DIR (default <repo>/regime-label-cache) unless CACHE is set.
ARGS=(--seed "${SEED:-42}" --num_workers "${NUM_WORKERS:-16}")
if [ -n "$CACHE" ]; then ARGS+=(--cache "$CACHE"); fi
if [ -n "$FEATURE" ]; then
    echo "Training descriptor: $FEATURE"
    python regime_predictor/train.py --feature "$FEATURE" "${ARGS[@]}"
else
    echo "Training forecastability, seasonality and trend"
    python regime_predictor/train.py "${ARGS[@]}"
fi

echo
echo "Done. Checkpoint written to regime_predictor/trained_predictors/"
echo "The model configs already point there, so no config change is needed."
