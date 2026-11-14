# Trained regime predictors

This directory holds the regime predictor checkpoint. The repository does not include weights.

Every model config under `configs/model/` loads this file:

```
regime_predictor/trained_predictors/regime_predictor_combined.pt
```

## What the checkpoint contains

`CombinedRegimePredictor` maps a raw time series window to four structural descriptors.

- Forecastability comes from a small CNN of class `SingleFeaturePredictor`.
- Seasonality comes from a small CNN of the same class.
- Trend comes from a small CNN of the same class.
- Sparsity is computed analytically and has no weights.

Each score is an independent value in [0, 1]. The three CNNs produce sigmoid outputs, and sparsity takes its analytic value. The scores form the structural prior over experts.

## Creating it

Run the training script from the repository root. It needs the pre-training pools under `PRETRAIN_DATA`.

```bash
python regime_predictor/train.py                  # all three learned descriptors
python regime_predictor/train.py --feature trend  # one descriptor
bash regime_predictor/train.sh                    # launcher for the same script
```

The script writes one file per descriptor, `regime_predictor_<descriptor>.pt`. It writes `regime_predictor_combined.pt` once all three descriptor files exist and come from the same label cache. The label cache lives in `REGIME_CACHE_DIR`, and `--cache` selects another file.

## Training without it

Set the path to null to run with a randomly initialized predictor:

```bash
python scripts/train.py --config-name ame_base model.regime_predictor_path=null
```

Every model config sets `freeze_regime_predictor: true`, so the predictor keeps its random weights and the descriptors carry no structural signal. Use this only to check that the pipeline runs. If the path is set and the file is missing, loading raises a `FileNotFoundError` that points to this file.
