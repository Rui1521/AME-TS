# AME-TS

AME-TS (Anchored Mixture-of-Experts for Time Series Forecasting) is a sparse time series foundation model. It adds structure-aware routing to a Mixture-of-Experts transformer.

This repository holds the code for the paper [AME-TS: Anchored Mixture-of-Experts for Time Series Forecasting](https://openreview.net/forum?id=NtOUejNcpe) by Rui Wang, Renhao Xue, Ray Razi, Huan Song, and Hannah R. Marlowe. The paper was accepted at NeurIPS 2026.

A small regime predictor estimates four interpretable descriptors from the raw series: forecastability, seasonality, trend, and sparsity. The model maps the descriptors to a soft structural prior over experts. During training, this series-level prior guides token-level routing, so experts specialize along temporal structure.

The four descriptor scores are independent values in [0, 1], not a softmax. Three are sigmoid outputs of small CNNs, and the predictor computes sparsity directly. A window that is both trending and seasonal puts weight on both descriptors. The prior is therefore multi-label.

The [paper](https://openreview.net/forum?id=NtOUejNcpe) describes the method and reports the experiments and results.

This is research code. Interfaces may change, and the repository has no test suite and no CI.

## Installation

```bash
git clone https://github.com/Rui1521/AME-TS.git
cd AME-TS

conda env create -f environment.yml
conda activate amets
pip install -e .
```

You can also run `pip install -e .` in an existing environment. The package needs Python 3.10 or newer, and `requirements.txt` lists the dependencies.

Training needs a GPU. The configs set `trainer.devices` to 8, and the `scripts/train_ame_<scale>.sh` launchers expect 8 GPUs. To train on another number of GPUs, run `scripts/train.py` and override `trainer.devices`.

### Paths

All paths resolve relative to the repository root by default, so a fresh clone needs no configuration. To change a location, set the environment variable or put it in a `.env` file at the repository root. `amets/common/env.py` loads `.env` and exports the defaults before Hydra composes a config, so `${oc.env:...}` interpolation works in both cases.

- `GIFT_EVAL_REPO` is the gift-eval code checkout. The default is `<repo>/gift-eval`.
- `GIFT_EVAL_DATA` is the GIFT-Eval benchmark data. The default is `<repo>/gift-eval-data`.
- `GIFT_EVAL` is the name that the `gift_eval` package reads for the same data. If you set only one of `GIFT_EVAL_DATA` and `GIFT_EVAL`, `env.py` copies its value to the other.
- `PRETRAIN_DATA` is the root of the pre-training data pools. The default is `<repo>/pretrain-data`.
- `REGIME_CACHE_DIR` holds the label cache for regime predictor training. The default is `<repo>/regime-label-cache`.
- `RESULTS_DIR` is where `scripts/evaluate_protocol.sh` writes its results. The default is `<repo>/results_combined/<run>`, and the script derives `<run>` from the checkpoint path.
- `SEASONAL_NAIVE_RESULTS` is the seasonal naive baseline CSV. The default is `results/seasonal_naive/all_results.csv` in the gift-eval checkout. The protocol scripts need it for the task list and for normalized scores.
- `TASK_LIST_CSV` is an optional CSV. If it is set, `scripts/evaluate.py` evaluates only the tasks listed in its `dataset` column. `scripts/evaluate_protocol.sh` sets it to the seasonal naive CSV.
- `LOTSA_V1_PATH` is the output directory of the LOTSA builders in `amets/data/builder/lotsa_v1/`. It has no default.
- `CONDA_ENV` names the conda environment that the shell launchers activate. The default is `amets`. If the launchers find no conda installation, they run in the current environment.

## Data setup

### GIFT-Eval

This repository does not include GIFT-Eval code or data. Both come from Salesforce.

```bash
# Benchmark code (provides the gift_eval package)
git clone https://github.com/SalesforceAIResearch/gift-eval

# Benchmark data
huggingface-cli download Salesforce/GiftEval --repo-type=dataset \
    --local-dir gift-eval-data
```

Recent versions of `huggingface_hub` replace `huggingface-cli download` with `hf download`, which takes the same arguments. If you run both commands from the repository root, the default paths work. To keep the code or data elsewhere, set `GIFT_EVAL_REPO` and `GIFT_EVAL_DATA`. Evaluation reads `dataset_properties.json` from the gift-eval checkout.

### Pre-training data

The training pools live under `PRETRAIN_DATA`. They stay apart from the full benchmark in `GIFT_EVAL_DATA`, which holds the val and test windows. The data configs under `configs/data/` name the datasets that each scale uses. Run these scripts from the repository root.

- `scripts/build_train_split.py` writes the GIFT-Eval training split. It cuts every series before the validation and test windows.
- `scripts/build_synthetic_dataset.py` generates the synthetic series dataset.
- `scripts/download_lotsa.py` downloads LOTSA datasets from Hugging Face. Pass `--only` with config names to download a subset.
- `scripts/download_chronos.py` downloads Chronos training datasets. By default it downloads KernelSynth, and `--only` selects other configs.

`GiftEvalDatasetBuilder` and `regime_predictor/train.py` check the training root with `amets/data/pool_guard.py` before they load any data.

- The check refuses a root that overlaps `GIFT_EVAL_DATA`.
- It refuses GIFT-Eval datasets unless the root holds the train-split manifest and the `.pool_gift-eval-train.json` marker. `scripts/build_train_split.py` writes both files.
- It refuses the datasets listed in `BANNED_DATASETS` in `pool_guard.py`.

## Training

AME-TS trains in two stages. Stage 1 trains the regime predictor, and stage 2 pre-trains the model.

### Stage 1: regime predictor

The descriptor labels come from a feature pipeline in `regime_predictor/feature_computation/`. It computes forecastability from spectral predictability, plus seasonality, trend, and sparsity.

`regime_predictor/train.py` crops windows from the pools under `PRETRAIN_DATA`, computes the descriptor labels, and trains one small CNN per learned descriptor. Sparsity is analytic and has no weights.

```bash
python regime_predictor/train.py                  # forecastability, seasonality, trend
python regime_predictor/train.py --feature trend  # one descriptor
bash regime_predictor/train.sh                    # launcher for the same script
bash regime_predictor/train.sh trend              # launcher, one descriptor
```

The script stores the label cache in `REGIME_CACHE_DIR` and reuses it on later runs. It writes `regime_predictor/trained_predictors/regime_predictor_combined.pt`, which every model config loads. See `regime_predictor/trained_predictors/README.md`.

### Stage 2: AME-TS pre-training

Training is Hydra-driven. Run it from the repository root with one config per scale: `ame_tiny`, `ame_small`, `ame_base`, `ame_large`, or `ame_ultra`. Without `--config-name`, `scripts/train.py` uses `configs/default.yaml`, which selects `ame_tiny`.

```bash
python scripts/train.py --config-name ame_<scale>
python scripts/train.py --config-name ame_<scale> ckpt_path=last   # resume

# smoke test on one GPU
python scripts/train.py --config-name ame_tiny trainer.devices=1 trainer.max_epochs=2 \
    train_dataloader.num_batches_per_epoch=20 model.num_warmup_steps=5

# override any hyperparameter
python scripts/train.py --config-name ame_base model.lambda_max=1.5
```

Set `HYDRA_FULL_ERROR=1` to see full tracebacks. The launchers `scripts/train_ame_<scale>.sh` run the same command on 8 GPUs. They accept `resume` or `test` as the first argument. Each run writes checkpoints and logs under `outputs/pretrain/<model>/<run_name>/`.

## How routing works

Two files implement the mechanism.

1. `amets/model/combined_regime_predictor.py` holds the regime predictor that emits the four descriptor scores.
2. `amets/module/ffn_moe.py` holds `MoEAnchoredFeedForward`, which builds the structural prior and routes tokens.

`MoEAnchoredFeedForward` works in three steps.

- It maps the descriptors onto expert slots and leaves a share of the experts unanchored as shared experts.
- An entropy-based gate moves weight to the shared experts when the descriptors are ambiguous.
- The resulting prior acts on the router according to `routing_mode`.

The `routing_mode` option takes one of two values.

- `kl_divergence` keeps the gate logits as pure learned routing. The prior enters only through a training-time KL alignment loss.
- `additive` adds the log prior to the router logits, so the prior also steers routing at inference.

The anchor strength changes with depth. `lambda_max` sets its largest value, and `lambda_schedule` sets how it varies across layers. The configs under `configs/model/` set these options. All five use `kl_divergence`.

## Evaluation

Evaluation uses GIFT-Eval. Run `scripts/evaluate.py` from the repository root. The `--split` flag is required and accepts `val` or `test`. The val split holds the same number of windows as test and sits immediately before the test windows, so no val input contains a test point.

```bash
python scripts/evaluate.py --checkpoint <path.ckpt> --split val \
    --datasets m4_weekly ett1/H --terms short medium long \
    --eval_mode both --output_dir <dir> --max_context 1024
```

- `--eval_mode` is `univariate`, `multivariate`, or `both`.
- `--max_context` sets the context length. The default is 512.
- `--selection <selection.json>` evaluates only the configurations chosen on the val split. Each chosen configuration carries its own context length, so `--max_context` cannot be combined with it.
- `--ensemble top5` takes the median of the point forecasts of the val top-5 configurations. It requires `--selection`.
- `--split test` requires `--selection`. `--allow-unselected-test` lifts this check for diagnostics only.
- `--per_item_eval` runs the slow per-item reference path instead of the batched default.

`scripts/evaluate_protocol.sh` runs the full protocol, sharded over GPUs:

```bash
bash scripts/evaluate_protocol.sh <checkpoint> [batch_size] [gpu_ids]
```

The batch size defaults to 256. The GPU list is comma-separated and defaults to `0,1,2,3,4,5,6,7`.

The script runs four steps.

1. It evaluates a val grid of 16 context lengths, the univariate and multivariate modes, and every valid patch size on all 55 dataset configurations.
2. `scripts/select_configs.py` ranks the candidates of each task by val MASE and writes `selection.json`.
3. It evaluates test with the rank-1 val config of each task, and again with the median of the val top-5 configs.
4. `scripts/score.py` computes the geometric mean of the model MASE over the seasonal naive MASE and writes `scores.json`.

All outputs go to `RESULTS_DIR`. The script stops if that directory already holds results, so use a new directory for each run. Set `STAGES=val` to stop after selection, or `STAGES=test` to run the test steps from an existing `selection.json`.

GIFT-Eval normalizes scores by the seasonal naive baseline. `evaluation/eval_utils.py` reads the same `SEASONAL_NAIVE_RESULTS` CSV and reports only raw metrics when the file is absent. The notebook `evaluation/evaluation.ipynb` plots and aggregates results.

## Repository layout

```text
amets/              model, modules, data pipeline, transforms, callbacks
configs/            Hydra configs: top-level ame_<scale> and default, model/, data/
scripts/            train, evaluate, data preparation, launchers
regime_predictor/   descriptor feature pipeline and regime predictor training
evaluation/         result aggregation utilities and analysis notebook
stats/              dataset statistics helpers
environment.yml     conda environment
llms.txt            short index of the repository
```

## Citation

If you use this code, please cite the [paper](https://openreview.net/forum?id=NtOUejNcpe).

```bibtex
@inproceedings{wang2026amets,
  title     = {{AME-TS}: Anchored Mixture-of-Experts for Time Series Forecasting},
  author    = {Wang, Rui and Xue, Renhao and Razi, Ray and Song, Huan and Marlowe, Hannah R.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=NtOUejNcpe}
}
```

## License and attribution

AME-TS is released under the Apache License 2.0. See `LICENSE`.

The structure-aware routing is new in AME-TS. It covers the regime predictor, the structural prior, and the anchored Mixture-of-Experts layers. The training recipe and the evaluation pipeline are also new. The data pipeline, transforms, and transformer building blocks build on Salesforce's [uni2ts](https://github.com/SalesforceAIResearch/uni2ts), which is also Apache-2.0. Files derived from uni2ts keep their original copyright headers.

Evaluation depends on [GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval), which you obtain separately and which this repository does not redistribute. See `NOTICE` for the full attribution.
