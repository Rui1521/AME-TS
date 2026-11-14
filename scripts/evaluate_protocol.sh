#!/bin/bash

# AMETS GIFT-Eval evaluation with selection on val and reporting on test.
#
# The script makes every selection on val and evaluates test only for the
# selected configs:
#
#   stage 1  val grid: 16 context lengths x eval_mode x patch_size on the val windows,
#            the windows just before GIFT-Eval's test windows, over 97 canonical tasks
#   select   rank per task by val MASE -> selection.json (rank-1 and same-mode top-5)
#   stage 2  test, val rank-1 config per task          -> test_valbest.csv  (primary)
#   stage 3  test, median of val top-5 point forecasts -> test_ensemble.csv (secondary)
#   score    geomean(model MASE / seasonal-naive MASE) -> scores.json
#
# Usage:
#   bash scripts/evaluate_protocol.sh <checkpoint_path> [batch_size] [gpu_ids]
#
# Environment variables:
#   STAGES    Default: all.
#             all: every stage above.
#             val: val grid, selection and val score.
#             test: stages 2 and 3 and test scores. Uses the selection.json already
#             in RESULTS_DIR, which must belong to this checkpoint.
#   CONTEXTS  For smoke tests only. Space-separated val-grid contexts that replace
#             the 16 below, for example "64 512". Must include 64, the only context
#             that runs for every task.
#
# Outputs go to $RESULTS_DIR, which defaults to
# <repo>/results_combined/<model>_ep<N>_protocol and must not already hold results
# from an earlier run:
#   val_all.csv  val_best.csv  selection.json  test_valbest.csv  test_ensemble.csv
#   scores.json, plus per-GPU logs and raw CSVs under val/ test_valbest/ test_ensemble/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

CHECKPOINT="${1:?Usage: $0 <checkpoint_path> [batch_size] [gpu_ids]}"
# Relative paths are relative to the caller's directory, not the repository root.
case "$CHECKPOINT" in /*) ;; *) CHECKPOINT="$PWD/$CHECKPOINT" ;; esac
if [ -n "${RESULTS_DIR:-}" ]; then
    case "$RESULTS_DIR" in /*) ;; *) RESULTS_DIR="$PWD/$RESULTS_DIR" ;; esac
fi
cd "$REPO_ROOT"
BATCH_SIZE="${2:-256}"
GPU_IDS="${3:-0,1,2,3,4,5,6,7}"

STAGES="${STAGES:-all}"
case "$STAGES" in
    all|val|test) ;;
    *) echo "Error: STAGES must be all, val or test (got $STAGES)"; exit 1 ;;
esac

IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

# Context grid: one short and one long context per GPU.
SHORT_CONTEXTS=(64 256 512 768 1024 1536 2048 3072)
LONG_CONTEXTS=(16384 12288 8192 6144 5120 4096 3584 2560)
if [ -n "${CONTEXTS:-}" ]; then
    read -ra SHORT_CONTEXTS <<< "$CONTEXTS"
    LONG_CONTEXTS=()
    case " $CONTEXTS " in *" 64 "*) ;; *) echo "Error: CONTEXTS must include 64"; exit 1 ;; esac
fi

# The seasonal-naive reference defines the 97 canonical tasks and the normalization.
GIFT_EVAL_REPO="${GIFT_EVAL_REPO:-$REPO_ROOT/gift-eval}"
SN_CSV="${SEASONAL_NAIVE_RESULTS:-$GIFT_EVAL_REPO/results/seasonal_naive/all_results.csv}"
if [ ! -f "$SN_CSV" ]; then
    echo "Error: seasonal-naive results not found: $SN_CSV. Set SEASONAL_NAIVE_RESULTS."
    exit 1
fi
export SEASONAL_NAIVE_RESULTS="$SN_CSV"
# evaluate.py evaluates only the tasks listed in this CSV's dataset column.
export TASK_LIST_CSV="$SN_CSV"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

CKPT_DIR=$(dirname "$CHECKPOINT")
CKPT_FILE=$(basename "$CHECKPOINT" .ckpt)
if [[ "$CKPT_DIR" == *"/checkpoints" ]]; then
    MODEL_NAME=$(basename $(dirname "$CKPT_DIR"))
    EPOCH=$(echo "$CKPT_FILE" | grep -oP 'epoch=\K[0-9]+' || echo "$CKPT_FILE")
    RUN_TAG="${MODEL_NAME}_ep${EPOCH}_protocol"
else
    MODEL_NAME=$(basename "$CKPT_DIR")
    RUN_TAG="${MODEL_NAME}_${CKPT_FILE}_protocol"
fi
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results_combined/$RUN_TAG}"

# evaluate.py appends to its CSVs, so a rerun into the same dir would duplicate rows.
if [ "$STAGES" = "test" ]; then
    FRESH="test_valbest test_ensemble test_valbest.csv test_ensemble.csv"
    if [ ! -f "$RESULTS_DIR/selection.json" ]; then
        echo "Error: STAGES=test needs $RESULTS_DIR/selection.json from a val run."
        exit 1
    fi
else
    FRESH="val test_valbest test_ensemble val_all.csv selection.json"
fi
for f in $FRESH; do
    if [ -e "$RESULTS_DIR/$f" ]; then
        echo "Error: $RESULTS_DIR/$f exists. Use a fresh RESULTS_DIR."
        exit 1
    fi
done
mkdir -p "$RESULTS_DIR"

# Hash the checkpoint once. Every row and selection.json carry the hash, and the
# test stages refuse a selection made for another checkpoint.
CKPT_SHA=$(python3 -c "import hashlib,sys
h=hashlib.sha256(); f=open(sys.argv[1],'rb')
for c in iter(lambda: f.read(1<<24), b''): h.update(c)
print(h.hexdigest())" "$CHECKPOINT")
if [ "$STAGES" = "test" ]; then
    python3 -c "import json,sys
s=json.load(open(sys.argv[1]))['checkpoint']['sha256']
sys.exit(0 if s == sys.argv[2] else 'selection.json is for checkpoint '+s+', not '+sys.argv[2])" \
        "$RESULTS_DIR/selection.json" "$CKPT_SHA"
fi

# All 55 GIFT-Eval datasets. The 97 tasks are these datasets crossed with their canonical terms.
ALL_DATASETS="m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly \
electricity/15T electricity/H electricity/D electricity/W \
solar/10T solar/H solar/D solar/W \
jena_weather/10T jena_weather/H jena_weather/D \
temperature_rain_with_missing \
ett1/15T ett1/H ett1/D ett1/W \
ett2/15T ett2/H ett2/D ett2/W \
LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D \
SZ_TAXI/15T SZ_TAXI/H \
M_DENSE/H M_DENSE/D \
hospital covid_deaths \
us_births/D us_births/M us_births/W \
car_parts_with_missing restaurant \
hierarchical_sales/D hierarchical_sales/W \
saugeenday/D saugeenday/M saugeenday/W \
kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D \
bitbrains_fast_storage/5T bitbrains_fast_storage/H \
bitbrains_rnd/5T bitbrains_rnd/H \
bizitobs_application bizitobs_service \
bizitobs_l2c/5T bizitobs_l2c/H"
read -ra DATASET_ARRAY <<< "$ALL_DATASETS"

echo "============================================================"
echo "AMETS protocol evaluation (select on val, report on test)"
echo "============================================================"
echo "  Checkpoint: $CHECKPOINT (sha256 $CKPT_SHA)"
echo "  GPUs: ${GPU_ARRAY[*]} ($NUM_GPUS GPUs)"
echo "  Batch size: $BATCH_SIZE"
echo "  Stages: $STAGES"
echo "  Seasonal naive: $SN_CSV"
echo "  Output: $RESULTS_DIR/"
echo "============================================================"

# Wait for all PIDs and report failures without aborting. Missing tasks show up in scoring.
wait_all() {
    local label=$1; shift
    local failed=0
    for pid in "$@"; do
        wait "$pid" || failed=$((failed + 1))
    done
    if [ $failed -gt 0 ]; then
        echo "WARNING: $failed $label GPU jobs failed. Check logs."
    fi
}

# Concatenate the evaluate.py CSVs found under $1 into $2. Headers must match.
merge_csvs() {
    python3 - "$1" "$2" <<'PY'
import sys
from pathlib import Path
import pandas as pd
src, out = Path(sys.argv[1]), sys.argv[2]
csvs = sorted(src.glob("**/metrics/all_results.csv"))
dfs = [pd.read_csv(p) for p in csvs]
if not dfs:
    sys.exit(f"No results under {src}")
cols = list(dfs[0].columns)
for p, d in zip(csvs, dfs):
    if list(d.columns) != cols:
        sys.exit(f"Header mismatch in {p}")
merged = pd.concat(dfs, ignore_index=True)
merged.to_csv(out, index=False)
print(f"  {len(csvs)} CSVs, {len(merged)} rows -> {out}")
PY
}

# ===== Stage 1: val grid =====
run_val_context() {
    local GPU=$1
    local CTX=$2
    local CTX_DIR="$RESULTS_DIR/val/ctx${CTX}"
    mkdir -p "$CTX_DIR"
    for dataset in $ALL_DATASETS; do
        echo "[$(date '+%H:%M:%S')] val ctx=${CTX} ${dataset}"
        CUDA_VISIBLE_DEVICES=$GPU python -u scripts/evaluate.py \
            --checkpoint "$CHECKPOINT" \
            --checkpoint_sha256 "$CKPT_SHA" \
            --datasets "$dataset" \
            --terms short medium long \
            --output_dir "$CTX_DIR" \
            --eval_mode both \
            --split val \
            --batch_size $BATCH_SIZE \
            --max_context $CTX 2>&1 || echo "  FAILED: $dataset"
    done
    echo "[$(date '+%H:%M:%S')] val ctx=${CTX} ALL DONE"
}

if [ "$STAGES" != "test" ]; then
echo ""
echo "Stage 1: val grid (${SHORT_CONTEXTS[*]} ${LONG_CONTEXTS[*]})"
mkdir -p "$RESULTS_DIR/val"
PIDS=()
# Context pair k, one short and one long context, runs on GPU k mod NUM_GPUS, so
# fewer than 8 GPUs still cover all 16 contexts.
for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU=${GPU_ARRAY[$i]}
    PAIRS=()
    for ((k = i; k < ${#SHORT_CONTEXTS[@]}; k += NUM_GPUS)); do PAIRS+=($k); done
    [ ${#PAIRS[@]} -eq 0 ] && continue
    (
        for k in "${PAIRS[@]}"; do
            run_val_context $GPU ${SHORT_CONTEXTS[$k]}
            if [ -n "${LONG_CONTEXTS[$k]:-}" ]; then
                run_val_context $GPU ${LONG_CONTEXTS[$k]}
            fi
        done
    ) > "$RESULTS_DIR/val/gpu${GPU}.log" 2>&1 &
    PIDS+=($!)
done
wait_all "val" "${PIDS[@]}"
merge_csvs "$RESULTS_DIR/val" "$RESULTS_DIR/val_all.csv"

# ===== Select on val =====
echo ""
echo "Selecting configs on val"
python3 "$SCRIPT_DIR/select_configs.py" \
    --val-csv "$RESULTS_DIR/val_all.csv" \
    --out "$RESULTS_DIR/selection.json" \
    --val-best-csv "$RESULTS_DIR/val_best.csv" \
    --seasonal-naive "$SN_CSV" \
    --checkpoint "$CHECKPOINT" \
    && SEL_RC=0 || SEL_RC=$?
if [ "$SEL_RC" -eq 2 ]; then
    echo "WARNING: some tasks have no valid val result. Test scores will be partial."
elif [ "$SEL_RC" -ne 0 ]; then
    echo "Error: selection failed with exit code $SEL_RC. Skipping test evaluation."
    exit 1
fi
fi

# ===== Stages 2 and 3: test on the selected configs only =====
# Datasets go round-robin across GPUs. Each selected config carries its own
# max_context, so the test stages need no context fan-out.
run_test_shard() {
    local GPU=$1
    local SHARD=$2
    local OUT=$3
    shift 3
    for ((j = SHARD; j < ${#DATASET_ARRAY[@]}; j += NUM_GPUS)); do
        local dataset=${DATASET_ARRAY[$j]}
        echo "[$(date '+%H:%M:%S')] test ${dataset} $*"
        CUDA_VISIBLE_DEVICES=$GPU python -u scripts/evaluate.py \
            --checkpoint "$CHECKPOINT" \
            --checkpoint_sha256 "$CKPT_SHA" \
            --datasets "$dataset" \
            --terms short medium long \
            --output_dir "$OUT/gpu${GPU}" \
            --split test \
            --selection "$RESULTS_DIR/selection.json" \
            --batch_size $BATCH_SIZE "$@" 2>&1 || echo "  FAILED: $dataset"
    done
    echo "[$(date '+%H:%M:%S')] test shard ${SHARD} ALL DONE"
}

run_test_stage() {
    local NAME=$1
    shift
    local OUT="$RESULTS_DIR/$NAME"
    mkdir -p "$OUT"
    local pids=()
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        GPU=${GPU_ARRAY[$i]}
        run_test_shard $GPU $i "$OUT" "$@" > "$OUT/gpu${GPU}.log" 2>&1 &
        pids+=($!)
    done
    wait_all "$NAME" "${pids[@]}"
    merge_csvs "$OUT" "$RESULTS_DIR/$NAME.csv"
}

if [ "$STAGES" != "val" ]; then
echo ""
echo "Stage 2: test, val rank-1 config per task"
run_test_stage test_valbest

echo ""
echo "Stage 3: test, median of val top-5 configs"
run_test_stage test_ensemble --ensemble top5
fi

# ===== Score =====
echo ""
echo "Scoring"
python3 - "$RESULTS_DIR" "$SN_CSV" "$REPO_ROOT" "$STAGES" <<'PY'
import json, sys
from pathlib import Path
out, sn, repo, stages = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, str(Path(repo) / "scripts"))
from score import score

def safe(csv, split):
    if not (out / csv).exists():
        return {"split": split, "score": None, "partial": True, "error": f"{csv} missing"}
    return score(out / csv, sn, split)

scores = {}
if stages != "test":
    scores["val"] = {"val_best": safe("val_best.csv", "val")}
if stages != "val":
    scores["test"] = {"val_best": safe("test_valbest.csv", "test"),
                      "val_top5_ensemble": safe("test_ensemble.csv", "test")}
(out / "scores.json").write_text(json.dumps(scores, indent=2))
for split, entries in scores.items():
    for name, r in entries.items():
        if r["partial"]:
            print(f"  {split}/{name}: PARTIAL, no score "
                  f"({r.get('n_tasks', 0)}/{r.get('n_expected', 97)} tasks) {r.get('error', '')}")
        else:
            print(f"  {split}/{name}: {r['score']:.4f} over {r['n_tasks']} tasks  [{r['label']}]")
print(f"  -> {out / 'scores.json'}")
PY

echo ""
echo "============================================================"
echo "EVALUATION COMPLETE (STAGES=$STAGES): $RESULTS_DIR"
[ "$STAGES" != "test" ] && echo "  val_all.csv selection.json val_best.csv"
[ "$STAGES" != "val" ] && echo "  test_valbest.csv (primary)  test_ensemble.csv (secondary)"
echo "  scores.json"
echo "============================================================"
