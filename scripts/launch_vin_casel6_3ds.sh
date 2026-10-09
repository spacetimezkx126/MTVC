#!/usr/bin/env bash
# Virtual-node connect ablation (keep market day-phase g):
#   mkt_as_ind : market connect → concat like industry (Proj([mean‖g_day]))
#   ind_as_mkt : industry connect → Hadamard like market (mean⊙v_k)
# Force market-global ON on all 3 datasets so both variants are meaningful.
# Stack: partial_unified + case-mine L6 → vin_{mkt_as_ind,ind_as_mkt}_casel6.
set -u
REPRO="$(cd "$(dirname "$0")/.." && pwd)"
DUAL="$(cd "$REPRO/.." && pwd)"
cd "$DUAL"
export DUAL_TF_MODEL_DIR="$DUAL"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MTVC_USE_TYPE_EMB=1 MTVC_UNIFIED_CAUSAL=1 MTVC_TRANSFORMER_LAYERS=6
export MTVC_SOFT_PRUNE=0 MTVC_UNIFIED_SAME_DAY=0 MTVC_SHARE_PRICE_NEWS_ENC=0
export DUAL_TF_CUDA_LOCK="${DUAL_TF_CUDA_LOCK:-1}"
unset MTVC_FIXED_TRAIN_ORDER MTVC_UNIFIED_LAYERS MTVC_HARD_FINETUNE MTVC_LOOP_ROUNDS 2>/dev/null || true

PY=/home/zhaokx/miniconda3/envs/CAMEF/bin/python
RUN="$REPRO/code/run_mtvc.py"
VOCAB_CSMD="$REPRO/dict/dict_csmd.pkl"
VOCAB_MASSIVE="$REPRO/dict/dict_massive.pkl"
OS_EXTRA="$REPRO/assets/massive_oversample/train_oversample_extra.jsonl"

DATASETS=(${DATASETS:-csmd50 csmd300 massive})
VARIANTS=(${VARIANTS:-mkt_as_ind ind_as_mkt})
SEEDS=(${SEEDS:-42 43 44 45 46})
ALLOWED_GPUS=(${ALLOWED_GPUS:-0 1})
MAX_PARALLEL="${MAX_PARALLEL:-8}"
POLL_SEC="${POLL_SEC:-25}"
CASE_LAYER=6

job_done() {
  local log="$1"
  [[ -f "$log" ]] || return 1
  grep -q "Training finished" "$log" 2>/dev/null || return 1
  grep -qiE "OutOfMemoryError|Traceback|CUDA_FATAL" "$log" 2>/dev/null && return 1
  local dir; dir="$(dirname "$log")"
  [[ -f "$dir/best.pt" && -f "$dir/best.pt.meta.json" ]] || return 1
  return 0
}

min_free_for() {
  case "$1" in
    csmd50) echo 3200 ;;
    csmd300) echo 3500 ;;
    massive) echo 4500 ;;
    *) echo 3500 ;;
  esac
}

pick_gpu() {
  local need="$1" best="" best_free=-1 idx free
  for idx in "${ALLOWED_GPUS[@]}"; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$idx" | tr -d ' ')
    if (( free >= need && free > best_free )); then
      best_free=$free
      best=$idx
    fi
  done
  [[ -n "$best" ]] || return 1
  echo "$best"
}

method_name() {
  local ds="$1" v="$2"
  echo "vin_${v}_casel${CASE_LAYER}"
}

start_one() {
  local ds="$1" v="$2" seed="$3" gpu="$4"
  local method ckpt root vocab extra=() mlp=full
  method="$(method_name "$ds" "$v")"
  ckpt="$REPRO/checkpoints/$ds/$method/s${seed}"
  mkdir -p "$ckpt"
  if job_done "$ckpt/train.log"; then
    echo "[skip] done $ds $v s${seed}"
    return 0
  fi
  case "$ds" in
    csmd50)
      root="$REPRO/data/CSMD50"
      vocab="$VOCAB_CSMD"
      extra=(
        --train_start 2021-01-01 --train_end 2023-01-01
        --val_start 2023-01-02 --val_end 2024-01-02
        --test_start 2024-01-03 --test_end 2024-12-31)
      ;;
    csmd300)
      root="${CSMD300_ROOT:-$REPRO/data/CSMD300}"
      vocab="$VOCAB_CSMD"
      extra=(
        --train_start 2021-01-01 --train_end 2023-01-01
        --val_start 2023-01-02 --val_end 2024-01-02
        --test_start 2024-01-03 --test_end 2024-12-31)
      ;;
    massive)
      root="${MASSIVE_ROOT:-$REPRO/data/massive_data}"
      vocab="$VOCAB_MASSIVE"
      extra=(--train_oversample_extra "$OS_EXTRA"
        --train_start 2022-01-01 --train_end 2023-12-31
        --val_start 2024-01-01 --val_end 2024-12-31
        --test_start 2025-01-01 --test_end 2025-12-31)
      ;;
    *) echo "bad ds $ds"; return 1 ;;
  esac

  : >"$ckpt/train.log"
  echo "[start] $(date '+%F %T') $ds $v s${seed} gpu=$gpu -> $method"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$RUN" -d "$ds" \
    --mode train --emb_dim 128 --news_emb_dim 32 \
    --epochs 50 --es_start_epoch 1 --es_patience 5 --es_metric acc_mcc \
    --split_mode contiguous --bce_pos_weight 1.0 \
    --news_word_emb random --news_text_tf_layers 3 --news_padding_k 5 \
    --industry_fuse_site pred_fusion \
    --vin_connect_mode "$v" \
    --price_encoder partial_unified \
    --pred_fusion_ablation_mode "$mlp" --news_ablation_mode full \
    --contrast_aux_weight 1.0 --contrast_mode mtvc \
    --sim_quantile 0.75 \
    --mtvc_lambda_pair 0.2 --mtvc_beta_hard 0.5 --mtvc_tau_y 0.5 --mtvc_temp_pair 0.07 \
    --mtvc_case_layer "$CASE_LAYER" --mtvc_contrast_news_mode nbag \
    --contrast_temperature 0.07 --contrast_max_anchors 48 \
    --contrast_max_news_per_bag 6 --contrast_day_mode all \
    --dataset_root "$root" --vocab_path "$vocab" \
    "${extra[@]}" \
    --seed "$seed" --device cuda:0 --ckpt_dir "$ckpt" \
    >>"$ckpt/train.log" 2>&1 &
  echo $! >"$ckpt/train.pid"
  echo "  pid=$! ckpt=$ckpt"
}

# ---- queue ----
PENDING=()
for ds in "${DATASETS[@]}"; do
  for v in "${VARIANTS[@]}"; do
    for s in "${SEEDS[@]}"; do
      method="$(method_name "$ds" "$v")"
      ckpt="$REPRO/checkpoints/$ds/$method/s${s}"
      if job_done "$ckpt/train.log"; then
        continue
      fi
      PENDING+=("$ds|$v|$s")
    done
  done
done

MASTER="$REPRO/launch_logs/vin_connect_abl_3ds_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$MASTER")"
exec > >(tee -a "$MASTER") 2>&1
echo "[LAUNCH] master=$MASTER max_par=$MAX_PARALLEL case_layer=$CASE_LAYER"
echo "[LAUNCH] datasets=${DATASETS[*]} variants=${VARIANTS[*]} seeds=${SEEDS[*]} n_pending=${#PENDING[@]}"

declare -A RUN_PID=()
fail_holds=0

launch_available() {
  local item ds v seed need gpu
  while ((${#PENDING[@]} > 0)) && ((${#RUN_PID[@]} < MAX_PARALLEL)); do
    item="${PENDING[0]}"
    PENDING=("${PENDING[@]:1}")
    IFS='|' read -r ds v seed <<<"$item"
    need="$(min_free_for "$ds")"
    if ! gpu="$(pick_gpu "$need")"; then
      PENDING=("$item" "${PENDING[@]}")
      return 1
    fi
    start_one "$ds" "$v" "$seed" "$gpu" || true
    # track by ckpt path
    local method ckpt pid
    method="$(method_name "$ds" "$v")"
    ckpt="$REPRO/checkpoints/$ds/$method/s${seed}"
    if [[ -f "$ckpt/train.pid" ]]; then
      pid="$(cat "$ckpt/train.pid")"
      RUN_PID["$ckpt"]=$pid
    fi
    sleep 2
  done
  return 0
}

launch_available || true

while ((${#PENDING[@]} > 0 || ${#RUN_PID[@]} > 0)); do
  # reap
  for ckpt in "${!RUN_PID[@]}"; do
    pid="${RUN_PID[$ckpt]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      unset RUN_PID["$ckpt"]
      if job_done "$ckpt/train.log"; then
        echo "[done] $ckpt"
      else
        echo "[fail] $ckpt (see train.log)"
        fail_holds=$((fail_holds + 1))
      fi
    fi
  done
  launch_available || true
  echo "[poll] $(date '+%F %T') pending=${#PENDING[@]} running=${#RUN_PID[@]} fail_holds=$fail_holds"
  sleep "$POLL_SEC"
done

echo "[LAUNCH] finished fail_holds=$fail_holds"
