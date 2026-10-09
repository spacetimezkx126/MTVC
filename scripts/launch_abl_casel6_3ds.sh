#!/usr/bin/env bash
# Component ablations @ official sep13 stack, case-mine L6.
# Variants: noglobal|add_global / noindustry / nogate / nocontrast / noprice(no-contrast+no-gate).
# CSMD50/300 baseline MLP=full → ablate noglobal.
# Massive baseline MLP=no_global → ablate add_global (NOT noglobal).
# Datasets: csmd50, csmd300, massive → checkpoints/{ds}/abl_*_casel6/
set -u
REPRO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPRO"
export DUAL_TF_MODEL_DIR="$REPRO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MTVC_USE_TYPE_EMB=1 MTVC_UNIFIED_CAUSAL=1 MTVC_TRANSFORMER_LAYERS=6
export MTVC_SOFT_PRUNE=0 MTVC_UNIFIED_SAME_DAY=0 MTVC_SHARE_PRICE_NEWS_ENC=0
export DUAL_TF_CUDA_LOCK="${DUAL_TF_CUDA_LOCK:-1}"
unset MTVC_FIXED_TRAIN_ORDER MTVC_UNIFIED_LAYERS MTVC_HARD_FINETUNE MTVC_LOOP_ROUNDS 2>/dev/null || true

PY="${PYTHON:-python3}"
RUN="$REPRO/code/run_mtvc.py"
VOCAB_CSMD="$REPRO/dict/dict_csmd.pkl"
VOCAB_MASSIVE="$REPRO/dict/dict_massive.pkl"
OS_EXTRA="$REPRO/assets/massive_oversample/train_oversample_extra.jsonl"

DATASETS=(${DATASETS:-csmd50 csmd300 massive})
# Default list is expanded per-dataset in PENDING build (massive: add_global not noglobal).
VARIANTS_CSMD=(${VARIANTS_CSMD:-noglobal noindustry nogate nocontrast noprice})
VARIANTS_MASSIVE=(${VARIANTS_MASSIVE:-add_global noindustry nogate nocontrast noprice})
SEEDS=(${SEEDS:-42 43 44 45 46})
ALLOWED_GPUS=(${ALLOWED_GPUS:-0 1})
MAX_PARALLEL="${MAX_PARALLEL:-6}"
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
  # Formal names: abl_<variant>_casel6 (massive add_global → abl_addglobal_casel6)
  case "$v" in
    add_global) echo "abl_addglobal_casel${CASE_LAYER}" ;;
    *) echo "abl_${v}_casel${CASE_LAYER}" ;;
  esac
}

resolve_variant() {
  local ds="$1" v="$2"
  MLP=full
  NEWS_ABL=full
  NO_PRICE=0
  CONTRAST_W=1.0
  CONTRAST_MODE=mtvc
  if [[ "$ds" == "massive" ]]; then
    MLP=no_global
  fi
  case "$v" in
    noglobal)
      # only meaningful on CSMD (baseline has global)
      MLP=no_global
      ;;
    add_global)
      # only meaningful on Massive (baseline is no_global)
      MLP=full
      ;;
    noindustry)
      MLP=no_industry
      ;;
    nogate)
      # L_pair on, news gates off
      CONTRAST_MODE=mtvc_nogate
      ;;
    nocontrast)
      # no contrast aux, no gates (temporal)
      CONTRAST_W=0.0
      CONTRAST_MODE=temporal
      ;;
    noprice)
      # no contrast + no gate, drop price token
      CONTRAST_W=0.0
      CONTRAST_MODE=temporal
      NO_PRICE=1
      ;;
    *)
      echo "[err] unknown variant $v" >&2
      return 1
      ;;
  esac
  METHOD=$(method_name "$ds" "$v")
}

count_live() {
  local n=0 item
  for item in "${PENDING[@]}"; do
    local ckpt="${item%%|*}"
    if [[ -f "$ckpt/train.pid" ]] && kill -0 "$(cat "$ckpt/train.pid")" 2>/dev/null; then
      n=$((n + 1))
    fi
  done
  echo "$n"
}

launch_one() {
  local ckpt="$1" ds="$2" v="$3" seed="$4"
  resolve_variant "$ds" "$v" || return 2
  mkdir -p "$ckpt"
  if job_done "$ckpt/train.log"; then
    echo "[skip] done $ckpt"
    return 3
  fi
  if [[ -f "$ckpt/train.pid" ]] && kill -0 "$(cat "$ckpt/train.pid")" 2>/dev/null; then
    echo "[skip] running $ckpt"
    return 4
  fi
  local need gpu
  need=$(min_free_for "$ds")
  gpu=$(pick_gpu "$need") || return 1

  local -a extra=(
    --contrast_aux_weight "$CONTRAST_W" --contrast_mode "$CONTRAST_MODE"
    --pred_fusion_ablation_mode "$MLP"
    --news_ablation_mode "$NEWS_ABL"
    --sim_quantile 0.75
    --mtvc_lambda_pair 0.2 --mtvc_beta_hard 0.5 --mtvc_tau_y 0.5 --mtvc_temp_pair 0.07
    --mtvc_case_layer "$CASE_LAYER"
    --contrast_temperature 0.07
    --contrast_max_anchors 48 --contrast_max_news_per_bag 6 --contrast_day_mode all
  )
  if [[ "$NO_PRICE" == "1" ]]; then
    extra+=(--no_price_token)
  fi

  local -a ds_extra=()
  if [[ "$ds" == "massive" ]]; then
    ds_extra=(
      --dataset_root "${MASSIVE_ROOT:-$REPRO/data/massive_data}"
      --vocab_path "$VOCAB_MASSIVE"
      --train_oversample_extra "$OS_EXTRA"
      --train_start 2022-01-01 --train_end 2023-12-31
      --val_start 2024-01-01 --val_end 2024-12-31
      --test_start 2025-01-01 --test_end 2025-12-31
    )
  else
    local root="$REPRO/data/CSMD50"
    [[ "$ds" == "csmd300" ]] && root="${CSMD300_ROOT:-$REPRO/data/CSMD300}"
    ds_extra=(
      --dataset_root "$root"
      --vocab_path "$VOCAB_CSMD"
     
      --train_start 2021-01-01 --train_end 2023-01-01
      --val_start 2023-01-02 --val_end 2024-01-02
      --test_start 2024-01-03 --test_end 2024-12-31
    )
  fi

  : > "$ckpt/train.log"
  echo "[start] $(date '+%F %T') ${ds} ${v} s${seed} gpu=$gpu -> $METHOD"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u "$RUN" -d "$ds" \
    --mode train --emb_dim 128 --news_emb_dim 32 \
    --epochs 50 --es_start_epoch 1 --es_patience 5 --es_metric acc_mcc \
    --split_mode contiguous --bce_pos_weight 1.0 \
    --news_word_emb random --news_text_tf_layers 3 --news_padding_k 5 \
    --industry_fuse_site pred_fusion \
    --price_encoder partial_unified \
    "${ds_extra[@]}" \
    "${extra[@]}" \
    --seed "$seed" --device cuda:0 --ckpt_dir "$ckpt" \
    >"$ckpt/train.log" 2>&1 &
  echo $! > "$ckpt/train.pid"
  echo "  pid=$! ckpt=$ckpt"
  sleep 8
  return 0
}

PENDING=()
for ds in "${DATASETS[@]}"; do
  if [[ "$ds" == "massive" ]]; then
    _vars=("${VARIANTS_MASSIVE[@]}")
  else
    _vars=("${VARIANTS_CSMD[@]}")
  fi
  for v in "${_vars[@]}"; do
    resolve_variant "$ds" "$v" || exit 1
    for seed in "${SEEDS[@]}"; do
      ckpt="$REPRO/checkpoints/${ds}/${METHOD}/s${seed}"
      PENDING+=("${ckpt}|${ds}|${v}|${seed}")
    done
  done
done

LOGDIR="$REPRO/launch_logs"
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/abl_casel6_sep13_3ds_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER") 2>&1
echo "[LAUNCH] master=$MASTER jobs=${#PENDING[@]} max_par=$MAX_PARALLEL case_layer=$CASE_LAYER"
echo "[LAUNCH] datasets=${DATASETS[*]} csmd_vars=${VARIANTS_CSMD[*]} massive_vars=${VARIANTS_MASSIVE[*]} seeds=${SEEDS[*]}"

while true; do
  NEW_PENDING=()
  for item in "${PENDING[@]}"; do
    ckpt="${item%%|*}"
    if job_done "$ckpt/train.log"; then
      echo "[done] $ckpt"
      continue
    fi
    NEW_PENDING+=("$item")
  done
  PENDING=("${NEW_PENDING[@]:-}")
  if (( ${#PENDING[@]} == 0 )); then
    echo "[LAUNCH] all jobs finished"
    break
  fi
  running=$(count_live)
  if (( running < MAX_PARALLEL )); then
    for item in "${PENDING[@]}"; do
      ckpt="${item%%|*}"
      rest="${item#*|}"
      ds="${rest%%|*}"; rest="${rest#*|}"
      v="${rest%%|*}"; seed="${rest##*|}"
      if [[ -f "$ckpt/train.pid" ]] && kill -0 "$(cat "$ckpt/train.pid")" 2>/dev/null; then
        continue
      fi
      if job_done "$ckpt/train.log"; then
        continue
      fi
      if (( running >= MAX_PARALLEL )); then
        break
      fi
      launch_one "$ckpt" "$ds" "$v" "$seed"
      rc=$?
      if [[ "$rc" == "0" ]]; then
        running=$((running + 1))
      elif [[ "$rc" == "1" ]]; then
        echo "[wait] no GPU free enough"
        break
      fi
    done
  fi
  sleep "$POLL_SEC"
done
