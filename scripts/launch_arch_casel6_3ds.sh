#!/usr/bin/env bash
# Architecture / contrast-news ablations @ L6:
#   multinews_casel6 / crossattn_casel6 / fullunified_casel6
# Datasets: csmd50, csmd300, massive.
#
# Phases (strict):
#   1) multinews  — partial_unified + --mtvc_contrast_news_mode multinews
#   2) crossattn  — cross_attn  (nbag L_pair)
#   3) fullunified — full_unified (nbag L_pair)
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
# Override e.g. PHASES="multinews" or PHASES="crossattn fullunified"
PHASES=(${PHASES:-multinews crossattn fullunified})
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
  local ds="$1" phase="$2"
  case "$phase" in
    multinews)
      case "$ds" in
        csmd50) echo 3200 ;;
        csmd300) echo 3500 ;;
        massive) echo 4500 ;;
        *) echo 3500 ;;
      esac
      ;;
    crossattn|fullunified)
      # heavier encoders
      case "$ds" in
        csmd50) echo 5500 ;;
        csmd300) echo 8000 ;;
        massive) echo 11000 ;;
        *) echo 8000 ;;
      esac
      ;;
    *) echo 4000 ;;
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
  local ds="$1" phase="$2"
  # Formal: multinews_casel6 / crossattn_casel6 / fullunified_casel6
  # Optional METHOD_TAG_SUFFIX (e.g. random) only for one-off probes
  local suf="${METHOD_TAG_SUFFIX:-}"
  [[ -n "$suf" ]] && suf="_${suf}"
  echo "${phase}${suf}_casel${CASE_LAYER}"
}

resolve_phase() {
  local phase="$1"
  ENC=partial_unified
  NEWS_MODE=nbag
  case "$phase" in
    multinews)
      ENC=partial_unified
      NEWS_MODE=multinews
      ;;
    crossattn)
      ENC=cross_attn
      NEWS_MODE=nbag
      ;;
    fullunified)
      ENC=full_unified
      NEWS_MODE=nbag
      ;;
    *)
      echo "[err] unknown phase $phase" >&2
      return 1
      ;;
  esac
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
  local ckpt="$1" ds="$2" phase="$3" seed="$4"
  resolve_phase "$phase" || return 2
  mkdir -p "$ckpt"
  if job_done "$ckpt/train.log"; then
    echo "[skip] done $ckpt"
    return 3
  fi
  if [[ -f "$ckpt/train.pid" ]] && kill -0 "$(cat "$ckpt/train.pid")" 2>/dev/null; then
    echo "[skip] running $ckpt"
    return 4
  fi
  if [[ -s "$ckpt/train.log" ]] && grep -qiE "Traceback|CUDA_FATAL|OutOfMemoryError" "$ckpt/train.log"; then
    if [[ -f "$ckpt/train.log.crash" ]]; then
      echo "[fail-hold] $ckpt crashed again"
      return 2
    fi
    echo "[retry] archive crash log $ckpt"
    mv "$ckpt/train.log" "$ckpt/train.log.crash"
  fi

  local need gpu MLP
  need=$(min_free_for "$ds" "$phase")
  gpu=$(pick_gpu "$need") || return 1
  MLP=full
  [[ "$ds" == "massive" ]] && MLP=no_global

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
  echo "[start] $(date '+%F %T') ${ds} ${phase} s${seed} enc=$ENC news_mode=$NEWS_MODE gpu=$gpu -> $(basename "$(dirname "$ckpt")")"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u "$RUN" -d "$ds" \
    --mode train --emb_dim 128 --news_emb_dim 32 \
    --epochs 50 --es_start_epoch 1 --es_patience 5 --es_metric acc_mcc \
    --split_mode contiguous --bce_pos_weight 1.0 \
    --news_word_emb random --news_text_tf_layers 3 --news_padding_k 5 \
    --industry_fuse_site pred_fusion \
    --price_encoder "$ENC" \
    --pred_fusion_ablation_mode "$MLP" --news_ablation_mode full \
    --contrast_aux_weight 1.0 --contrast_mode mtvc \
    --sim_quantile 0.75 \
    --mtvc_lambda_pair 0.2 --mtvc_beta_hard 0.5 --mtvc_tau_y 0.5 --mtvc_temp_pair 0.07 \
    --mtvc_case_layer "$CASE_LAYER" \
    --mtvc_contrast_news_mode "$NEWS_MODE" \
    --contrast_temperature 0.07 \
    --contrast_max_anchors 48 --contrast_max_news_per_bag 6 --contrast_day_mode all \
    "${ds_extra[@]}" \
    --seed "$seed" --device cuda:0 --ckpt_dir "$ckpt" \
    >"$ckpt/train.log" 2>&1 &
  echo $! > "$ckpt/train.pid"
  echo "  pid=$! ckpt=$ckpt"
  sleep 8
  return 0
}

run_phase() {
  local phase="$1"
  PENDING=()
  local ds seed method ckpt
  for ds in "${DATASETS[@]}"; do
    method=$(method_name "$ds" "$phase")
    for seed in "${SEEDS[@]}"; do
      ckpt="$REPRO/checkpoints/${ds}/${method}/s${seed}"
      PENDING+=("${ckpt}|${ds}|${phase}|${seed}")
    done
  done
  echo "[PHASE] start phase=$phase jobs=${#PENDING[@]} $(date '+%F %T')"
  FAIL_HOLD=()
  while true; do
    NEW_PENDING=()
    for item in "${PENDING[@]}"; do
      ckpt="${item%%|*}"
      if job_done "$ckpt/train.log"; then
        echo "[done] $ckpt"
        continue
      fi
      # permanently skip fail-holds so later phases can start
      skip=0
      for fh in "${FAIL_HOLD[@]:-}"; do
        [[ "$fh" == "$ckpt" ]] && skip=1 && break
      done
      if (( skip )); then
        continue
      fi
      NEW_PENDING+=("$item")
    done
    PENDING=("${NEW_PENDING[@]:-}")
    if (( ${#PENDING[@]} == 0 )); then
      n_fail=${#FAIL_HOLD[@]}
      if (( n_fail > 0 )); then
        echo "[PHASE] finished phase=$phase with fail-holds=${n_fail} $(date '+%F %T')"
        for fh in "${FAIL_HOLD[@]}"; do echo "  [fail-skip] $fh"; done
      else
        echo "[PHASE] finished phase=$phase $(date '+%F %T')"
      fi
      break
    fi
    running=$(count_live)
    if (( running < MAX_PARALLEL )); then
      for item in "${PENDING[@]}"; do
        ckpt="${item%%|*}"
        rest="${item#*|}"
        ds="${rest%%|*}"; rest="${rest#*|}"
        phase_i="${rest%%|*}"; seed="${rest##*|}"
        if [[ -f "$ckpt/train.pid" ]] && kill -0 "$(cat "$ckpt/train.pid")" 2>/dev/null; then
          continue
        fi
        if job_done "$ckpt/train.log"; then
          continue
        fi
        if (( running >= MAX_PARALLEL )); then
          break
        fi
        launch_one "$ckpt" "$ds" "$phase_i" "$seed"
        rc=$?
        if [[ "$rc" == "0" ]]; then
          running=$((running + 1))
        elif [[ "$rc" == "1" ]]; then
          echo "[wait] no GPU free enough (need for $ds/$phase_i)"
          break
        elif [[ "$rc" == "2" ]]; then
          echo "[fail-skip] dropping $ckpt from queue"
          FAIL_HOLD+=("$ckpt")
        fi
      done
    fi
    n_fail=${#FAIL_HOLD[@]}
    echo "[poll] $(date '+%F %T') phase=$phase pending=${#PENDING[@]} running=$(count_live) fail_holds=${n_fail}"
    sleep "$POLL_SEC"
  done
}

LOGDIR="$REPRO/launch_logs"
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/multinews_cross_fulluni_casel6_sep13_3ds_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER") 2>&1
echo "[LAUNCH] master=$MASTER max_par=$MAX_PARALLEL case_layer=$CASE_LAYER"
echo "[LAUNCH] datasets=${DATASETS[*]} phases=${PHASES[*]} seeds=${SEEDS[*]}"

for ph in "${PHASES[@]}"; do
  resolve_phase "$ph" || exit 1
  run_phase "$ph"
done
echo "[LAUNCH] all phases finished $(date '+%F %T')"
