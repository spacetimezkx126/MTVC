#!/usr/bin/env bash
# Clean re-run of contrast baselines (NO cmin, NO volume).
# Datasets: CSMD50 / CSMD300 / MASSIVE (contiguous).
# MASSIVE train only: oversample extra jsonl. Seeds 42-46.
# Checkpoint / early-stop: highest val Acc+MCC (--best_metric val_acc_mcc).
set -u

ROOT="/home/zhaokx/Pattern/Pattern_Mining/dual_tf/contrast_exp_novol_s42_46"
PYTHON="/home/zhaokx/miniconda3/envs/CAMEF/bin/python"
OS_EXTRA="/home/zhaokx/Pattern/Pattern_Mining/dual_tf/shared_splits/massive_contiguous_match_test_posrate/train_oversample_extra.jsonl"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOGDIR="$ROOT/ablation_logs/novol_accmcc_s42_46"
CKPT_ROOT="$ROOT/checkpoints/novol_accmcc_s42_46"
PIDFILE="$LOGDIR/pids.txt"
BATCHLOG="$LOGDIR/batch.log"
LOCKFILE="$LOGDIR/batch.lock"
mkdir -p "$LOGDIR" "$CKPT_ROOT"

exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "[$(date -Iseconds)] another batch orchestrator is running (lock=$LOCKFILE)" >&2
  exit 1
fi

: > "$PIDFILE"

SEEDS=(42 43 44 45 46)
DATASETS=(
  "CSMD50|/home/zhaokx/Pattern/Pattern_Mining/dataset/CSMD50"
  "CSMD300|/home/zhaokx/Pattern/Pattern_Mining/dataset/CSMD300"
  "MASSIVE|/home/zhaokx/Pattern/Pattern_Mining/dataset/massive_data"
)

PREFERRED_GPUS=(0 1)
HEADROOM_MIB="${HEADROOM_MIB:-1800}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-16}"
MAX_PARALLEL="${MAX_PARALLEL:-32}"
POLL_SEC="${POLL_SEC:-4}"
LAUNCH_GAP_SEC="${LAUNCH_GAP_SEC:-1}"

exec > >(tee -a "$BATCHLOG") 2>&1 9>&-
echo "[$(date -Iseconds)] === novol + val_acc_mcc seeds 42-46, no cmin, massive oversample ==="

gpu_mem_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '
}

# Reserved MiB by model × dataset. Sized so both 24GB cards stay full without a first-wave OOM.
job_mem_mib() {
  local model=$1 ds=$2
  local base
  case "$model" in
    dtml) base=2800 ;;
    pen|stocknet) base=2000 ;;
    bi_lstm|adv_alstm) base=1400 ;;
    *) base=1000 ;;
  esac
  case "$ds" in
    CSMD300) echo $(( base + 600 )) ;;
    MASSIVE) echo $(( base + 900 )) ;;
    *) echo "$base" ;;
  esac
}

declare -A GPU_NJOBS=()
declare -A GPU_RESERVED=()
for g in "${PREFERRED_GPUS[@]}"; do
  GPU_NJOBS[$g]=0
  GPU_RESERVED[$g]=0
done

gpu_slot_ok() {
  local g=$1 need=$2 free_mib nrun reserved
  free_mib="$(gpu_mem_free_mib "$g")"
  nrun="${GPU_NJOBS[$g]:-0}"
  reserved="${GPU_RESERVED[$g]:-0}"
  [[ -n "$free_mib" ]] || return 1
  (( nrun < MAX_JOBS_PER_GPU )) || return 1
  (( free_mib >= need + HEADROOM_MIB ))
}

pick_gpu() {
  local need=$1
  local best="" best_n=999999 best_free=-1 g free_mib nrun
  for g in "${PREFERRED_GPUS[@]}"; do
    gpu_slot_ok "$g" "$need" || continue
    nrun="${GPU_NJOBS[$g]:-0}"
    free_mib="$(gpu_mem_free_mib "$g")"
    if (( nrun < best_n )) || { (( nrun == best_n )) && (( free_mib > best_free )); }; then
      best="$g"
      best_n=$nrun
      best_free=$free_mib
    fi
  done
  [[ -n "$best" ]] && echo "$best"
}

JOB_LINES=()
for ds_line in "${DATASETS[@]}"; do
  IFS='|' read -r ds_name ds_root <<< "$ds_line"
  for seed in "${SEEDS[@]}"; do
    for model in lstm alstm bi_lstm adv_alstm dtml; do
      case "$model" in
        bi_lstm) epochs=30 ;;
        dtml)    epochs=300 ;;
        *)       epochs=10 ;;
      esac
      JOB_LINES+=("single|${model}|${ds_name}|${ds_root}|${seed}|${epochs}")
    done
    JOB_LINES+=("pen|pen|${ds_name}|${ds_root}|${seed}|10")
    JOB_LINES+=("stocknet|stocknet|${ds_name}|${ds_root}|${seed}|10")
  done
done

job_tag() {
  local kind=$1 model=$2
  if [[ "$kind" == "single" ]]; then echo "$model"; else echo "$kind"; fi
}

job_log_path() {
  local kind=$1 model=$2 ds_name=$3 seed=$4
  echo "$LOGDIR/$(job_tag "$kind" "$model")_${ds_name,,}_seed${seed}.log"
}

job_ckpt_dir() {
  local kind=$1 model=$2 ds_name=$3 seed=$4
  echo "$CKPT_ROOT/$(job_tag "$kind" "$model")_${ds_name,,}_seed${seed}"
}

job_done() {
  local log=$1 kind=$2
  [[ -f "$log" ]] || return 1
  case "$kind" in
    single)   grep -q '\[FINAL\] best_epoch=' "$log" ;;
    pen)      grep -q 'done best_epoch=' "$log" ;;
    stocknet) grep -q '\[FINAL TEST @ best epoch' "$log" ;;
    *)        return 1 ;;
  esac
}

job_running() {
  local ckpt=$1
  pgrep -af "main_(single|pen|stocknet)\\.py" 2>/dev/null | grep -F -- "$ckpt" >/dev/null
}

write_reproduce_info() {
  local kind=$1 model=$2 ds_name=$3 ds_root=$4 seed=$5 epochs=$6 gpu=$7 ckpt=$8 log=$9 ec=${10} cmd=${11}
  [[ "$ec" -eq 0 ]] || return 0
  "$PYTHON" - "$kind" "$model" "$ds_name" "$ds_root" "$seed" "$epochs" "$gpu" "$ckpt" "$log" "$cmd" <<'PY'
import json, sys, datetime
from pathlib import Path
kind, model, ds_name, ds_root, seed, epochs, gpu, ckpt, log, cmd = sys.argv[1:]
ckpt = Path(ckpt)
best_files = sorted(ckpt.rglob("best_*.pt"))
info = {
    "experiment": "contrast_novol_accmcc_s42_46",
    "kind": kind,
    "method": model if kind == "single" else kind,
    "model": model or None,
    "dataset": ds_name,
    "dataset_root": ds_root,
    "seed": int(seed),
    "epochs": int(epochs),
    "use_volume": False,
    "best_metric": "val_acc_mcc",
    "split_mode": "contiguous",
    "device": f"cuda:{gpu}",
    "log_path": log,
    "save_dir": str(ckpt),
    "best_checkpoints": [str(p) for p in best_files],
    "reproduce_cmd": cmd,
    "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
}
(ckpt / "reproduce_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
PY
}

launch_job() {
  local kind=$1 model=$2 ds_name=$3 ds_root=$4 seed=$5 epochs=$6 gpu=$7
  local log ckpt_dir os_args=()
  log="$(job_log_path "$kind" "$model" "$ds_name" "$seed")"
  ckpt_dir="$(job_ckpt_dir "$kind" "$model" "$ds_name" "$seed")"
  mkdir -p "$ckpt_dir"
  if [[ "$ds_name" == "MASSIVE" ]]; then
    os_args=(--train_oversample_extra "$OS_EXTRA")
  fi
  echo "[$(date -Iseconds)] [LAUNCH] $(job_tag "$kind" "$model") ${ds_name} seed=${seed} epochs=${epochs} cuda:${gpu}" >&2
  : > "$log"

  local cmd
  if [[ "$kind" == "single" ]]; then
    cmd=("$PYTHON" -u main_single.py
      --model "$model"
      --dataset_root "$ds_root"
      --mode train
      --seed "$seed"
      --epochs "$epochs"
      --val_every 1
      --device "cuda:${gpu}"
      --save_dir "$ckpt_dir"
      --log_file "$log"
      --log_every 5
      --eval_test_each_epoch
      --input_size 3
      --best_metric val_acc_mcc
      --split_mode contiguous
      "${os_args[@]}")
  elif [[ "$kind" == "pen" ]]; then
    cmd=("$PYTHON" -u main_pen.py
      --dataset_root "$ds_root"
      --mode train
      --seed "$seed"
      --epochs "$epochs"
      --device "cuda:${gpu}"
      --save_dir "$ckpt_dir"
      --log_file "$log"
      --log_every 5
      --price_input_size 3
      --best_metric val_acc_mcc
      --split_mode contiguous
      "${os_args[@]}")
  else
    cmd=("$PYTHON" -u main_stocknet.py
      --dataset_root "$ds_root"
      --mode train
      --seed "$seed"
      --epochs "$epochs"
      --device "cuda:${gpu}"
      --save_dir "$ckpt_dir"
      --log_file "$log"
      --log_every 5
      --eval_test_each_epoch
      --price_input_size 3
      --best_metric val_acc_mcc
      --split_mode contiguous
      "${os_args[@]}")
  fi
  nohup "${cmd[@]}" 9>&- >> "$log" 2>&1 &
  LAUNCH_PID=$!
  LAUNCH_CMD="$(printf '%q ' "${cmd[@]}")"
  echo "$(job_tag "$kind" "$model")_${ds_name}_seed${seed} ${LAUNCH_PID} cuda:${gpu} ckpt=${ckpt_dir} log=${log}" >> "$PIDFILE"
}

total=${#JOB_LINES[@]}
idx=0
fail=0
declare -a ACTIVE=() ACTIVE_KIND=() ACTIVE_MODEL=() ACTIVE_DS=() ACTIVE_DSROOT=()
declare -a ACTIVE_SEED=() ACTIVE_EPOCHS=() ACTIVE_GPU=() ACTIVE_CKPT=() ACTIVE_LOG=() ACTIVE_MEM=() ACTIVE_CMD=()

reap_done() {
  local -a alive=() akind=() amodel=() ads=() adsroot=() aseed=() aepochs=() agpu=() ackpt=() alog=() amem=() acmd=()
  local pid ec i g mem
  for i in "${!ACTIVE[@]}"; do
    pid="${ACTIVE[$i]}"
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$pid"); akind+=("${ACTIVE_KIND[$i]}"); amodel+=("${ACTIVE_MODEL[$i]}")
      ads+=("${ACTIVE_DS[$i]}"); adsroot+=("${ACTIVE_DSROOT[$i]}"); aseed+=("${ACTIVE_SEED[$i]}")
      aepochs+=("${ACTIVE_EPOCHS[$i]}"); agpu+=("${ACTIVE_GPU[$i]}"); ackpt+=("${ACTIVE_CKPT[$i]}")
      alog+=("${ACTIVE_LOG[$i]}"); amem+=("${ACTIVE_MEM[$i]}"); acmd+=("${ACTIVE_CMD[$i]}")
      continue
    fi
    if wait "$pid" 2>/dev/null; then ec=0; else ec=$?; fi
    g="${ACTIVE_GPU[$i]}"
    mem="${ACTIVE_MEM[$i]}"
    GPU_NJOBS[$g]=$(( ${GPU_NJOBS[$g]:-0} - 1 ))
    GPU_RESERVED[$g]=$(( ${GPU_RESERVED[$g]:-0} - mem ))
    (( GPU_NJOBS[$g] < 0 )) && GPU_NJOBS[$g]=0
    (( GPU_RESERVED[$g] < 0 )) && GPU_RESERVED[$g]=0
    if (( ec != 0 )); then fail=$((fail + 1)); fi
    write_reproduce_info \
      "${ACTIVE_KIND[$i]}" "${ACTIVE_MODEL[$i]}" "${ACTIVE_DS[$i]}" "${ACTIVE_DSROOT[$i]}" \
      "${ACTIVE_SEED[$i]}" "${ACTIVE_EPOCHS[$i]}" "${ACTIVE_GPU[$i]}" \
      "${ACTIVE_CKPT[$i]}" "${ACTIVE_LOG[$i]}" "$ec" "${ACTIVE_CMD[$i]}" || true
    echo "[$(date -Iseconds)] [DONE] pid=${pid} exit=${ec} ckpt=${ACTIVE_CKPT[$i]}"
  done
  ACTIVE=("${alive[@]+"${alive[@]}"}")
  ACTIVE_KIND=("${akind[@]+"${akind[@]}"}")
  ACTIVE_MODEL=("${amodel[@]+"${amodel[@]}"}")
  ACTIVE_DS=("${ads[@]+"${ads[@]}"}")
  ACTIVE_DSROOT=("${adsroot[@]+"${adsroot[@]}"}")
  ACTIVE_SEED=("${aseed[@]+"${aseed[@]}"}")
  ACTIVE_EPOCHS=("${aepochs[@]+"${aepochs[@]}"}")
  ACTIVE_GPU=("${agpu[@]+"${agpu[@]}"}")
  ACTIVE_CKPT=("${ackpt[@]+"${ackpt[@]}"}")
  ACTIVE_LOG=("${alog[@]+"${alog[@]}"}")
  ACTIVE_MEM=("${amem[@]+"${amem[@]}"}")
  ACTIVE_CMD=("${acmd[@]+"${acmd[@]}"}")
}

echo "[$(date -Iseconds)] total jobs=${total} (7 methods × 3 datasets × 5 seeds)"
echo "no volume; best_metric=val_acc_mcc; massive oversample=$OS_EXTRA"
echo "epochs: bi_lstm=30 dtml=300 others=10"
echo "ckpt=${CKPT_ROOT} log=${LOGDIR}"
echo "parallel max=${MAX_PARALLEL} per_gpu=${MAX_JOBS_PER_GPU} headroom=${HEADROOM_MIB}MiB"

LAUNCH_PID=""
LAUNCH_CMD=""

while (( idx < total )) || (( ${#ACTIVE[@]} > 0 )); do
  reap_done
  while (( idx < total )) && (( ${#ACTIVE[@]} < MAX_PARALLEL )); do
    IFS='|' read -r kind model ds_name ds_root seed epochs <<< "${JOB_LINES[$idx]}"
    log="$(job_log_path "$kind" "$model" "$ds_name" "$seed")"
    ckpt_dir="$(job_ckpt_dir "$kind" "$model" "$ds_name" "$seed")"
    if job_done "$log" "$kind"; then
      echo "[$(date -Iseconds)] [SKIP] already done: $(basename "$log")"
      idx=$((idx + 1))
      continue
    fi
    if job_running "$ckpt_dir"; then
      echo "[$(date -Iseconds)] [SKIP] already running: $(basename "$log")"
      idx=$((idx + 1))
      continue
    fi
    need="$(job_mem_mib "$model" "$ds_name")"
    gpu="$(pick_gpu "$need" || true)"
    if [[ -z "${gpu:-}" ]]; then
      echo "[$(date -Iseconds)] no slot for $(job_tag "$kind" "$model") ${ds_name} need=${need}MiB active=${#ACTIVE[@]} reserved0=${GPU_RESERVED[0]:-0} reserved1=${GPU_RESERVED[1]:-0}"
      break
    fi
    idx=$((idx + 1))
    launch_job "$kind" "$model" "$ds_name" "$ds_root" "$seed" "$epochs" "$gpu"
    GPU_NJOBS[$gpu]=$(( ${GPU_NJOBS[$gpu]:-0} + 1 ))
    GPU_RESERVED[$gpu]=$(( ${GPU_RESERVED[$gpu]:-0} + need ))
    ACTIVE+=("$LAUNCH_PID")
    ACTIVE_KIND+=("$kind")
    ACTIVE_MODEL+=("$model")
    ACTIVE_DS+=("$ds_name")
    ACTIVE_DSROOT+=("$ds_root")
    ACTIVE_SEED+=("$seed")
    ACTIVE_EPOCHS+=("$epochs")
    ACTIVE_GPU+=("$gpu")
    ACTIVE_CKPT+=("$ckpt_dir")
    ACTIVE_LOG+=("$log")
    ACTIVE_MEM+=("$need")
    ACTIVE_CMD+=("$LAUNCH_CMD")
    sleep "$LAUNCH_GAP_SEC"
  done
  if (( idx < total )) || (( ${#ACTIVE[@]} > 0 )); then
    sleep "$POLL_SEC"
  fi
done

echo "[$(date -Iseconds)] === batch finished jobs=${total} failed=${fail} ==="
