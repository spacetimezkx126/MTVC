#!/usr/bin/env bash
# Wait for launch_abl_casel6_3ds.sh, then start launch_arch_casel6_3ds.sh
# (multinews → crossattn → fullunified).
set -u
REPRO="$(cd "$(dirname "$0")/.." && pwd)"
DUAL="$(cd "$REPRO/.." && pwd)"
LOGDIR="$REPRO/launch_logs"
mkdir -p "$LOGDIR"
WAIT_LOG="$LOGDIR/wait_abl_then_arch_casel6_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$WAIT_LOG") 2>&1

echo "[WAIT] started $(date '+%F %T') log=$WAIT_LOG"
POLL="${POLL_SEC:-60}"

abl_launcher_alive() {
  pgrep -af 'launch_abl_casel6_3ds\.sh' | grep -v grep >/dev/null 2>&1
}

abl_workers_alive() {
  pgrep -af 'run_mtvc\.py .*/abl_.*casel6' | grep -v grep >/dev/null 2>&1
}

while abl_launcher_alive || abl_workers_alive; do
  n_live=$(pgrep -af 'run_mtvc\.py .*/abl_.*casel6' | grep -v grep | wc -l | tr -d ' ')
  echo "[WAIT] abl still going launcher=$(abl_launcher_alive && echo yes || echo no) workers=$n_live $(date '+%F %T')"
  sleep "$POLL"
done

echo "[WAIT] abl finished $(date '+%F %T'); launching arch (multinews→crossattn→fullunified)"
cd "$DUAL"
nohup bash "$REPRO/scripts/launch_arch_casel6_3ds.sh" \
  > "$LOGDIR/nohup_arch_casel6_3ds.out" 2>&1 &
echo "[WAIT] next_launcher_pid=$! out=$LOGDIR/nohup_arch_casel6_3ds.out"
