#!/usr/bin/env bash
# Example entry: retrain / eval via the formal MTVC package.
set -euo pipefail
REPRO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
# See scripts/launch_*_casel6_3ds.sh for full flag sets.
# Key flags: --contrast_mode mtvc --mtvc_case_layer 6 --mtvc_lambda_pair 0.2
exec "$PY" "$REPRO/code/run_mtvc.py" "$@"
