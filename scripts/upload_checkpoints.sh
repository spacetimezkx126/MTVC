#!/usr/bin/env bash
set -euo pipefail
SOURCE="${1:-../MTVC_checkpoints}"
DEFAULT_DESTINATION="gdrive:MTVC/verified-checkpoints"
if [[ "$(basename "$SOURCE")" == "MTVC_checkpoints_retrained" ]]; then
  DEFAULT_DESTINATION="gdrive:MTVC/retrained-checkpoints"
fi
DESTINATION="${2:-$DEFAULT_DESTINATION}"
rclone copy "$SOURCE" "$DESTINATION" --progress
rclone check "$SOURCE" "$DESTINATION" --one-way
