#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ok=0; miss=0
while IFS= read -r -d '' p; do
  if [[ -f "$p/best.pt" && -f "$p/best.pt.meta.json" ]]; then
    ok=$((ok+1))
  else
    echo "MISS $p"; miss=$((miss+1))
  fi
done < <(find -L "$ROOT/checkpoints" -type d -name 's4[2-6]' -print0)
echo "MTVC seed dirs ok=$ok miss=$miss"
base_ok=$(find -L "$ROOT/baselines/novol_accmcc_s42_46" -name 'best_*.pt' 2>/dev/null | wc -l)
echo "baseline best_*.pt count=$base_ok (expect 105)"
test -f "$ROOT/code/run_mtvc.py"
test -d "$ROOT/code/mtvc"
test -f "$ROOT/code/mtvc/contrast_aux.py"
test -f "$ROOT/code/mtvc/model.py"
test -d "$ROOT/code/mtvc/tool"
test -f "$ROOT/code/mtvc/tool/numeric.py"
test -f "$ROOT/code/mtvc/tool/cuda_guard.py"
test -f "$ROOT/code/mtvc/tool/oversample_utils.py"
test ! -e "$ROOT/code/mtvc/encode_tokens.py"
test ! -e "$ROOT/code/mtvc/partial_unified_transformer.py"
test ! -e "$ROOT/code/mtvc/model"
test ! -e "$ROOT/code/mtvc/encoder.py"
test ! -e "$ROOT/code/mtvc/partial_unified_backbone.py"
test ! -e "$ROOT/code/mtvc/dataset_mine.py"
test ! -e "$ROOT/code/mtvc/token_prune.py"
test ! -e "$ROOT/code/mtvc/cuda_guard.py"
test ! -e "$ROOT/code/mtvc/oversample_utils.py"
test ! -e "$ROOT/code/code_overlay"
test ! -e "$ROOT/code/support"
test ! -e "$ROOT/code/mtvc/news_halt_aux.py"
test ! -e "$ROOT/code/mtvc/method13_aux.py"
test ! -e "$ROOT/code/mtvc/pretrain_ssl.py"
test ! -e "$ROOT/code/mtvc/legacy"
echo "OK"
