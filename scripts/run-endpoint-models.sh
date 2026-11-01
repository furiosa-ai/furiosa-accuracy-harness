#!/usr/bin/env bash
# Full suite on every chat model of the public RNGD endpoint, one model per tmux session (all in parallel).
# Benches run short-first so partial results land early; MMLU-Pro (12K items) is last.
#   bash scripts/run-endpoint-models.sh            # CONC=8 per model (4 x 8 = 32 in flight on the shared endpoint)
#   tmux ls ; tmux attach -t fah-gpt-oss-120b      # watch;  logs also in results/<model>/std/run.log
set -euo pipefail
cd "$(dirname "$0")/.."
FAH=${FAH:-"uv run fah"}
CONC=${CONC:-8}
CAF=$(command -v caffeinate >/dev/null && echo "caffeinate -i -s" || true)   # macOS: keep the Mac awake
BENCHES=${BENCHES:-ifeval,gpqa_diamond,aime_2025,livecodebench,mmlu_pro}
MODELS=${MODELS:-"furiosa-ai/gpt-oss-120b furiosa-ai/Qwen3-32B-FP8 furiosa-ai/Qwen3-VL-32B-Instruct furiosa-ai/gemma-4-31B-it-FP8-block"}
for m in $MODELS; do
  name=fah-$(basename "$m" | tr '.' '-')
  log=results/${m//\//__}/std/run.log
  mkdir -p "$(dirname "$log")"
  tmux new-session -d -s "$name" \
    "$CAF $FAH run --model $m --bench $BENCHES --concurrency $CONC ${EXTRA:-} 2>&1 | tee -a $log; echo; echo 'exit '\$?; read"
  echo "started $name -> $log"
done
