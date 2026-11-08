#!/usr/bin/env bash
# Run the fix-the-failing-test benchmark against a local CPU model via Ollama.
#
#   bash run_local.sh [smoke|full|all] [model]
#
#   smoke  2 tasks, thinking off: checks everything works and shows time per call (default)
#   full   all 31 tasks x 2 arms, thinking off
#   all    full with thinking off, then again with thinking on
#   model  any Ollama model name; default hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M
#
# It starts a private Ollama server on 127.0.0.1:11435 with a 16K context window (the
# default context silently truncates our multi-turn prompts), so it does not touch an
# Ollama system service you may already run. Models go to this user's ~/.ollama.
# Results land in results/; commit them (git add results && git commit) to share.
set -euo pipefail

MODE="${1:-smoke}"
MODEL="${2:-hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M}"
PORT=11435
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

die() { echo "error: $*" >&2; exit 1; }

command -v python3 >/dev/null || die "python3 not found (sudo apt install python3)"
python3 - <<'PY' || die "Python 3.10+ is required"
import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY
command -v ollama >/dev/null || die "Ollama not found. Install it from https://ollama.com/download (Linux: the official install script), then re-run."
command -v ruff >/dev/null || echo "note: ruff not found; the linter specialist will report itself unavailable (pip install ruff to enable it)"

export OLLAMA_HOST="127.0.0.1:${PORT}"
export OLLAMA_CONTEXT_LENGTH=16384
LOG="$HERE/results/ollama-serve.log"
mkdir -p "$HERE/results"

echo "==> starting private Ollama server on ${OLLAMA_HOST} (log: ${LOG})"
ollama serve >"$LOG" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
    if python3 -c "import urllib.request; urllib.request.urlopen('http://${OLLAMA_HOST}/api/version', timeout=2)" 2>/dev/null; then
        break
    fi
    kill -0 "$SERVE_PID" 2>/dev/null || die "ollama serve exited; see $LOG"
    sleep 1
done

echo "==> pulling ${MODEL} (first time only; ~1-6 GB)"
ollama pull "$MODEL"

echo "==> offline harness check (oracle planner, ~10 s)"
python3 run_bench.py --planner oracle --tasks stats-median-even,bsearch-hang --workers 2 >/dev/null
rm -f results/oracle.json results/oracle.md

bench() {  # $1 = think mode, remaining args passed through
    local think="$1"; shift
    python3 run_bench.py --planner local --model "$MODEL" --base-url "http://${OLLAMA_HOST}/v1" \
        --think "$think" --workers 1 "$@"
}

case "$MODE" in
    smoke)
        echo "==> smoke test: 2 tasks x 2 arms, thinking off"
        bench off --tasks stats-median-even,bsearch-hang ;;
    full)
        echo "==> full run, thinking off (several hours on CPU is normal)"
        bench off ;;
    all)
        echo "==> full run, thinking off"
        bench off
        echo "==> full run, thinking on (much slower)"
        bench on ;;
    *) die "unknown mode '$MODE' (smoke|full|all)" ;;
esac

echo "==> done. Summary tables: $HERE/results/*.md"
