#!/bin/bash
# Fast GLM-5-FP8 profiling benchmark for AMD MI355X with TP=8.
# Uses --disable-cuda-graph to skip long graph compilation during baseline/profile work.
# Output streams in real-time for background monitoring.
set -euo pipefail

LOGFILE=$(mktemp /tmp/bench_fast_XXXXXX.log)

PYTHONUNBUFFERED=1 /opt/venv/bin/python3 -u -m sglang.bench_one_batch \
    --model-path zai-org/GLM-5-FP8 \
    --tensor-parallel-size 8 \
    --batch-size 1 \
    --input-len 1024 \
    --output-len 128 \
    --dtype bfloat16 \
    --quantization fp8 \
    --mem-fraction-static 0.9 \
    --disable-cuda-graph \
    2>&1 | tee "$LOGFILE" || true

DECODE_SEC=$(grep -oP 'Decode\.\s+median latency:\s+\K[\d.]+' "$LOGFILE" | tail -1)
rm -f "$LOGFILE"

if [ -n "$DECODE_SEC" ]; then
    DECODE_MS=$(/opt/venv/bin/python3 -c "print(f'{float(\"$DECODE_SEC\") * 1000:.1f}')")
    echo "Decode median (ms): $DECODE_MS | tp=8 batch=1"
else
    echo "ERROR: Could not extract decode median from benchmark output" >&2
    exit 1
fi
