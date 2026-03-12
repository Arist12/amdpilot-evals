#!/bin/bash
# GLM-5-FP8 decode latency benchmark for AMD MI355X with TP=8.
# Immutable parameters — do NOT change model, tp, batch, input/output lengths.
set -euo pipefail

OUTPUT=$(/opt/venv/bin/python3 -m sglang.bench_one_batch \
    --model-path zai-org/GLM-5-FP8 \
    --tensor-parallel-size 8 \
    --batch-size 1 \
    --input-len 1024 \
    --output-len 128 \
    --dtype bfloat16 \
    --quantization fp8 \
    --mem-fraction-static 0.9 \
    2>&1) || true

echo "$OUTPUT"

DECODE_SEC=$(echo "$OUTPUT" | grep -oP 'Decode\.\s+median latency:\s+\K[\d.]+' | tail -1)

if [ -n "$DECODE_SEC" ]; then
    DECODE_MS=$(/opt/venv/bin/python3 -c "print(f'{float(\"$DECODE_SEC\") * 1000:.1f}')")
    echo "Decode median (ms): $DECODE_MS | tp=8 batch=1"
else
    echo "ERROR: Could not extract decode median from benchmark output" >&2
    exit 1
fi
