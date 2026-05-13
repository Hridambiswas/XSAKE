#!/bin/bash
# HADS ablation study: sparsity ratio vs perplexity vs latency
set -euo pipefail

OUT=benchmarks/results/ablation
mkdir -p $OUT

echo "=== HADS Ablation Study ==="

for SEQ in 512 1024 2048; do
  echo "Running seq_len=$SEQ..."
  python -m benchmarks.hads_benchmark \
    --seq_len $SEQ \
    --out_json $OUT/hads_seq${SEQ}.json \
    --out_plot $OUT/memory_savings_seq${SEQ}.png
done

echo ""
echo "HADS ablation complete. Results:"
ls -lh $OUT/
