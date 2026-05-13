#!/bin/bash
# Run the full XSAKE benchmark suite
set -euo pipefail

OUT=benchmarks/results
mkdir -p $OUT

echo "=== XSAKE Benchmark Suite ==="
echo "Results → $OUT"
echo ""

echo "[1/4] Attention latency benchmark..."
python -m benchmarks.attention_benchmark \
  --seq_lens 512 1024 2048 4096 \
  --out $OUT/attention_results.json

echo "[2/4] Memory benchmark..."
python -m benchmarks.memory_benchmark \
  --seq_lens 512 1024 2048 4096 \
  --out $OUT/memory_results.json

echo "[3/4] Latency curves..."
python -m benchmarks.latency_benchmark \
  --seq_lens 512 1024 2048 4096 \
  --batch_sizes 1 2 4 \
  --out_json $OUT/latency_results.json \
  --out_plot $OUT/latency_curves.png

echo "[4/4] Throughput scaling..."
python -m benchmarks.throughput_benchmark \
  --seq_len 1024 --batch_size 8 \
  --device_counts 1 2 4 8 \
  --out_json $OUT/throughput_results.json \
  --out_plot $OUT/scaling_efficiency.png

echo ""
echo "=== All benchmarks complete ==="
ls -lh $OUT/
