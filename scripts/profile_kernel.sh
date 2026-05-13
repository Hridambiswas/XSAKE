#!/bin/bash
# Launch XLA kernel profiler for the sparse attention kernel
set -euo pipefail

PROFILE_DIR=${PROFILE_DIR:-/tmp/xsake_profile}
SEQ_LEN=${SEQ_LEN:-1024}

echo "=== XSAKE Kernel Profiler ==="
echo "Profile dir: $PROFILE_DIR"
echo "Seq length:  $SEQ_LEN"
echo ""

python -c "
import jax
import jax.numpy as jnp
from observability.kernel_profiler import xla_trace, time_kernel
from kernels.pallas.sparse_attention import sparse_attention_reference
from kernels.hads.hads_pattern import dummy_hads_profile

key = jax.random.PRNGKey(0)
B, H, S, D = 1, 12, $SEQ_LEN, 64
q = jax.random.normal(key, (B, H, S, D), jnp.bfloat16)
k = jax.random.normal(key, (B, H, S, D), jnp.bfloat16)
v = jax.random.normal(key, (B, H, S, D), jnp.bfloat16)

profile = dummy_hads_profile(n_heads=H, seq_len=S, block_size=128)

import jax
fn = jax.jit(lambda q,k,v: sparse_attention_reference(q, k, v, profile.block_masks, 128)[0])

print('Warming up...')
jax.block_until_ready(fn(q, k, v))

with xla_trace('$PROFILE_DIR'):
    for _ in range(10):
        jax.block_until_ready(fn(q, k, v))

timing = time_kernel(fn, q, k, v)
print(f'Median latency: {timing[\"median_ms\"]:.2f} ms ± {timing[\"std_ms\"]:.2f} ms')
print(f'Open tensorboard: tensorboard --logdir $PROFILE_DIR')
"
