# Engineering Lessons Learned

## 1. Measure before you sparsify
Uniform sparsity destroys semantic heads. HADS exists because we learned
this the hard way. Calibrate head entropy first, then design sparsity.

## 2. Pallas > Triton for JAX-native systems
Pallas gives Triton-equivalent GPU performance with TPU support and CPU fallback.
If your stack is JAX, start with Pallas — not Triton.

## 3. Gradient checkpointing is non-negotiable at seq > 1024
At seq=2048 on 16GB VRAM, training fails without `nn.remat`. Enable it by default.

## 4. bfloat16 without master weights silently kills training
bf16 gradient underflow is invisible for ~50 steps, then grad norms → 0.
Always use the master-weight pattern: bf16 forward, fp32 optimizer update.

## 5. Enforce causal masking at BOTH token and block level
Block-level mask must respect causality independently of token-level mask.
Every mask generator must accept `causal: bool`.

## 6. Regression CI catches what code review misses
A deliberate bad commit (12% latency regression) was caught by CI in 5 minutes.
The threshold YAML + reporter is ~150 lines of code for enormous engineering safety.

## 7. Block size dominates at short sequences
At seq ≤ 1024, tuning block_size from 32→128 beats increasing sparsity ratio.
Larger blocks amortise launch overhead and improve memory access patterns.
