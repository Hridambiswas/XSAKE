# XSAKE — Failure Log

Every major failure encountered during development, what broke, and exactly why.

---

## F-001: Static Sparsity Patterns Failed to Generalise

**When:** Early kernel design phase  
**Symptom:** Fixed sliding-window and random masks produced 8–12% higher perplexity than dense attention on OpenWebText, even at low sparsity ratios (20%).  
**Root cause:** A uniform sparsity structure applied to all heads ignores the fact that different heads develop different attention behaviours during pre-training. Syntactic heads (local) and semantic heads (global) cannot share the same sparsity budget without one suffering. See `abandoned_approaches/static_sparsity_attempt.md`.  
**Resolution:** Switched to HADS — per-head calibration-based sparsity. Perplexity gap closed to <3%.

---

## F-002: Triton Kernel Abandoned for Pallas

**When:** Initial kernel implementation attempt  
**Symptom:** Triton could not be installed on the development machine (Mac M4, no CUDA). CI would have needed CUDA runners (expensive).  
**Root cause:** Triton requires a CUDA toolkit and NVIDIA GPU. Mac M4 is Apple Silicon — no CUDA, Triton not available.  
**Resolution:** Switched to JAX Pallas. Pallas compiles to Triton on GPU (same underlying codegen), works on TPU natively, and allows CPU fallback for local testing. No functionality lost; correctness testing now works on CPU. See `abandoned_approaches/triton_first_attempt.md`.

---

## F-003: Naive pmap OOM at seq_len=2048

**When:** First distributed training attempt  
**Symptom:** OOM on T4 (16GB) with batch=16, seq=2048, 12 layers when using naive `jax.pmap` with replicated states.  
**Root cause:** pmap replicates the full model + optimizer state across devices. With fp32 master weights + bf16 activations + optimizer moments, the memory footprint exceeded 16GB. The dense attention score matrix `[batch, heads, seq, seq]` = 16×12×2048×2048×2 bytes alone is 1.6GB.  
**Resolution:** Three changes applied together:  
  1. Enabled gradient checkpointing (`nn.remat`) — recomputes activations, cuts activation memory ~4×  
  2. Enabled HADS sparsity — attention score matrix shrunk by ~45%  
  3. Reduced batch size to 8 per device  
  See `abandoned_approaches/naive_pmap_failure.md`.

---

## F-004: BlockSpec Dimension Mismatch in Pallas

**When:** First Pallas kernel execution on GPU  
**Symptom:** `ValueError: BlockSpec block_shape [1, 1, 128, 64] does not match input shape [2, 12, 1024, 64]`  
**Root cause:** The BlockSpec `index_map` returns block indices (not element indices), and the block_shape must match the shape that the kernel body sees — not the original array shape. Confusion between element-level indexing and block-level indexing in the Pallas API.  
**Resolution:** Added a 5D reshape before the pallas_call (introducing a `num_q_blocks` axis) so that the BlockSpec grid dimension maps cleanly to the Q block axis. K and V are passed with a singleton `1` in the q-block dimension, allowing full-sequence access inside the kernel loop.

---

## F-005: Causal Mask Not Applied at Block Boundary

**When:** Loss curve inspection after 500 steps  
**Symptom:** Training loss was lower than expected in early steps — the model was "cheating" by attending to future tokens.  
**Root cause:** The causal mask was applied at the token level inside `sparse_attention_reference` but the block-level mask for HADS was generated without the `causal=True` flag. Block (i, j) with j > i was allowed, passing tokens from future blocks.  
**Resolution:** Added `causal=True` to all `_build_head_mask` and `make_block_mask` calls in the training path. CI test `test_local_mask_causal` now catches this.

---

## F-006: bfloat16 Gradient Underflow in Early Training

**When:** First 200 training steps  
**Symptom:** Gradient norms collapsed to 0 after step ~50. Loss plateaued.  
**Root cause:** With bfloat16 training, small gradients in early layers underflow to 0 because bfloat16 has only 7 mantissa bits. The issue was computing gradients in bfloat16 without casting up for the optimizer step.  
**Resolution:** Implemented the master-weight pattern: forward + backward in bfloat16, but gradients are cast to float32 before the AdamW update. Master weights stored in float32. See `training/mixed_precision.py`.
