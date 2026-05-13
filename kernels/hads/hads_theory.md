# HADS — Head-Adaptive Dynamic Sparsity: Theory and Motivation

## 1. The Problem With Uniform Sparsity

Standard sparse attention methods (BigBird, Longformer, sliding-window) apply a **single
sparsity pattern to all attention heads equally**. This is a blunt instrument.

Empirically, transformer attention heads specialise:

| Head type       | Behaviour                                       | Entropy |
|-----------------|-------------------------------------------------|---------|
| Syntactic       | Attends to adjacent tokens, delimiters, POS tags | Low     |
| Positional      | Attends to fixed offsets (e.g., -1, +1, start)  | Very low |
| Semantic/global | Attends to relevant tokens across full context   | High    |
| Rare-pattern    | Attends to specific rare constructions           | Medium  |

Applying 70% sparsity uniformly destroys the semantic heads (high entropy, need dense
patterns) while under-exploiting the syntactic heads (low entropy, could tolerate 90%+).

## 2. HADS: The Core Idea

**HADS assigns each head its own sparsity ratio based on its measured entropy.**

### 2.1 Entropy Measurement

For a head $h$ with attention distribution $\mathbf{a}^{(h)} \in \mathbb{R}^{B \times L \times L}$:

$$H_h = -\frac{1}{BL} \sum_{b=1}^{B} \sum_{i=1}^{L} \sum_{j=1}^{L} a^{(h)}_{b,i,j} \log a^{(h)}_{b,i,j}$$

$H_h$ is computed over a **calibration set** of 10 forward passes using the current
model checkpoint. It reflects the true distributional behaviour of each head on real data.

### 2.2 Entropy → Sparsity Mapping

Linear interpolation maps entropy to sparsity ratio:

$$\rho_h = \rho_{\max} - \frac{H_h - H_{\min}}{H_{\max} - H_{\min}} \cdot (\rho_{\max} - \rho_{\min})$$

Where:
- $\rho_{\min} = 0.10$ (densest: high-entropy semantic heads)
- $\rho_{\max} = 0.90$ (sparsest: low-entropy syntactic heads)
- $H_{\min}, H_{\max}$ are min/max entropy across all heads in the model

This is a **monotone decreasing mapping**: higher entropy → lower sparsity (denser attention).

### 2.3 Block Mask Construction

For each head $h$, the block mask $M^{(h)} \in \{0,1\}^{N_b \times N_b}$ is constructed as:

1. **Local window** (always on): $M^{(h)}_{i,j} = 1$ for $|i - j| \leq w$ where $w=2$  
   — ensures every position attends to its local context regardless of sparsity

2. **Random long-range blocks**: randomly select $\lfloor N_b \cdot (1 - \rho_h) \rfloor$ 
   additional kv blocks per q block, using a deterministic per-head seed  
   — provides long-range coverage proportional to the head's entropy

The seed is `seed=head_index` making masks **deterministic given a profile** but
**different across heads**, avoiding correlated sparsity patterns.

## 3. Why This Beats Sliding Window Sparsity

**Sliding window** (Longformer-style) uses a fixed window of width $w$ for all heads.
At the same overall FLOP count as HADS:

| Metric                | Sliding Window        | HADS                         |
|-----------------------|-----------------------|------------------------------|
| Semantic head coverage | Fixed window only    | Dense (low sparsity)         |
| Syntactic head coverage| Fixed window         | Very sparse (saves FLOPs)    |
| Perplexity at same sparsity | Higher (worse) | Lower (target: 2–5% gain)   |
| Configuration         | One param: window size| Two params: min/max sparsity |

The gain comes because HADS **allocates FLOPs where they matter** (semantic heads)
and **reclaims FLOPs from heads that don't need them** (syntactic heads).

## 4. Relationship to Prior Work

| Method     | Sparsity type      | Per-head adaptation |
|------------|--------------------|---------------------|
| BigBird    | Local + global + random | No              |
| Longformer | Local + global     | No                  |
| Reformer   | LSH-based dynamic  | No                  |
| A-Star     | Learned per-layer  | Layer-level only    |
| **HADS**   | Entropy-calibrated | **Yes — per head**  |

HADS is the first method (to our knowledge) to use **per-head entropy measured on real
data** as the criterion for assigning sparsity structure.

## 5. Computational Cost of Calibration

Calibration requires $C$ forward passes (default $C=10$) and memory to store the
attention weights $\mathbf{a}^{(h)} \in \mathbb{R}^{B \times H \times L \times L}$.

For a 12-layer, 12-head model at $L=1024$, $B=4$:
- Attention weights per step: $12 \times 12 \times 1024^2 \times 4$ bytes ≈ **600 MB**
- 10 calibration steps: single-pass, no gradient computation
- Total calibration time: ~5% of a training epoch

This is a **one-time cost** per checkpoint; the resulting HADSProfile is serialised
and reused across all subsequent training steps.

## 6. Expected Results

Based on the sparsity budget analysis:

| Sequence length | Dense FLOPs | HADS FLOPs | Reduction |
|----------------|-------------|------------|-----------|
| 512            | 1×          | ~0.60×     | ~40%      |
| 1024           | 1×          | ~0.58×     | ~42%      |
| 2048           | 1×          | ~0.57×     | ~43%      |
| 4096           | 1×          | ~0.55×     | ~45%      |

Target: **35–45% latency reduction at seq_len=4096** vs dense attention.
Target: **2–5% perplexity improvement** over sliding-window at matched sparsity.
