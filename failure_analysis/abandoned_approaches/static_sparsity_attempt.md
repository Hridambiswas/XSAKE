# Abandoned: Static Sparsity Patterns

## What was attempted

Applied a single sliding-window sparsity mask (window=3 blocks, ~50% sparsity)
uniformly across all attention heads and all layers.

## What broke

Perplexity on OpenWebText was 8–12% higher than dense attention at the same
number of training steps. The gap did not close with longer training.

## Why it failed

Visualising the attention weights of a trained dense model reveals that heads
are NOT uniform:

- Heads 0–3 (syntactic): near-diagonal, highly peaked — sliding window fits perfectly
- Heads 4–7 (positional): attend to fixed offsets (position 0, -1) — very sparse by nature
- Heads 8–11 (semantic): diffuse, global — require attending to tokens far outside any fixed window

A window-3 block mask works well for syntactic/positional heads but catastrophically
restricts semantic heads that need long-range context. The model compensates by routing
all long-range information through the few dense layers, creating a bottleneck.

## What we learned

Sparsity must be calibrated per head, not applied globally.
HADS emerged from this failure.
