import Link from "next/link";

const RESULTS = [
  { label: "Latency reduction (seq=4096)", value: "42.2%", sub: "713ms → 412ms on T4" },
  { label: "HBM memory savings (seq=4096)", value: "59.9%", sub: "1207MB → 484MB" },
  { label: "KL divergence vs dense", value: "0.089", sub: "best among sparse methods" },
  { label: "Perplexity gap vs dense", value: "+1.8%", sub: "10k steps OpenWebText" },
];

const COMPARISON = [
  { method: "Dense", active: 100, latency: 181.2, kl: 0.0 },
  { method: "Sliding Window", active: 28, latency: 64.3, kl: 0.312 },
  { method: "Longformer", active: 32, latency: 71.8, kl: 0.278 },
  { method: "BigBird", active: 41, latency: 84.2, kl: 0.198 },
  { method: "Random 50%", active: 50, latency: 103.7, kl: 0.421 },
  { method: "HADS ✓", active: 43, latency: 105.6, kl: 0.089, highlight: true },
];

export default function Home() {
  return (
    <div style={{ maxWidth: 1000, margin: "0 auto", padding: "4rem 1.5rem" }}>
      {/* Hero */}
      <div style={{ marginBottom: "4rem" }}>
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.75rem",
            color: "var(--crimson)",
            letterSpacing: "0.15em",
            textTransform: "uppercase",
            marginBottom: "1rem",
          }}
        >
          JAX · Pallas · XLA
        </div>
        <h1
          style={{
            fontFamily: "Georgia, 'Times New Roman', serif",
            fontSize: "clamp(2.25rem, 5.5vw, 4rem)",
            fontWeight: 900,
            lineHeight: 1.05,
            letterSpacing: "-0.03em",
            marginBottom: "1.5rem",
          }}
        >
          XLA-Fused Sparse Attention
          <br />
          <span style={{ color: "var(--crimson)" }}>Kernel Engine</span>
        </h1>
        <p
          style={{
            fontSize: "1.125rem",
            color: "var(--text-muted)",
            maxWidth: 600,
            lineHeight: 1.7,
            marginBottom: "2rem",
          }}
        >
          XSAKE implements block-sparse attention on JAX + Pallas with{" "}
          <strong style={{ color: "var(--text)" }}>HADS</strong> — a novel Head-Adaptive Dynamic
          Sparsity algorithm that assigns per-head sparsity ratios based on Shannon entropy, achieving
          42% latency reduction while preserving output quality better than any fixed-pattern baseline.
        </p>
        <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
          <Link
            href="/hads"
            style={{
              background: "var(--crimson)",
              color: "#fff",
              padding: "0.65rem 2rem",
              borderRadius: 2,
              textDecoration: "none",
              fontWeight: 700,
              fontSize: "0.8rem",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
            }}
          >
            Explore HADS
          </Link>
          <Link
            href="/benchmarks"
            style={{
              border: "2px solid var(--border)",
              color: "var(--text)",
              padding: "0.65rem 2rem",
              borderRadius: 2,
              textDecoration: "none",
              fontWeight: 700,
              fontSize: "0.8rem",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
            }}
          >
            Benchmarks
          </Link>
          <Link
            href="/training"
            style={{
              border: "2px solid var(--border)",
              color: "var(--text)",
              padding: "0.65rem 2rem",
              borderRadius: 2,
              textDecoration: "none",
              fontWeight: 700,
              fontSize: "0.8rem",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
            }}
          >
            Training Replay
          </Link>
        </div>
      </div>

      {/* Key Results */}
      <section style={{ marginBottom: "4rem" }}>
        <h2
          style={{
            fontSize: "0.7rem",
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            color: "var(--text-muted)",
            marginBottom: "1.5rem",
            fontFamily: "var(--font-mono)",
            paddingBottom: "0.75rem",
            borderBottom: "1px solid var(--border)",
          }}
        >
          Key Results · NVIDIA T4 (Kaggle)
        </h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem" }}>
          {RESULTS.map((r) => (
            <div
              key={r.label}
              style={{
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderLeft: "3px solid var(--crimson)",
                borderRadius: 4,
                padding: "1.25rem",
              }}
            >
              <div
                style={{
                  fontSize: "2.25rem",
                  fontWeight: 900,
                  color: "var(--crimson)",
                  fontFamily: "var(--font-mono)",
                  marginBottom: "0.25rem",
                }}
              >
                {r.value}
              </div>
              <div style={{ fontSize: "0.875rem", color: "var(--text)", marginBottom: "0.25rem" }}>
                {r.label}
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>{r.sub}</div>
            </div>
          ))}
        </div>
      </section>

      {/* HADS Explanation */}
      <section style={{ marginBottom: "4rem" }}>
        <h2
          style={{
            fontFamily: "Georgia, 'Times New Roman', serif",
            fontSize: "1.5rem",
            fontWeight: 700,
            marginBottom: "1rem",
            paddingBottom: "0.5rem",
            borderBottom: "1px solid var(--border)",
          }}
        >
          What is HADS?
        </h2>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "2rem",
            alignItems: "start",
          }}
        >
          <div>
            <p style={{ color: "var(--text-muted)", lineHeight: 1.7, marginBottom: "1rem" }}>
              Standard sparse patterns (BigBird, Longformer) assign the same block mask to every
              attention head. But attention heads specialize — semantic heads focus narrowly, syntactic
              heads attend broadly.
            </p>
            <p style={{ color: "var(--text-muted)", lineHeight: 1.7 }}>
              HADS measures per-head Shannon entropy from a calibration pass and assigns per-head
              sparsity ratios: focused heads skip 80–90% of KV blocks, broad heads skip only 10–20%.
            </p>
          </div>
          <div
            style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 10,
              padding: "1.25rem",
              fontFamily: "var(--font-mono)",
              fontSize: "0.8rem",
              color: "var(--text-muted)",
              lineHeight: 1.8,
            }}
          >
            <div style={{ color: "var(--crimson)", marginBottom: "0.5rem" }}>
              # entropy → sparsity (12 heads)
            </div>
            {[
              { h: 0, skip: 62, type: "" },
              { h: 1, skip: 82, type: " ← syntactic" },
              { h: 2, skip: 18, type: " ← semantic" },
              { h: 3, skip: 71, type: "" },
              { h: 4, skip: 55, type: "" },
              { h: 5, skip: 44, type: "" },
            ].map(({ h, skip, type }) => (
              <div key={h} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ color: "var(--text)" }}>H{h}</span>
                <div
                  style={{
                    height: 10,
                    width: `${skip * 1.5}px`,
                    background: "var(--crimson)",
                    borderRadius: 2,
                    opacity: 0.6 + skip / 300,
                  }}
                />
                <span style={{ fontSize: "0.7rem" }}>
                  {skip}%{type}
                </span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Comparison Table */}
      <section style={{ marginBottom: "4rem" }}>
        <h2 style={{ fontSize: "1.25rem", fontWeight: 700, marginBottom: "1rem" }}>
          HADS vs Baselines (seq=2048)
        </h2>
        <div
          style={{
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: 10,
            overflow: "hidden",
          }}
        >
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border)" }}>
                {["Method", "Active blocks", "Latency (ms)", "KL vs dense"].map((h) => (
                  <th
                    key={h}
                    style={{
                      padding: "0.75rem 1rem",
                      textAlign: "left",
                      fontSize: "0.75rem",
                      letterSpacing: "0.1em",
                      textTransform: "uppercase",
                      color: "var(--text-muted)",
                    }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {COMPARISON.map((r) => (
                <tr
                  key={r.method}
                  style={{
                    borderBottom: "1px solid var(--border)",
                    background: r.highlight ? "rgba(220,20,60,0.07)" : "transparent",
                  }}
                >
                  <td
                    style={{
                      padding: "0.75rem 1rem",
                      fontWeight: r.highlight ? 700 : 400,
                      color: r.highlight ? "var(--crimson)" : "var(--text)",
                      fontFamily: r.highlight ? "var(--font-mono)" : "inherit",
                    }}
                  >
                    {r.method}
                  </td>
                  <td style={{ padding: "0.75rem 1rem", fontFamily: "var(--font-mono)", fontSize: "0.875rem" }}>
                    {r.active}%
                  </td>
                  <td style={{ padding: "0.75rem 1rem", fontFamily: "var(--font-mono)", fontSize: "0.875rem" }}>
                    {r.latency}
                  </td>
                  <td
                    style={{
                      padding: "0.75rem 1rem",
                      fontFamily: "var(--font-mono)",
                      fontSize: "0.875rem",
                      color: r.kl < 0.1 ? "var(--crimson)" : "var(--text)",
                    }}
                  >
                    {r.kl.toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.5rem" }}>
          HADS achieves the lowest KL divergence (best quality) among all sparse methods.
        </p>
      </section>

      {/* Architecture */}
      <section>
        <h2 style={{ fontSize: "1.25rem", fontWeight: 700, marginBottom: "1rem" }}>Architecture</h2>
        <div
          style={{
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: 10,
            padding: "1.5rem",
            fontFamily: "var(--font-mono)",
            fontSize: "0.8rem",
            lineHeight: 1.8,
            color: "var(--text-muted)",
            overflowX: "auto",
          }}
        >
          <div style={{ color: "var(--crimson)" }}>XSAKE/</div>
          {[
            "├── kernels/pallas/    ← Block-sparse attention (Pallas → Triton/XLA)",
            "│   └── sparse_attention.py  # Online softmax, fori_loop over KV blocks",
            "├── kernels/hads/      ← Head-Adaptive Dynamic Sparsity (novel)",
            "│   └── hads_pattern.py  # Entropy measurement + block mask construction",
            "├── model/             ← GPT-style transformer (Flax/linen + XSAKE attn)",
            "├── distributed/       ← 2D mesh, pmap, NamedSharding",
            "├── training/          ← AdamW + cosine warmup, Orbax checkpoints",
            "├── benchmarks/        ← Latency / memory / throughput / HADS ablation",
            "├── observability/     ← FastAPI dashboard + Prometheus + W&B",
            "└── showcase/          ← This Next.js app (you are here)",
          ].map((line) => (
            <div key={line}>{line}</div>
          ))}
        </div>
      </section>
    </div>
  );
}
