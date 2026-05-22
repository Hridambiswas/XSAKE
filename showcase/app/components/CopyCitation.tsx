"use client";

import { useState } from "react";

const BIBTEX = `@software{biswas2024xsake,
  author    = {Biswas, Hridam},
  title     = {{XSAKE}: {XLA}-Fused Sparse Attention Kernel Engine},
  year      = {2024},
  publisher = {GitHub},
  url       = {https://github.com/Hridambiswas/XSAKE},
  note      = {Head-Adaptive Dynamic Sparsity (HADS) for JAX/Pallas.
               42\\% latency reduction, 60\\% HBM savings on NVIDIA T4.}
}`;

export default function CopyCitation() {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(BIBTEX).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
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
        Cite This Work
      </h2>
      <div
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderTop: "3px solid var(--crimson)",
          borderRadius: 4,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "0.75rem 1.25rem",
            borderBottom: "1px solid var(--border)",
            background: "var(--surface2)",
          }}
        >
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.7rem",
              color: "var(--text-muted)",
              textTransform: "uppercase",
              letterSpacing: "0.1em",
            }}
          >
            BibTeX
          </span>
          <button
            onClick={handleCopy}
            style={{
              background: copied ? "var(--crimson)" : "transparent",
              color: copied ? "#fff" : "var(--text-muted)",
              border: "1px solid var(--border)",
              borderRadius: 2,
              padding: "0.25rem 0.75rem",
              cursor: "pointer",
              fontSize: "0.7rem",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.07em",
              transition: "all 0.15s",
            }}
          >
            {copied ? "Copied!" : "Copy"}
          </button>
        </div>
        <pre
          style={{
            margin: 0,
            padding: "1.25rem",
            fontFamily: "var(--font-mono)",
            fontSize: "0.8rem",
            lineHeight: 1.7,
            color: "var(--text-muted)",
            overflowX: "auto",
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
          }}
        >
          {BIBTEX}
        </pre>
      </div>
    </section>
  );
}
