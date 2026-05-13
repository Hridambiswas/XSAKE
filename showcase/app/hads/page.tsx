"use client";

import { useState, useMemo } from "react";

const N_HEADS = 12;
const N_BLOCKS = 8;

function generateEntropy(scale: number): number[] {
  // Simulated head entropies: mix of focused (low) and broad (high) heads
  const base = [0.8, 3.2, 1.1, 2.7, 0.5, 1.9, 3.6, 2.1, 0.3, 2.9, 1.4, 3.0];
  return base.map((v) => Math.max(0.1, Math.min(4.0, v * scale)));
}

function entropyToSparsity(entropy: number[], minS: number, maxS: number): number[] {
  const min = Math.min(...entropy);
  const max = Math.max(...entropy);
  return entropy.map((h) => {
    if (max === min) return (minS + maxS) / 2;
    return maxS - ((h - min) / (max - min)) * (maxS - minS);
  });
}

function buildBlockMask(headIdx: number, sparsity: number): boolean[][] {
  const mask: boolean[][] = Array.from({ length: N_BLOCKS }, (_, i) =>
    Array.from({ length: N_BLOCKS }, (_, j) => j <= i)
  );
  // Always keep first column (global token)
  for (let i = 0; i < N_BLOCKS; i++) mask[i][0] = true;

  // Randomly skip upper-triangle blocks based on sparsity
  const rng = (seed: number) => {
    let s = seed;
    return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0x100000000; };
  };
  const rand = rng(headIdx * 31 + 7);
  for (let i = 1; i < N_BLOCKS; i++) {
    for (let j = 1; j < i; j++) {
      if (rand() < sparsity) mask[i][j] = false;
    }
  }
  return mask;
}

export default function HADSPage() {
  const [selectedHead, setSelectedHead] = useState(0);
  const [minSparsity, setMinSparsity] = useState(0.1);
  const [maxSparsity, setMaxSparsity] = useState(0.9);
  const [entropyScale, setEntropyScale] = useState(1.0);

  const entropy = useMemo(() => generateEntropy(entropyScale), [entropyScale]);
  const sparsity = useMemo(
    () => entropyToSparsity(entropy, minSparsity, maxSparsity),
    [entropy, minSparsity, maxSparsity]
  );
  const mask = useMemo(
    () => buildBlockMask(selectedHead, sparsity[selectedHead]),
    [selectedHead, sparsity]
  );

  const activeCount = mask.flat().filter(Boolean).length;
  const totalBlocks = N_BLOCKS * N_BLOCKS;

  return (
    <div style={{ maxWidth: 1000, margin: "0 auto", padding: "3rem 1.5rem" }}>
      <h1 style={{ fontSize: "1.75rem", fontWeight: 800, marginBottom: "0.5rem" }}>
        HADS Pattern Visualizer
      </h1>
      <p style={{ color: "var(--text-muted)", marginBottom: "2.5rem" }}>
        Explore how Head-Adaptive Dynamic Sparsity assigns per-head block masks based on Shannon
        entropy. Semantic heads (low entropy) get high sparsity; syntactic heads (high entropy) stay
        dense.
      </p>

      {/* Controls */}
      <div
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: 10,
          padding: "1.5rem",
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
          gap: "1.5rem",
          marginBottom: "2rem",
        }}
      >
        {[
          {
            label: "Entropy Scale",
            value: entropyScale,
            min: 0.3,
            max: 2.0,
            step: 0.1,
            set: setEntropyScale,
          },
          {
            label: `Min Sparsity (${(minSparsity * 100).toFixed(0)}%)`,
            value: minSparsity,
            min: 0.05,
            max: 0.4,
            step: 0.05,
            set: setMinSparsity,
          },
          {
            label: `Max Sparsity (${(maxSparsity * 100).toFixed(0)}%)`,
            value: maxSparsity,
            min: 0.5,
            max: 0.95,
            step: 0.05,
            set: setMaxSparsity,
          },
        ].map((ctrl) => (
          <div key={ctrl.label}>
            <label style={{ fontSize: "0.8rem", color: "var(--text-muted)", display: "block", marginBottom: "0.4rem" }}>
              {ctrl.label}
            </label>
            <input
              type="range"
              min={ctrl.min}
              max={ctrl.max}
              step={ctrl.step}
              value={ctrl.value}
              onChange={(e) => ctrl.set(parseFloat(e.target.value))}
              style={{ width: "100%", accentColor: "var(--crimson)" }}
            />
          </div>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2rem", alignItems: "start" }}>
        {/* Head entropy panel */}
        <div
          style={{
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: 10,
            padding: "1.25rem",
          }}
        >
          <h3 style={{ fontSize: "0.875rem", fontWeight: 600, marginBottom: "1rem" }}>
            Per-Head Entropy & Sparsity
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            {entropy.map((h, i) => (
              <button
                key={i}
                onClick={() => setSelectedHead(i)}
                style={{
                  background: i === selectedHead ? "rgba(220,20,60,0.12)" : "transparent",
                  border: i === selectedHead ? "1px solid var(--crimson)" : "1px solid transparent",
                  borderRadius: 6,
                  padding: "0.5rem 0.75rem",
                  cursor: "pointer",
                  textAlign: "left",
                  width: "100%",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: "0.75rem",
                      color: "var(--text-muted)",
                      minWidth: 40,
                    }}
                  >
                    H{i}
                  </span>
                  <div
                    style={{
                      flex: 1,
                      height: 8,
                      background: "var(--surface2)",
                      borderRadius: 4,
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        height: "100%",
                        width: `${(sparsity[i] * 100).toFixed(0)}%`,
                        background: `hsl(${360 - sparsity[i] * 120}, 70%, 50%)`,
                        borderRadius: 4,
                        transition: "width 0.3s ease",
                      }}
                    />
                  </div>
                  <span
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: "0.7rem",
                      color: "var(--text-muted)",
                      minWidth: 70,
                      textAlign: "right",
                    }}
                  >
                    H={h.toFixed(2)} ρ={( sparsity[i] * 100).toFixed(0)}%
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Block mask grid */}
        <div>
          <div
            style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 10,
              padding: "1.25rem",
              marginBottom: "1rem",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "1rem",
              }}
            >
              <h3 style={{ fontSize: "0.875rem", fontWeight: 600 }}>
                Block Mask — Head {selectedHead}
              </h3>
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.75rem",
                  color: "var(--crimson)",
                }}
              >
                {activeCount}/{totalBlocks} active ({((activeCount / totalBlocks) * 100).toFixed(0)}%)
              </span>
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: `repeat(${N_BLOCKS}, 1fr)`,
                gap: 3,
              }}
            >
              {mask.map((row, i) =>
                row.map((active, j) => (
                  <div
                    key={`${i}-${j}`}
                    title={`Q-block ${i} → K-block ${j}: ${active ? "active" : "skipped"}`}
                    style={{
                      aspectRatio: "1",
                      borderRadius: 3,
                      background: active
                        ? i === j
                          ? "var(--crimson)"
                          : "rgba(220,20,60,0.45)"
                        : "var(--surface2)",
                      transition: "background 0.2s ease",
                    }}
                  />
                ))
              )}
            </div>
            <div
              style={{
                display: "flex",
                gap: "1rem",
                marginTop: "0.75rem",
                fontSize: "0.7rem",
                color: "var(--text-muted)",
              }}
            >
              <span>
                <span
                  style={{
                    display: "inline-block",
                    width: 10,
                    height: 10,
                    background: "var(--crimson)",
                    borderRadius: 2,
                    marginRight: 4,
                  }}
                />
                Diagonal (self)
              </span>
              <span>
                <span
                  style={{
                    display: "inline-block",
                    width: 10,
                    height: 10,
                    background: "rgba(220,20,60,0.45)",
                    borderRadius: 2,
                    marginRight: 4,
                  }}
                />
                Active
              </span>
              <span>
                <span
                  style={{
                    display: "inline-block",
                    width: 10,
                    height: 10,
                    background: "var(--surface2)",
                    borderRadius: 2,
                    marginRight: 4,
                  }}
                />
                Skipped
              </span>
            </div>
          </div>

          {/* Stats */}
          <div
            style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 10,
              padding: "1rem 1.25rem",
              fontFamily: "var(--font-mono)",
              fontSize: "0.8rem",
            }}
          >
            <div style={{ color: "var(--crimson)", marginBottom: "0.5rem" }}>
              # Head {selectedHead} statistics
            </div>
            <div style={{ color: "var(--text-muted)", lineHeight: 1.8 }}>
              <div>
                entropy ={" "}
                <span style={{ color: "var(--text)" }}>{entropy[selectedHead].toFixed(3)}</span>
              </div>
              <div>
                sparsity ={" "}
                <span style={{ color: "var(--text)" }}>
                  {(sparsity[selectedHead] * 100).toFixed(1)}%
                </span>
              </div>
              <div>
                active_blocks ={" "}
                <span style={{ color: "var(--text)" }}>
                  {activeCount} / {totalBlocks}
                </span>
              </div>
              <div>
                type ={" "}
                <span style={{ color: "var(--text)" }}>
                  {entropy[selectedHead] > 2.5
                    ? "syntactic (broad)"
                    : entropy[selectedHead] < 1.2
                    ? "semantic (focused)"
                    : "mixed"}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
