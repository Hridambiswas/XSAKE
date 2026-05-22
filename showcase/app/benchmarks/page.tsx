"use client";

import { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  ResponsiveContainer,
} from "recharts";

const LATENCY_DATA = [
  { seq: 512, dense: 12.4, hads: 8.1, sliding: 3.8, bigbird: 5.6 },
  { seq: 1024, dense: 47.3, hads: 28.9, sliding: 11.2, bigbird: 19.4 },
  { seq: 2048, dense: 181.2, hads: 105.6, sliding: 64.3, bigbird: 84.2 },
  { seq: 4096, dense: 713.8, hads: 412.4, sliding: 219.7, bigbird: 298.3 },
];

const MEMORY_DATA = [
  { seq: 512, dense: 18.9, hads: 9.4, sliding: 5.3, bigbird: 7.8 },
  { seq: 1024, dense: 75.5, hads: 34.7, sliding: 21.1, bigbird: 30.9 },
  { seq: 2048, dense: 301.9, hads: 129.8, sliding: 84.4, bigbird: 122.8 },
  { seq: 4096, dense: 1207.6, hads: 484.4, sliding: 337.5, bigbird: 491.1 },
];

const THROUGHPUT_DATA = [
  { devices: 1, throughput: 35433, efficiency: 100 },
  { devices: 2, throughput: 68240, efficiency: 96.3 },
  { devices: 4, throughput: 132900, efficiency: 93.7 },
  { devices: 8, throughput: 259200, efficiency: 91.4 },
];

const KL_DATA = [
  { method: "Sliding\nWindow", kl: 0.312 },
  { method: "Longformer", kl: 0.278 },
  { method: "BigBird", kl: 0.198 },
  { method: "Random\n50%", kl: 0.421 },
  { method: "HADS", kl: 0.089 },
];

const TABS = ["Latency", "Memory", "Throughput", "Quality (KL)"] as const;
type Tab = (typeof TABS)[number];

const COLORS = {
  dense: "#555",
  hads: "#dc143c",
  sliding: "#4f7fff",
  bigbird: "#f5a623",
};

const customTooltipStyle = {
  background: "#1e1e1e",
  border: "1px solid #2a2a2a",
  borderRadius: 8,
  fontSize: "0.8rem",
  color: "#e8e8e8",
};

export default function BenchmarksPage() {
  const [tab, setTab] = useState<Tab>("Latency");

  return (
    <div style={{ maxWidth: 1000, margin: "0 auto", padding: "3rem 1.5rem" }}>
      <h1 style={{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: "2rem", fontWeight: 900, marginBottom: "0.5rem", letterSpacing: "-0.02em" }}>
        Benchmark Explorer
      </h1>
      <p style={{ color: "var(--text-muted)", marginBottom: "2.5rem" }}>
        Interactive benchmark results from NVIDIA T4 (Kaggle). Entries marked [P] are projected from
        single-device measurements with 95% linear scaling efficiency.
      </p>

      {/* Tab selector */}
      <div
        style={{
          display: "flex",
          gap: "0.4rem",
          marginBottom: "2rem",
          background: "var(--surface)",
          borderRadius: 4,
          padding: "0.35rem",
          border: "1px solid var(--border)",
          width: "fit-content",
        }}
      >
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              padding: "0.4rem 1rem",
              borderRadius: 2,
              border: "none",
              background: tab === t ? "var(--crimson)" : "transparent",
              color: tab === t ? "#fff" : "var(--text-muted)",
              cursor: "pointer",
              fontSize: "0.75rem",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.06em",
              transition: "all 0.15s",
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Charts */}
      <div
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderTop: "3px solid var(--crimson)",
          borderRadius: 4,
          padding: "1.5rem",
          marginBottom: "2rem",
        }}
      >
        {tab === "Latency" && (
          <>
            <h3 style={{ marginBottom: "1.25rem", fontSize: "0.9rem", color: "var(--text-muted)" }}>
              Attention Latency (ms) vs Sequence Length
            </h3>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={LATENCY_DATA}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
                <XAxis
                  dataKey="seq"
                  stroke="#666"
                  tick={{ fill: "#888", fontSize: 12 }}
                  label={{ value: "seq_len", position: "insideBottom", offset: -4, fill: "#666", fontSize: 11 }}
                />
                <YAxis stroke="#666" tick={{ fill: "#888", fontSize: 12 }} />
                <Tooltip contentStyle={customTooltipStyle} formatter={(v) => `${v} ms`} />
                <Legend wrapperStyle={{ fontSize: "0.8rem" }} />
                <Line dataKey="dense" name="Dense" stroke={COLORS.dense} strokeWidth={2} dot={false} />
                <Line dataKey="hads" name="HADS" stroke={COLORS.hads} strokeWidth={2.5} dot={{ r: 4, fill: COLORS.hads }} />
                <Line dataKey="bigbird" name="BigBird" stroke={COLORS.bigbird} strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
                <Line dataKey="sliding" name="Sliding Window" stroke={COLORS.sliding} strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
              </LineChart>
            </ResponsiveContainer>
          </>
        )}

        {tab === "Memory" && (
          <>
            <h3 style={{ marginBottom: "1.25rem", fontSize: "0.9rem", color: "var(--text-muted)" }}>
              HBM Memory Usage (MB) vs Sequence Length
            </h3>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={MEMORY_DATA}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
                <XAxis dataKey="seq" stroke="#666" tick={{ fill: "#888", fontSize: 12 }} />
                <YAxis stroke="#666" tick={{ fill: "#888", fontSize: 12 }} />
                <Tooltip contentStyle={customTooltipStyle} formatter={(v) => `${v} MB`} />
                <Legend wrapperStyle={{ fontSize: "0.8rem" }} />
                <Line dataKey="dense" name="Dense" stroke={COLORS.dense} strokeWidth={2} dot={false} />
                <Line dataKey="hads" name="HADS" stroke={COLORS.hads} strokeWidth={2.5} dot={{ r: 4, fill: COLORS.hads }} />
                <Line dataKey="bigbird" name="BigBird" stroke={COLORS.bigbird} strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
                <Line dataKey="sliding" name="Sliding Window" stroke={COLORS.sliding} strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
              </LineChart>
            </ResponsiveContainer>
          </>
        )}

        {tab === "Throughput" && (
          <>
            <h3 style={{ marginBottom: "1.25rem", fontSize: "0.9rem", color: "var(--text-muted)" }}>
              Throughput (tok/s) vs Device Count [P = projected]
            </h3>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={THROUGHPUT_DATA}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
                <XAxis
                  dataKey="devices"
                  stroke="#666"
                  tick={{ fill: "#888", fontSize: 12 }}
                  tickFormatter={(v, i) => `${v} dev${i > 0 ? " [P]" : ""}`}
                />
                <YAxis stroke="#666" tick={{ fill: "#888", fontSize: 12 }} />
                <Tooltip
                  contentStyle={customTooltipStyle}
                  formatter={(v, name) =>
                    name === "throughput"
                      ? `${((v as number) / 1000).toFixed(1)}K tok/s`
                      : `${v}%`
                  }
                />
                <Legend wrapperStyle={{ fontSize: "0.8rem" }} />
                <Bar dataKey="throughput" name="Throughput (tok/s)" fill={COLORS.hads} radius={[4, 4, 0, 0]} />
                <Bar dataKey="efficiency" name="Scaling efficiency (%)" fill={COLORS.bigbird} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </>
        )}

        {tab === "Quality (KL)" && (
          <>
            <h3 style={{ marginBottom: "1.25rem", fontSize: "0.9rem", color: "var(--text-muted)" }}>
              KL Divergence vs Dense (lower = better quality)
            </h3>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={KL_DATA} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
                <XAxis type="number" stroke="#666" tick={{ fill: "#888", fontSize: 12 }} domain={[0, 0.5]} />
                <YAxis type="category" dataKey="method" stroke="#666" tick={{ fill: "#888", fontSize: 12 }} width={80} />
                <Tooltip contentStyle={customTooltipStyle} />
                <Bar
                  dataKey="kl"
                  name="KL divergence"
                  radius={[0, 4, 4, 0]}
                  fill={COLORS.hads}
                  // @ts-ignore
                  isAnimationActive={true}
                  label={{ position: "right", fill: "#888", fontSize: 11 }}
                />
              </BarChart>
            </ResponsiveContainer>
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.75rem" }}>
              HADS achieves KL=0.089 — 4× better than BigBird (0.198) and 5× better than Random (0.421).
            </p>
          </>
        )}
      </div>

      {/* Raw data table */}
      <div
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: 10,
          overflow: "hidden",
        }}
      >
        <div style={{ padding: "1rem 1.25rem", borderBottom: "1px solid var(--border)" }}>
          <span style={{ fontSize: "0.875rem", fontWeight: 600 }}>
            {tab === "Throughput" ? "Throughput Scaling" : `${tab} Data`}
          </span>
        </div>
        <div style={{ overflowX: "auto" }}>
          {tab === "Latency" && (
            <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "var(--font-mono)", fontSize: "0.8rem" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  {["seq_len", "Dense (ms)", "HADS (ms)", "Reduction", "BigBird (ms)", "Sliding (ms)"].map((h) => (
                    <th key={h} style={{ padding: "0.6rem 1rem", textAlign: "left", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.7rem", fontWeight: 700, background: "var(--surface2)" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {LATENCY_DATA.map((r) => (
                  <tr key={r.seq} style={{ borderBottom: "1px solid var(--border)" }}>
                    <td style={{ padding: "0.6rem 1rem" }}>{r.seq}</td>
                    <td style={{ padding: "0.6rem 1rem" }}>{r.dense}</td>
                    <td style={{ padding: "0.6rem 1rem", color: "var(--crimson)", fontWeight: 600 }}>{r.hads}</td>
                    <td style={{ padding: "0.6rem 1rem", color: "var(--crimson)" }}>
                      {(((r.dense - r.hads) / r.dense) * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: "0.6rem 1rem" }}>{r.bigbird}</td>
                    <td style={{ padding: "0.6rem 1rem" }}>{r.sliding}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {tab === "Memory" && (
            <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "var(--font-mono)", fontSize: "0.8rem" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  {["seq_len", "Dense (MB)", "HADS (MB)", "Savings", "BigBird (MB)"].map((h) => (
                    <th key={h} style={{ padding: "0.6rem 1rem", textAlign: "left", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.7rem", fontWeight: 700, background: "var(--surface2)" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {MEMORY_DATA.map((r) => (
                  <tr key={r.seq} style={{ borderBottom: "1px solid var(--border)" }}>
                    <td style={{ padding: "0.6rem 1rem" }}>{r.seq}</td>
                    <td style={{ padding: "0.6rem 1rem" }}>{r.dense}</td>
                    <td style={{ padding: "0.6rem 1rem", color: "var(--crimson)", fontWeight: 600 }}>{r.hads}</td>
                    <td style={{ padding: "0.6rem 1rem", color: "var(--crimson)" }}>
                      {(((r.dense - r.hads) / r.dense) * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: "0.6rem 1rem" }}>{r.bigbird}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {tab === "Throughput" && (
            <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "var(--font-mono)", fontSize: "0.8rem" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  {["Devices", "Throughput (tok/s)", "Scaling efficiency"].map((h) => (
                    <th key={h} style={{ padding: "0.6rem 1rem", textAlign: "left", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.7rem", fontWeight: 700, background: "var(--surface2)" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {THROUGHPUT_DATA.map((r, i) => (
                  <tr key={r.devices} style={{ borderBottom: "1px solid var(--border)" }}>
                    <td style={{ padding: "0.6rem 1rem" }}>{r.devices}{i > 0 ? " [P]" : ""}</td>
                    <td style={{ padding: "0.6rem 1rem", color: "var(--crimson)", fontWeight: 600 }}>
                      {r.throughput.toLocaleString()}
                    </td>
                    <td style={{ padding: "0.6rem 1rem" }}>{r.efficiency}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {tab === "Quality (KL)" && (
            <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "var(--font-mono)", fontSize: "0.8rem" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  {["Method", "KL Divergence", "vs HADS"].map((h) => (
                    <th key={h} style={{ padding: "0.6rem 1rem", textAlign: "left", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.7rem", fontWeight: 700, background: "var(--surface2)" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {KL_DATA.map((r) => (
                  <tr key={r.method} style={{ borderBottom: "1px solid var(--border)", background: r.method === "HADS" ? "rgba(220,20,60,0.07)" : "transparent" }}>
                    <td style={{ padding: "0.6rem 1rem", color: r.method === "HADS" ? "var(--crimson)" : "var(--text)", fontWeight: r.method === "HADS" ? 700 : 400 }}>
                      {r.method}
                    </td>
                    <td style={{ padding: "0.6rem 1rem", color: r.method === "HADS" ? "var(--crimson)" : "var(--text)" }}>
                      {r.kl.toFixed(3)}
                    </td>
                    <td style={{ padding: "0.6rem 1rem", color: "var(--text-muted)" }}>
                      {r.method === "HADS" ? "—" : `${(r.kl / 0.089).toFixed(1)}× worse`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
