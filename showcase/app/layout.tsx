import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "XSAKE — XLA-Fused Sparse Attention Kernel Engine",
  description:
    "Head-Adaptive Dynamic Sparsity (HADS) for long-sequence transformers on JAX + Pallas. 42% latency reduction, 60% HBM savings.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col" style={{ background: "var(--bg)", color: "var(--text)" }}>
        <nav
          style={{
            background: "var(--surface)",
            borderBottom: "1px solid var(--border)",
            position: "sticky",
            top: 0,
            zIndex: 50,
            boxShadow: "0 4px 24px rgba(0,0,0,0.5)",
          }}
        >
          <div
            style={{
              maxWidth: 1100,
              margin: "0 auto",
              padding: "0 1.5rem",
              height: 72,
              display: "flex",
              alignItems: "center",
              gap: "2rem",
            }}
          >
            <Link
              href="/"
              style={{
                fontFamily: "var(--font-mono)",
                fontWeight: 800,
                fontSize: "1.125rem",
                color: "var(--crimson)",
                textDecoration: "none",
                letterSpacing: "0.1em",
                borderLeft: "3px solid var(--crimson)",
                paddingLeft: "0.75rem",
              }}
            >
              XSAKE
            </Link>
            <div style={{ display: "flex", gap: "1.5rem", flex: 1 }}>
              {[
                { href: "/hads", label: "HADS Visualizer" },
                { href: "/benchmarks", label: "Benchmarks" },
                { href: "/training", label: "Training Replay" },
              ].map((n) => (
                <Link
                  key={n.href}
                  href={n.href}
                  style={{
                    color: "var(--text-muted)",
                    textDecoration: "none",
                    fontSize: "0.75rem",
                    fontWeight: 600,
                    textTransform: "uppercase",
                    letterSpacing: "0.09em",
                  }}
                >
                  {n.label}
                </Link>
              ))}
            </div>
            <a
              href="https://github.com/Hridambiswas/XSAKE"
              target="_blank"
              rel="noopener noreferrer"
              style={{
                fontSize: "0.7rem",
                color: "var(--text-muted)",
                textDecoration: "none",
                border: "1px solid var(--border)",
                padding: "0.35rem 0.9rem",
                borderRadius: 2,
                textTransform: "uppercase",
                letterSpacing: "0.08em",
                fontWeight: 600,
              }}
            >
              GitHub
            </a>
          </div>
        </nav>
        <main style={{ flex: 1 }}>{children}</main>
        <footer
          style={{
            borderTop: "1px solid var(--border)",
            padding: "2rem 1.5rem",
            textAlign: "center",
          }}
        >
          <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "0.35rem" }}>
            XSAKE · JAX + Pallas · Head-Adaptive Dynamic Sparsity · MIT License
          </div>
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", opacity: 0.6 }}>
            Hridam Biswas · KIIT University
          </div>
        </footer>
      </body>
    </html>
  );
}
