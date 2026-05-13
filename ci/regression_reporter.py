"""
Benchmark regression reporter for XSAKE CI.

Compares the current benchmark results against a baseline stored in
benchmarks/results/baseline_*.json and flags any regressions that
exceed the thresholds in ci/regression_thresholds.yaml.

Exit code 0 = no regression, 1 = regression detected.

Usage:
    python -m ci.regression_reporter \
        --current  benchmarks/results/latency_results.json \
        --baseline benchmarks/results/baseline_latency.json \
        --metric   latency

Called by .github/workflows/benchmark_regression.yml on every commit.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


THRESHOLDS_PATH = Path("ci/regression_thresholds.yaml")


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_thresholds() -> dict:
    with open(THRESHOLDS_PATH) as f:
        return yaml.safe_load(f)


# ─── Regression checks ────────────────────────────────────────────────────────

def check_latency_regression(current: dict, baseline: dict, thresholds: dict) -> list[str]:
    failures = []
    max_pct = thresholds["latency"]["max_regression_pct"]

    for seq in thresholds["latency"]["seq_lens"]:
        key = str(seq)
        if key not in current.get("seq_lens", {}):
            continue
        if key not in baseline.get("seq_lens", {}):
            continue

        cur_ms  = current["seq_lens"][key]["xsake_ms"]
        base_ms = baseline["seq_lens"][key]["xsake_ms"]

        if base_ms <= 0:
            continue

        pct_change = (cur_ms - base_ms) / base_ms * 100

        if pct_change > max_pct:
            failures.append(
                f"LATENCY REGRESSION at seq={seq}: "
                f"{base_ms:.2f}ms → {cur_ms:.2f}ms "
                f"(+{pct_change:.1f}% > threshold {max_pct}%)"
            )

    return failures


def check_memory_regression(current: dict, baseline: dict, thresholds: dict) -> list[str]:
    failures = []
    max_pct = thresholds["memory"]["max_regression_pct"]

    for seq in thresholds["memory"]["seq_lens"]:
        key = str(seq)
        if key not in current.get("seq_lens", {}):
            continue
        if key not in baseline.get("seq_lens", {}):
            continue

        cur_mb  = current["seq_lens"][key]["sparse_mb"]
        base_mb = baseline["seq_lens"][key]["sparse_mb"]

        if base_mb <= 0:
            continue

        pct_change = (cur_mb - base_mb) / base_mb * 100
        if pct_change > max_pct:
            failures.append(
                f"MEMORY REGRESSION at seq={seq}: "
                f"{base_mb:.1f}MB → {cur_mb:.1f}MB "
                f"(+{pct_change:.1f}% > threshold {max_pct}%)"
            )

    return failures


def check_sparsity_regression(current: dict, thresholds: dict) -> list[str]:
    failures = []
    min_reduction = thresholds["sparsity"]["min_active_reduction_pct"]

    patterns = current.get("patterns", {})
    if "hads" in patterns:
        actual_reduction = (1 - patterns["hads"]["active_frac"]) * 100
        if actual_reduction < min_reduction:
            failures.append(
                f"SPARSITY REGRESSION: HADS only skipping "
                f"{actual_reduction:.1f}% of blocks, "
                f"threshold {min_reduction}%"
            )
    return failures


def check_correctness(current: dict, thresholds: dict) -> list[str]:
    failures = []
    max_kl = thresholds["correctness"]["max_kl_from_dense"]
    patterns = current.get("patterns", {})

    for name, data in patterns.items():
        if name == "dense":
            continue
        kl = data.get("kl_div", 0)
        if kl > max_kl:
            failures.append(
                f"CORRECTNESS: {name} KL from dense = {kl:.6f} > threshold {max_kl}"
            )
    return failures


# ─── Report generation ────────────────────────────────────────────────────────

def build_report(failures: list[str], current: dict, baseline: dict | None) -> str:
    lines = ["# XSAKE Benchmark Regression Report\n"]

    if failures:
        lines.append(f"## ❌ {len(failures)} regression(s) detected\n")
        for f in failures:
            lines.append(f"- {f}")
    else:
        lines.append("## ✅ No regressions detected\n")

    lines.append("\n## Current results\n```json")
    lines.append(json.dumps(current, indent=2)[:2000])
    lines.append("```")

    if baseline:
        lines.append("\n## Baseline results\n```json")
        lines.append(json.dumps(baseline, indent=2)[:2000])
        lines.append("```")

    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current",  required=True, help="Current benchmark JSON")
    parser.add_argument("--baseline", help="Baseline benchmark JSON (optional)")
    parser.add_argument("--metric",   choices=["latency", "memory", "hads", "correctness"],
                        required=True)
    parser.add_argument("--out", default="benchmarks/results/regression_report.md")
    args = parser.parse_args()

    current  = load_json(args.current)
    baseline = load_json(args.baseline) if args.baseline else {}
    thresholds = load_thresholds()

    failures = []
    if args.metric == "latency" and baseline:
        failures += check_latency_regression(current, baseline, thresholds)
    elif args.metric == "memory" and baseline:
        failures += check_memory_regression(current, baseline, thresholds)
    elif args.metric == "hads":
        failures += check_sparsity_regression(current, thresholds)
        failures += check_correctness(current, thresholds)

    report = build_report(failures, current, baseline)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report)

    if failures:
        print(f"\n🔴 {len(failures)} regression(s):\n")
        for f in failures:
            print(f"  • {f}")
        print(f"\nFull report: {args.out}")
        sys.exit(1)
    else:
        print("✅ No regressions detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
