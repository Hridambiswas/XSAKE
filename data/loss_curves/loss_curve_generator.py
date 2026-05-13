"""
Generate and save training loss curve plots.

Reads W&B run history or a local CSV log and produces:
  - training_loss.png  : loss vs step with smoothed trend line
  - perplexity.png     : perplexity vs step

Run after training:
    python -m data.loss_curves.loss_curve_generator \
        --log results/training_log.csv \
        --out  data/loss_curves/
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def smooth(values: np.ndarray, weight: float = 0.9) -> np.ndarray:
    """Exponential moving average smoothing (TensorBoard-style)."""
    smoothed = np.zeros_like(values, dtype=float)
    last = values[0]
    for i, v in enumerate(values):
        last = weight * last + (1 - weight) * v
        smoothed[i] = last / (1 - weight ** (i + 1))   # bias correction
    return smoothed


def plot_loss_curve(
    steps: np.ndarray,
    losses: np.ndarray,
    out_path: str,
    title: str = "XSAKE Training Loss — OpenWebText",
    smoothing: float = 0.9,
) -> None:
    """
    Plot raw + smoothed loss curve and save to out_path.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, alpha=0.3, color="#4C72B0", linewidth=0.8, label="raw")
    ax.plot(steps, smooth(losses, smoothing), color="#4C72B0", linewidth=2.0, label="smoothed")

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Cross-entropy loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate final loss
    ax.annotate(
        f"Final: {losses[-1]:.3f}",
        xy=(steps[-1], losses[-1]),
        xytext=(-60, 15),
        textcoords="offset points",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_perplexity(
    steps: np.ndarray,
    losses: np.ndarray,
    out_path: str,
) -> None:
    ppl = np.exp(losses)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, ppl, alpha=0.3, color="#DD8452", linewidth=0.8)
    ax.plot(steps, np.exp(smooth(losses)), color="#DD8452", linewidth=2.0)
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title("XSAKE Perplexity — OpenWebText", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def from_csv(csv_path: str, out_dir: str) -> None:
    import csv
    steps, losses = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))

    steps  = np.array(steps)
    losses = np.array(losses)
    os.makedirs(out_dir, exist_ok=True)
    plot_loss_curve(steps, losses, os.path.join(out_dir, "training_loss.png"))
    plot_perplexity(steps, losses, os.path.join(out_dir, "perplexity.png"))


def from_wandb(run_path: str, out_dir: str) -> None:
    import wandb
    api  = wandb.Api()
    run  = api.run(run_path)
    hist = run.scan_history(keys=["step", "loss"])
    rows = [(r["step"], r["loss"]) for r in hist if r.get("loss") is not None]
    steps  = np.array([r[0] for r in rows])
    losses = np.array([r[1] for r in rows])
    os.makedirs(out_dir, exist_ok=True)
    plot_loss_curve(steps, losses, os.path.join(out_dir, "training_loss.png"))
    plot_perplexity(steps, losses, os.path.join(out_dir, "perplexity.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",   help="CSV file with step,loss columns")
    parser.add_argument("--wandb", help="W&B run path (entity/project/run_id)")
    parser.add_argument("--out",   default="data/loss_curves/", help="output directory")
    args = parser.parse_args()

    if args.log:
        from_csv(args.log, args.out)
    elif args.wandb:
        from_wandb(args.wandb, args.out)
    else:
        print("Provide --log <csv> or --wandb <run_path>")
