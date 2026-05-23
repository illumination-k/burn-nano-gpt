#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib>=3.8",
#     "pandas>=2.1",
# ]
# ///
"""Plot per-step training loss from the CSV written by the `train` binary.

Run after `BURN_NANO_GPT_LOSS_CSV=docs/loss.csv ./target/release/train`:

    uv run docs/plot_loss.py docs/loss.csv docs/loss_curve.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <loss.csv> <out.png>")
    csv_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    df = pd.read_csv(csv_path)
    # Smooth a little so the trend is readable next to the noisy per-step values.
    df["smoothed"] = df["loss"].rolling(window=50, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    ax.plot(df["step"], df["loss"], alpha=0.25, color="#888", label="per-step")
    ax.plot(df["step"], df["smoothed"], color="#1f77b4", linewidth=2.0, label="50-step mean")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("nano-GPT on TinyShakespeare (char-level)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
