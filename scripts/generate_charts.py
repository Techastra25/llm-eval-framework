"""
Generates the PNG charts embedded in the README, from the actual
results/eval_results.csv produced by run_demo.py -- these are not
mockups, they are plots of the real (mock-model) run's output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "eval_results.csv"
IMG_DIR = ROOT / "docs" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

NAVY = "#1F3864"
ACCENT = "#2E86AB"
WARN = "#D9534F"
OK = "#4C9A6A"


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    _plot_overall_score_by_model(df)
    _plot_latency_vs_cost(df)
    _plot_category_breakdown(df)
    print(f"Charts written to {IMG_DIR}")


def _plot_overall_score_by_model(df: pd.DataFrame) -> None:
    grouped = df.groupby("model_name")["rubric_overall"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(grouped.index, grouped.values, color=[NAVY, ACCENT, WARN])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Average Rubric Score (0-1)")
    ax.set_title("Model Quality Comparison — Rubric Judge Overall Score")
    for bar, val in zip(bars, grouped.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "model_quality_comparison.png", dpi=150)
    plt.close(fig)


def _plot_latency_vs_cost(df: pd.DataFrame) -> None:
    grouped = df.groupby("model_name").agg(
        avg_latency_ms=("latency_ms", "mean"),
        total_cost_usd=("cost_usd", "sum"),
    )

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax2 = ax1.twinx()

    x = range(len(grouped))
    ax1.bar([i - 0.2 for i in x], grouped["avg_latency_ms"], width=0.4, label="Avg Latency (ms)", color=ACCENT)
    ax2.bar([i + 0.2 for i in x], grouped["total_cost_usd"], width=0.4, label="Total Cost (USD)", color=OK)

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(grouped.index, rotation=10)
    ax1.set_ylabel("Avg Latency (ms)", color=ACCENT)
    ax2.set_ylabel("Total Cost (USD)", color=OK)
    ax1.set_title("Latency vs. Cost by Model")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "latency_vs_cost.png", dpi=150)
    plt.close(fig)


def _plot_category_breakdown(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(index="category", columns="model_name", values="rubric_overall", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    pivot.plot(kind="bar", ax=ax, color=[NAVY, ACCENT, WARN])
    ax.set_ylabel("Avg Rubric Score")
    ax.set_title("Performance by Question Category")
    ax.set_ylim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=15)
    plt.legend(title="Model", fontsize=8)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "category_breakdown.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
