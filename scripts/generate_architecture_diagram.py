"""Generates docs/images/architecture.png -- a static diagram of the pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "docs" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

NAVY = "#1F3864"
ACCENT = "#2E86AB"
LIGHT = "#EAF1F8"
GRAY = "#555555"


def box(ax, xy, w, h, text, facecolor=LIGHT, edgecolor=NAVY, fontsize=10, textcolor="#111111"):
    rect = mpatches.FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.6, edgecolor=edgecolor, facecolor=facecolor,
    )
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=textcolor, wrap=True)


def arrow(ax, start, end):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.6, shrinkA=2, shrinkB=2),
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 7.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    # Row 1: dataset -> models
    box(ax, (0.3, 5.6), 2.0, 1.0, "Dataset\n(sample_prompts.jsonl)\nreasoning + math tasks", facecolor="#FFF6E5")

    box(ax, (3.0, 6.1), 2.1, 0.7, "MockModel\n(offline demo)", facecolor=LIGHT)
    box(ax, (3.0, 5.35), 2.1, 0.7, "OpenAIModel\n(production)", facecolor=LIGHT)
    box(ax, (3.0, 4.6), 2.1, 0.7, "AnthropicModel\n(production)", facecolor=LIGHT)
    box(ax, (2.75, 4.35), 2.65, 2.65, "", facecolor="none", edgecolor="#999999")
    ax.text(4.075, 7.3, "BaseModelClient (pluggable interface)", ha="center", fontsize=9, color=GRAY, style="italic")

    arrow(ax, (2.3, 6.1), (3.0, 6.45))

    # Row 2: evaluators
    box(ax, (6.0, 6.1), 2.2, 0.9, "Metrics\nexact match, token-F1,\nhallucination, toxicity", facecolor="#EAF6EE")
    box(ax, (6.0, 4.9), 2.2, 0.9, "RubricJudge\ncorrectness / completeness\nclarity / safety", facecolor="#EAF6EE")

    arrow(ax, (5.1, 6.1), (6.0, 6.55))
    arrow(ax, (5.1, 5.0), (6.0, 5.35))

    # Row 3: storage
    box(ax, (8.7, 5.5), 2.0, 1.0, "Storage\n(SQLite)\neval_results.db", facecolor="#F0EAF6", edgecolor=ACCENT)
    arrow(ax, (8.2, 6.4), (8.7, 6.1))
    arrow(ax, (8.2, 5.35), (8.7, 5.8))

    # Row 4: outputs
    box(ax, (6.0, 2.9), 2.2, 0.9, "CSV Export\neval_results.csv", facecolor=LIGHT)
    box(ax, (8.7, 2.9), 2.0, 0.9, "Streamlit Dashboard\ncharts + human review", facecolor=LIGHT)

    arrow(ax, (9.2, 5.5), (9.6, 3.8))
    arrow(ax, (9.0, 5.5), (7.1, 3.8))

    box(ax, (2.9, 2.9), 2.4, 0.9, "generate_charts.py\nmatplotlib PNGs for README", facecolor="#FFF6E5")
    arrow(ax, (6.0, 3.2), (5.3, 3.2))

    # Human-in-the-loop callout
    box(ax, (8.7, 1.3), 2.0, 0.9, "Human Reviewer\nconfirms / overrides\njudge score", facecolor="#FDEBEA", edgecolor="#D9534F")
    arrow(ax, (9.7, 2.9), (9.7, 2.2))
    ax.annotate("", xy=(9.4, 2.9), xytext=(9.4, 2.2),
                arrowprops=dict(arrowstyle="-|>", color="#D9534F", lw=1.4, shrinkA=2, shrinkB=2))

    plt.title("LLM Evaluation & Benchmarking Framework — Architecture", fontsize=13, color=NAVY, pad=26)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "architecture.png", dpi=160)
    plt.close(fig)
    print(f"Saved {IMG_DIR / 'architecture.png'}")


if __name__ == "__main__":
    main()
