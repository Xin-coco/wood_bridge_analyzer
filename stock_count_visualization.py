from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import tempfile

_cache_root = Path(tempfile.gettempdir()) / "wood_bridge_analyzer_cache"
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg"))

import matplotlib.pyplot as plt
import pandas as pd


def plot_length_rounding_distribution(rounded_df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    included = rounded_df[rounded_df["include_in_stock_count"] == True]  # noqa: E712
    fig, ax = plt.subplots(figsize=(9, 5))
    if included.empty:
        ax.text(0.5, 0.5, "No included wood members", ha="center", va="center")
        ax.axis("off")
    else:
        bins = min(16, max(6, int(len(included) ** 0.5) + 4))
        ax.hist(included["original_length_mm"], bins=bins, alpha=0.55, label="original", color="#4c78a8")
        ax.hist(included["rounded_length_mm"], bins=bins, alpha=0.45, label="rounded", color="#f58518")
        ax.set_xlabel("Length / mm")
        ax.set_ylabel("Count")
        ax.set_title("Length Rounding Distribution")
        ax.legend()
        ax.grid(True, axis="y", lw=0.2, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "length_rounding_distribution.png", dpi=dpi)
    plt.close(fig)


def plot_stock_pairing_plan(plan_df: pd.DataFrame, output_dir: Path, dpi: int, stock_length_mm: float = 1300.0) -> None:
    counted = plan_df[plan_df["pairing_type"] != "ignored"].copy() if not plan_df.empty else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.25 * max(len(counted), 1))))
    if counted.empty:
        ax.text(0.5, 0.5, "No stock pairing plan", ha="center", va="center")
        ax.axis("off")
    else:
        y_positions = range(len(counted))
        colors = ["#4c78a8", "#f58518"]
        for y, row in zip(y_positions, counted.itertuples()):
            start = 0.0
            lengths = [float(x) for x in str(row.member_lengths_mm).split(";") if x]
            ids = [x for x in str(row.member_ids).split(";") if x]
            for idx, length in enumerate(lengths):
                ax.barh(y, length, left=start, color=colors[idx % len(colors)], edgecolor="white")
                if length >= 120:
                    ax.text(start + length / 2, y, ids[idx], ha="center", va="center", fontsize=7, color="white")
                start += length
            waste = max(0.0, stock_length_mm - start)
            if waste > 0:
                ax.barh(y, waste, left=start, color="#dddddd", edgecolor="white")
            ax.text(stock_length_mm + 20, y, str(row.pairing_type), va="center", fontsize=7)
        ax.set_xlim(0, stock_length_mm + 260)
        ax.set_yticks(list(y_positions))
        ax.set_yticklabels([str(int(x)) for x in counted["stock_id"].tolist()], fontsize=7)
        ax.set_xlabel("Used length per 1300mm stock / mm")
        ax.set_ylabel("Stock ID")
        ax.set_title("Stock Pairing Plan")
        ax.grid(True, axis="x", lw=0.2, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "stock_pairing_plan.png", dpi=dpi)
    plt.close(fig)


def plot_material_stock_count_summary(summary: dict[str, Any], output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["program", "manual", "base"]
    values = [summary.get("stock_wood_count", 0), summary.get("manual_stock_count", 0), summary.get("base_stock_count", 60)]
    axes[0].bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[0].set_title("Stock Wood Count")
    axes[0].set_ylabel("Count")
    score = summary.get("capped_material_score", summary.get("raw_material_score", 0))
    waste = summary.get("average_waste_mm", 0)
    pair_rate = summary.get("pair_success_rate", 0)
    axes[1].bar(["score", "avg waste", "pair rate %"], [score, waste, pair_rate * 100], color=["#4c78a8", "#d62728", "#54a24b"])
    axes[1].set_title("Score / Waste / Pairing")
    axes[1].grid(True, axis="y", lw=0.2, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "material_stock_count_summary.png", dpi=dpi)
    plt.close(fig)


def create_stock_count_visualizations(rounded_df: pd.DataFrame, plan_df: pd.DataFrame, summary: dict[str, Any], output_dir: Path, config: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    dpi = int(config.get("visualization", {}).get("dpi", 180))
    tasks = [
        ("length_rounding_distribution.png", lambda: plot_length_rounding_distribution(rounded_df, output_dir, dpi)),
        ("stock_pairing_plan.png", lambda: plot_stock_pairing_plan(plan_df, output_dir, dpi, float(summary.get("stock_length_mm", 1300.0)))),
        ("material_stock_count_summary.png", lambda: plot_material_stock_count_summary(summary, output_dir, dpi)),
    ]
    for name, func in tasks:
        try:
            func()
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    return failures
