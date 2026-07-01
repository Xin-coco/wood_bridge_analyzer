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
import numpy as np
import pandas as pd

from bridge_parser import Rod
from fem_truss_solver import TrussModel, LoadCaseResult


def _draw_members_2d(ax, rods: list[Rod], axes: tuple[int, int], title: str) -> None:
    labels = ["X", "Y", "Z"]
    for rod in rods:
        ax.plot([rod.start[axes[0]], rod.end[axes[0]]], [rod.start[axes[1]], rod.end[axes[1]]], color="#555", lw=1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{labels[axes[0]]} / mm")
    ax.set_ylabel(f"{labels[axes[1]]} / mm")
    ax.set_title(title)
    ax.grid(True, lw=0.2, alpha=0.4)


def plot_three_views(rods: list[Rod], output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    _draw_members_2d(axes[0], rods, (0, 1), "Plan X-Y")
    _draw_members_2d(axes[1], rods, (0, 2), "Elevation X-Z")
    _draw_members_2d(axes[2], rods, (1, 2), "Section Y-Z")
    fig.tight_layout()
    fig.savefig(output_dir / "bridge_3views.png", dpi=dpi)
    plt.close(fig)


def plot_rod_count(rods: list[Rod], output_dir: Path, dpi: int) -> None:
    lengths = np.array([r.length_mm for r in rods])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(lengths, bins=min(12, max(4, len(rods) // 3)), color="#607d8b")
    axes[0].set_title("Rod Length Distribution")
    axes[0].set_xlabel("Length / mm")
    axes[0].set_ylabel("Count")
    dirs = np.array([np.argmax(np.abs(r.direction)) for r in rods])
    axes[1].bar(["X-main", "Y-main", "Z-main"], [int(np.sum(dirs == i)) for i in range(3)], color=["#4c78a8", "#f58518", "#54a24b"])
    axes[1].set_title("Main Direction Count")
    fig.tight_layout()
    fig.savefig(output_dir / "rod_count_diagram.png", dpi=dpi)
    plt.close(fig)


def plot_force_diagram(model: TrussModel, result: LoadCaseResult, output_dir: Path, dpi: int) -> None:
    if result.member_results.empty:
        _plot_message(output_dir / "force_diagram.png", "Force diagram unavailable", result.message, dpi)
        return
    force_map = {int(row.member_id): row.axial_force_n for row in result.member_results.itertuples()}
    max_force = max([abs(v) for v in force_map.values()] + [1.0])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for member in model.members:
        p1 = model.nodes[member["node_i"]]
        p2 = model.nodes[member["node_j"]]
        force = force_map.get(member["member_id"], 0.0)
        color = "#d62728" if force >= 0 else "#1f77b4"
        lw = 0.6 + 3.0 * abs(force) / max_force
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, lw=lw)
    ax.set_title(f"Axial Force: {result.name}")
    ax.set_xlabel("X / mm")
    ax.set_ylabel("Y / mm")
    ax.set_zlabel("Z / mm")
    fig.tight_layout()
    fig.savefig(output_dir / "force_diagram.png", dpi=dpi)
    plt.close(fig)


def plot_deflection(model: TrussModel, result: LoadCaseResult, output_dir: Path, dpi: int, scale: float) -> None:
    displaced = model.nodes + result.displacements.reshape((-1, 3)) * scale
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for member in model.members:
        i, j = member["node_i"], member["node_j"]
        p1, p2 = model.nodes[i], model.nodes[j]
        d1, d2 = displaced[i], displaced[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="#999", lw=0.7, alpha=0.55)
        ax.plot([d1[0], d2[0]], [d1[1], d2[1]], [d1[2], d2[2]], color="#e45756", lw=1.0)
    ax.set_title(f"Deflection x{scale:g}: {result.name}")
    ax.set_xlabel("X / mm")
    ax.set_ylabel("Y / mm")
    ax.set_zlabel("Z / mm")
    fig.tight_layout()
    fig.savefig(output_dir / "deflection_diagram.png", dpi=dpi)
    plt.close(fig)


def _plot_message(path: Path, title: str, message: str, dpi: int) -> None:
    ascii_message = message.encode("ascii", errors="ignore").decode("ascii") or "See analysis_report.md for details."
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=14, weight="bold")
    ax.text(0.5, 0.42, ascii_message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_node_debug(model: TrussModel, output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for member in model.members:
        p1 = model.nodes[member["node_i"]]
        p2 = model.nodes[member["node_j"]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#777", lw=0.8, alpha=0.7)
    for idx, p in enumerate(model.nodes):
        count = len(model.node_members.get(idx, []))
        color = "#d62728" if count <= 1 else ("#ff9800" if count == 2 else "#2e7d32")
        ax.scatter(p[0], p[1], s=28, color=color, zorder=3)
        ax.text(p[0], p[1], f"{idx}({count})", fontsize=7, color="#111")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X / mm")
    ax.set_ylabel("Y / mm")
    ax.set_title("Node Debug: node_id(member_count)")
    ax.grid(True, lw=0.2, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "node_debug.png", dpi=dpi)
    plt.close(fig)


def plot_buckling_risk_map(model: TrussModel, result: LoadCaseResult, output_dir: Path, dpi: int) -> None:
    if result.member_results.empty:
        _plot_message(output_dir / "buckling_risk_map.png", "Buckling risk map unavailable", result.message, dpi)
        return
    risk_map = {int(row.member_id): (row.risk_score, row.risk_level) for row in result.member_results.itertuples()}
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for member in model.members:
        p1 = model.nodes[member["node_i"]]
        p2 = model.nodes[member["node_j"]]
        risk, level = risk_map.get(member["member_id"], (0.0, "low"))
        color = {"低风险": "#4caf50", "中风险": "#ff9800", "高风险": "#d62728", "危险": "#7f0000", "拉杆": "#9e9e9e"}.get(level, "#666")
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, lw=0.8 + 2.5 * min(risk, 1.5))
    ax.set_title("Buckling Risk Map")
    ax.set_xlabel("X / mm")
    ax.set_ylabel("Y / mm")
    ax.set_zlabel("Z / mm")
    fig.tight_layout()
    fig.savefig(output_dir / "buckling_risk_map.png", dpi=dpi)
    plt.close(fig)


def plot_load_case_comparison(load_case_df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    if load_case_df.empty:
        _plot_message(output_dir / "load_case_comparison.png", "Load comparison unavailable", "No load case results.", dpi)
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = load_case_df["case"].tolist()
    values = load_case_df["max_vertical_displacement_mm"].tolist()
    colors = ["#4c78a8" if ok else "#bdbdbd" for ok in load_case_df["success"].tolist()]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Max vertical displacement / mm")
    ax.set_title("Load Case Comparison")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", lw=0.2, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "load_case_comparison.png", dpi=dpi)
    plt.close(fig)


def plot_moving_load(moving_df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(moving_df["position_ratio"], moving_df["max_vertical_displacement_mm"], color="#d62728", marker="o", label="Max vertical displacement")
    ax1.set_xlabel("Load position along span")
    ax1.set_ylabel("Displacement / mm", color="#d62728")
    ax2 = ax1.twinx()
    ax2.plot(moving_df["position_ratio"], moving_df["max_abs_force_n"], color="#1f77b4", marker="s", label="Max member force")
    ax2.set_ylabel("Force / N", color="#1f77b4")
    ax1.grid(True, lw=0.2, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "moving_load_envelope.png", dpi=dpi)
    plt.close(fig)


def plot_report_summary(rods: list[Rod], model: TrussModel, governing: LoadCaseResult, rod_summary: dict[str, Any], output_dir: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    _draw_members_2d(ax1, rods, (0, 1), "Plan")
    _draw_members_2d(ax2, rods, (0, 2), "Elevation")
    _draw_members_2d(ax3, rods, (1, 2), "Section")
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.axis("off")
    worst = "N/A"
    if not governing.member_results.empty:
        row = governing.member_results.sort_values("risk_score", ascending=False).iloc[0]
        worst = f"member {int(row.member_id)}\nrisk {row.risk_score:.3f}\n{row.force_type}"
    summary = (
        f"Load case: {governing.name}\n"
        f"Max deflection: {governing.max_vertical_displacement_mm:.3f} mm\n"
        f"Max force: {governing.max_abs_force_n:.1f} N\n"
        f"Worst member: {worst}\n"
        f"Model rods: {rod_summary['model_rod_count']}\n"
        f"Stock rods: {rod_summary.get('stock_wood_count', rod_summary['equivalent_standard_count'])}\n"
        f"Material score: {rod_summary.get('raw_material_score', rod_summary['material_score'])}"
    )
    ax4.text(0.02, 0.98, summary, va="top", fontsize=11)
    ax5 = fig.add_subplot(2, 3, 5)
    counts = [rod_summary["model_rod_count"], rod_summary.get("stock_wood_count", rod_summary["equivalent_standard_count"]), rod_summary["base_count"]]
    ax5.bar(["model", "stock", "base"], counts, color=["#4c78a8", "#f58518", "#54a24b"])
    ax5.set_title("Rod Count Overview")
    ax6 = fig.add_subplot(2, 3, 6)
    if not governing.member_results.empty:
        top = governing.member_results.sort_values("risk_score", ascending=False).head(5)
        ax6.barh(top["member_id"].astype(str), top["risk_score"], color="#d62728")
        ax6.invert_yaxis()
        ax6.set_title("Top 5 Risk")
    else:
        ax6.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "report_summary.png", dpi=dpi)
    plt.close(fig)


def plot_top_10_risk_members(result: LoadCaseResult, output_dir: Path, dpi: int) -> None:
    if result.member_results.empty:
        _plot_message(output_dir / "top_10_risk_members.png", "Top 10 risk unavailable", result.message, dpi)
        return
    top = result.member_results.sort_values("risk_score", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["member_id"].astype(str), top["risk_score"], color="#d62728")
    ax.invert_yaxis()
    ax.set_xlabel("Risk score")
    ax.set_ylabel("Member ID")
    ax.set_title("Top 10 Risk Members")
    fig.tight_layout()
    fig.savefig(output_dir / "top_10_risk_members.png", dpi=dpi)
    plt.close(fig)


def plot_support_and_loads(model: TrussModel, supports: list[dict[str, Any]], deck_node_ids: list[int], output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for member in model.members:
        p1 = model.nodes[member["node_i"]]
        p2 = model.nodes[member["node_j"]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#999", lw=0.8)
    deck_set = set(deck_node_ids)
    for idx, p in enumerate(model.nodes):
        if idx in deck_set:
            ax.scatter(p[0], p[1], color="#f58518", s=35, label="deck/load" if deck_node_ids and idx == deck_node_ids[0] else None)
    for support in supports:
        idx = int(support["node_id"])
        p = model.nodes[idx]
        ax.scatter(p[0], p[1], marker="^", color="#1f77b4", s=80, label="support")
        ax.text(p[0], p[1], f"S{idx}", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Supports and Load Nodes")
    ax.set_xlabel("X / mm")
    ax.set_ylabel("Y / mm")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())
    fig.tight_layout()
    fig.savefig(output_dir / "support_and_loads.png", dpi=dpi)
    plt.close(fig)


def plot_node_connectivity_map(model: TrussModel, output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = np.array([len(model.node_members.get(i, [])) for i in range(len(model.nodes))], dtype=float)
    for member in model.members:
        p1 = model.nodes[member["node_i"]]
        p2 = model.nodes[member["node_j"]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#ccc", lw=0.7)
    sc = ax.scatter(model.nodes[:, 0], model.nodes[:, 1], c=counts, cmap="viridis", s=55, zorder=3)
    for idx, p in enumerate(model.nodes):
        ax.text(p[0], p[1], str(int(counts[idx])), fontsize=7)
    fig.colorbar(sc, ax=ax, label="member count")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Node Connectivity Map")
    ax.set_xlabel("X / mm")
    ax.set_ylabel("Y / mm")
    fig.tight_layout()
    fig.savefig(output_dir / "node_connectivity_map.png", dpi=dpi)
    plt.close(fig)


def plot_clean_centerline_3views(model: TrussModel, output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    views = [((0, 1), "Clean Plan X-Y"), ((0, 2), "Clean Elevation X-Z"), ((1, 2), "Clean Section Y-Z")]
    labels = ["X", "Y", "Z"]
    for ax, (axes_idx, title) in zip(axes, views):
        for member in model.members:
            p1 = model.nodes[member["node_i"]]
            p2 = model.nodes[member["node_j"]]
            ax.plot([p1[axes_idx[0]], p2[axes_idx[0]]], [p1[axes_idx[1]], p2[axes_idx[1]]], color="#555", lw=1.0)
        ax.scatter(model.nodes[:, axes_idx[0]], model.nodes[:, axes_idx[1]], s=12, color="#1f77b4", zorder=3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"{labels[axes_idx[0]]} / mm")
        ax.set_ylabel(f"{labels[axes_idx[1]]} / mm")
        ax.set_title(title)
        ax.grid(True, lw=0.2, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "clean_centerline_3views.png", dpi=dpi)
    plt.close(fig)


def plot_node_quality_map(model: TrussModel, diagnostics: dict[str, Any], output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    dangling = set(int(x) for x in diagnostics.get("dangling_members", []))
    single_nodes = set(int(x) for x in diagnostics.get("single_member_nodes", []))
    abnormal = set(int(x) for x in diagnostics.get("abnormal_nodes", []))
    for member in model.members:
        p1 = model.nodes[member["node_i"]]
        p2 = model.nodes[member["node_j"]]
        is_bad = int(member["member_id"]) in dangling
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#d62728" if is_bad else "#bdbdbd", lw=1.4 if is_bad else 0.7)
    for idx, p in enumerate(model.nodes):
        count = len(model.node_members.get(idx, []))
        if idx in single_nodes:
            color = "#d62728"
        elif idx in abnormal:
            color = "#f58518"
        else:
            color = "#2e7d32"
        ax.scatter(p[0], p[1], s=40, color=color, zorder=3)
        ax.text(p[0], p[1], f"{idx}:{count}", fontsize=7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X / mm")
    ax.set_ylabel("Y / mm")
    ax.set_title("Node Quality Map: node_id:member_count")
    ax.grid(True, lw=0.2, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "node_quality_map.png", dpi=dpi)
    plt.close(fig)


def plot_support_load_manual_check(
    model: TrussModel,
    supports: list[dict[str, Any]],
    deck_node_ids: list[int],
    support_load_info: dict[str, Any],
    output_dir: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for member in model.members:
        p1 = model.nodes[member["node_i"]]
        p2 = model.nodes[member["node_j"]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#c7c7c7", lw=0.7)
    deck_set = set(int(x) for x in deck_node_ids)
    for idx in deck_set:
        if 0 <= idx < len(model.nodes):
            p = model.nodes[idx]
            ax.scatter(p[0], p[1], color="#f58518", s=55, label="deck/load")
            ax.text(p[0], p[1], f"L{idx}", fontsize=8)
    for support in supports:
        idx = int(support["node_id"])
        if 0 <= idx < len(model.nodes):
            p = model.nodes[idx]
            ax.scatter(p[0], p[1], marker="^", color="#1f77b4", s=90, label="support")
            ax.text(p[0], p[1], f"S{idx}", fontsize=8)
    status = support_load_info.get("message", "")
    if not support_load_info.get("fixed_nodes_confirmed") or not support_load_info.get("roller_nodes_confirmed") or not support_load_info.get("deck_nodes_confirmed"):
        status = "Manual support and deck load nodes required before FEM."
    else:
        status = "Manual support and deck load nodes confirmed."
    ax.text(
        0.02,
        0.98,
        status,
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X / mm")
    ax.set_ylabel("Y / mm")
    ax.set_title("Support and Load Manual Check")
    ax.grid(True, lw=0.2, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "support_load_manual_check.png", dpi=dpi)
    plt.close(fig)


def plot_disconnected_members(model: TrussModel, diagnostics: dict[str, Any], output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    dangling = set(int(x) for x in diagnostics.get("dangling_members", []))
    for member in model.members:
        p1 = model.nodes[member["node_i"]]
        p2 = model.nodes[member["node_j"]]
        if int(member["member_id"]) in dangling:
            color, lw, alpha = "#d62728", 2.0, 1.0
            mid = (p1 + p2) / 2.0
            ax.text(mid[0], mid[1], str(member["member_id"]), fontsize=7, color="#7f0000")
        else:
            color, lw, alpha = "#dddddd", 0.6, 0.45
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, alpha=alpha)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X / mm")
    ax.set_ylabel("Y / mm")
    ax.set_title("Disconnected / Dangling Members")
    ax.grid(True, lw=0.2, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "disconnected_members.png", dpi=dpi)
    plt.close(fig)


def plot_connectivity_map(model: TrussModel, components: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    node_component: dict[int, int] = {}
    for comp in components:
        for node_id in comp.get("node_ids", []):
            node_component[int(node_id)] = int(comp["component_id"])
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(components))))
    fig, ax = plt.subplots(figsize=(10, 6))
    for member in model.members:
        ci = node_component.get(int(member["node_i"]), 0) - 1
        color = colors[ci % len(colors)] if ci >= 0 else "#999999"
        p1 = model.nodes[member["node_i"]]
        p2 = model.nodes[member["node_j"]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=1.1)
    for idx, p in enumerate(model.nodes):
        ci = node_component.get(idx, 0) - 1
        ax.scatter(p[0], p[1], color=colors[ci % len(colors)] if ci >= 0 else "#999999", s=28, zorder=3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X / mm")
    ax.set_ylabel("Y / mm")
    ax.set_title(f"Connectivity Map: {len(components)} component(s)")
    ax.grid(True, lw=0.2, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "connectivity_map.png", dpi=dpi)
    plt.close(fig)


def plot_deck_node_map(model: TrussModel, deck_node_ids: list[int], supports: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    support_ids = {int(s["node_id"]) for s in supports}
    deck = {int(i) for i in deck_node_ids}
    fig, ax = plt.subplots(figsize=(10, 6))
    for member in model.members:
        p1 = model.nodes[member["node_i"]]
        p2 = model.nodes[member["node_j"]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#d0d0d0", lw=0.7)
    for idx, p in enumerate(model.nodes):
        if idx in deck:
            color, size, label = "#f58518", 60, "deck/load"
        elif idx in support_ids:
            color, size, label = "#1f77b4", 50, "support"
        else:
            color, size, label = "#aaaaaa", 14, None
        ax.scatter(p[0], p[1], color=color, s=size, zorder=3, label=label)
        if idx in deck:
            ax.text(p[0], p[1], f"D{idx}", fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X / mm")
    ax.set_ylabel("Y / mm")
    ax.set_title("Deck Node Check Map")
    ax.grid(True, lw=0.2, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "deck_node_map.png", dpi=dpi)
    plt.close(fig)


def plot_support_node_map(model: TrussModel, supports: list[dict[str, Any]], deck_node_ids: list[int], output_dir: Path, dpi: int) -> None:
    support_by_id = {int(s["node_id"]): s for s in supports}
    deck = {int(i) for i in deck_node_ids}
    fig, ax = plt.subplots(figsize=(10, 6))
    for member in model.members:
        p1 = model.nodes[member["node_i"]]
        p2 = model.nodes[member["node_j"]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#d0d0d0", lw=0.7)
    for idx, p in enumerate(model.nodes):
        if idx in support_by_id:
            marker = "^" if "Uy" in support_by_id[idx].get("restraints", []) else "s"
            color, size, label = "#1f77b4", 80, "support"
        elif idx in deck:
            marker = "o"
            color, size, label = "#f58518", 40, "deck/load"
        else:
            marker = "o"
            color, size, label = "#aaaaaa", 14, None
        ax.scatter(p[0], p[1], color=color, s=size, marker=marker, zorder=3, label=label)
        if idx in support_by_id:
            ax.text(p[0], p[1], f"S{idx}", fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X / mm")
    ax.set_ylabel("Y / mm")
    ax.set_title("Support Node Check Map")
    ax.grid(True, lw=0.2, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "support_node_map.png", dpi=dpi)
    plt.close(fig)


def plot_centerline_review_board(
    model: TrussModel,
    diagnostics: dict[str, Any],
    supports: list[dict[str, Any]],
    deck_node_ids: list[int],
    components: list[dict[str, Any]],
    score: dict[str, Any],
    output_dir: Path,
    dpi: int,
) -> None:
    single_nodes = set(int(x) for x in diagnostics.get("single_member_nodes", []))
    dangling = set(int(x) for x in diagnostics.get("dangling_members", []))
    abnormal = set(int(x) for x in diagnostics.get("abnormal_nodes", []))
    support_ids = {int(s["node_id"]) for s in supports}
    deck = {int(i) for i in deck_node_ids}
    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    for ax, axes_idx, title in [(ax1, (0, 1), "Plan"), (ax2, (0, 2), "Elevation"), (ax3, (1, 2), "Section")]:
        for member in model.members:
            p1 = model.nodes[member["node_i"]]
            p2 = model.nodes[member["node_j"]]
            is_dangling = int(member["member_id"]) in dangling
            ax.plot([p1[axes_idx[0]], p2[axes_idx[0]]], [p1[axes_idx[1]], p2[axes_idx[1]]], color="#d62728" if is_dangling else "#777777", lw=1.4 if is_dangling else 0.7)
        for idx, p in enumerate(model.nodes):
            color = "#d62728" if idx in single_nodes or idx in abnormal else "#2e7d32"
            if idx in deck:
                color = "#f58518"
            if idx in support_ids:
                color = "#1f77b4"
            ax.scatter(p[axes_idx[0]], p[axes_idx[1]], color=color, s=24, zorder=3)
            if idx in deck or idx in support_ids or idx in single_nodes:
                prefix = "D" if idx in deck else ("S" if idx in support_ids else "")
                ax.text(p[axes_idx[0]], p[axes_idx[1]], f"{prefix}{idx}", fontsize=7)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.grid(True, lw=0.2, alpha=0.4)
    ax4 = fig.add_subplot(2, 3, 4)
    counts = [len(model.nodes), len(model.members), len(single_nodes), len(dangling), len(components)]
    ax4.bar(["nodes", "members", "single", "dangling", "comp."], counts, color=["#4c78a8", "#54a24b", "#d62728", "#ff9800", "#b279a2"])
    ax4.set_title("Validation Counts")
    ax4.tick_params(axis="x", rotation=20)
    ax5 = fig.add_subplot(2, 3, 5)
    parts = [
        score.get("geometry_score", 0),
        score.get("node_connection_score", 0),
        score.get("support_definition_score", 0),
        score.get("load_definition_score", 0),
        score.get("material_count_score", 0),
        score.get("fem_solvability_score", 0),
    ]
    ax5.bar(["geo", "node", "support", "load", "material", "fem"], parts, color="#4c78a8")
    ax5.set_ylim(0, 25)
    ax5.set_title("Score Breakdown")
    ax5.tick_params(axis="x", rotation=20)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    text = (
        f"Centerline model score: {score.get('centerline_model_score', 0)}/100\n"
        f"Reliable FEM ready: {'YES' if score.get('can_enter_reliable_fem') else 'NO'}\n\n"
        f"Components: {len(components)}\n"
        f"Single-member nodes: {len(single_nodes)}\n"
        f"Dangling members: {len(dangling)}\n"
        f"Supports: {len(support_ids)}\n"
        f"Deck/load nodes: {len(deck)}"
    )
    ax6.text(0.02, 0.98, text, va="top", fontsize=12)
    fig.suptitle("Centerline Model Review Board", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "centerline_review_board.png", dpi=dpi)
    plt.close(fig)


def plot_fix_suggestions(model: TrussModel, suggestions: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for member in model.members:
        p1 = model.nodes[member["node_i"]]
        p2 = model.nodes[member["node_j"]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#888", lw=0.8)
    colors = ["#d62728", "#f58518", "#4c78a8", "#54a24b", "#b279a2", "#eeca3b"]
    for item in suggestions:
        target = item.get("target")
        if isinstance(target, list):
            for value in target[:10]:
                try:
                    value_int = int(value)
                except Exception:
                    continue
                for member in model.members:
                    if int(member["member_id"]) == value_int:
                        p1 = model.nodes[member["node_i"]]
                        p2 = model.nodes[member["node_j"]]
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=colors[(item["priority"] - 1) % len(colors)], lw=2.2)
                if 0 <= value_int < len(model.nodes):
                    p = model.nodes[value_int]
                    ax.scatter(p[0], p[1], color=colors[(item["priority"] - 1) % len(colors)], s=70)
                    ax.text(p[0], p[1], f"P{item['priority']}", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Fix Suggestions Map")
    ax.set_xlabel("X / mm")
    ax.set_ylabel("Y / mm")
    fig.tight_layout()
    fig.savefig(output_dir / "fix_suggestions.png", dpi=dpi)
    plt.close(fig)


def plot_final_review_board(
    rods: list[Rod],
    model: TrussModel,
    governing: LoadCaseResult,
    suggestions: list[dict[str, Any]],
    rod_summary: dict[str, Any],
    geom: dict[str, Any],
    output_dir: Path,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(16, 10))
    axes = [fig.add_subplot(2, 3, i + 1) for i in range(3)]
    _draw_members_2d(axes[0], rods, (0, 1), "Plan")
    _draw_members_2d(axes[1], rods, (0, 2), "Elevation")
    _draw_members_2d(axes[2], rods, (1, 2), "Section")

    ax = fig.add_subplot(2, 3, 4)
    ax.set_title("Critical Members")
    for member in model.members:
        p1, p2 = model.nodes[member["node_i"]], model.nodes[member["node_j"]]
        ax.plot([p1[0], p2[0]], [p1[2], p2[2]], color="#bbb", lw=0.6)
    if not governing.member_results.empty:
        comp = governing.member_results[governing.member_results["force_type"] == "compression"].sort_values("axial_force_n", ascending=True).head(1)
        tens = governing.member_results[governing.member_results["force_type"] == "tension"].sort_values("axial_force_n", ascending=False).head(1)
        risk = governing.member_results.sort_values("risk_score", ascending=False).head(5)
        for df, color, label in [(comp, "#1f77b4", "max compression"), (tens, "#d62728", "max tension"), (risk, "#ff9800", "buckling/risk")]:
            for row in df.itertuples():
                for member in model.members:
                    if int(member["member_id"]) == int(row.member_id):
                        p1, p2 = model.nodes[member["node_i"]], model.nodes[member["node_j"]]
                        ax.plot([p1[0], p2[0]], [p1[2], p2[2]], color=color, lw=2.6, label=label)
        disp = governing.displacements.reshape((-1, 3))
        max_node = int(np.argmax(np.abs(disp[:, 2])))
        p = model.nodes[max_node]
        ax.scatter(p[0], p[2], marker="*", s=160, color="#7f0000", label="max deflection")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X / mm")
    ax.set_ylabel("Z / mm")

    ax2 = fig.add_subplot(2, 3, 5)
    ax2.set_title("Suggested Reinforcement")
    for member in model.members:
        p1, p2 = model.nodes[member["node_i"]], model.nodes[member["node_j"]]
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#bbb", lw=0.6)
    colors = ["#d62728", "#f58518", "#4c78a8", "#54a24b"]
    for item in suggestions[:4]:
        target = item.get("target")
        if isinstance(target, list):
            for value in target[:8]:
                try:
                    idx = int(value)
                except Exception:
                    continue
                if 0 <= idx < len(model.nodes):
                    p = model.nodes[idx]
                    ax2.scatter(p[0], p[1], s=60, color=colors[(item["priority"] - 1) % len(colors)])
                for member in model.members:
                    if int(member["member_id"]) == idx:
                        p1, p2 = model.nodes[member["node_i"]], model.nodes[member["node_j"]]
                        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], color=colors[(item["priority"] - 1) % len(colors)], lw=2.0)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("X / mm")
    ax2.set_ylabel("Y / mm")

    ax3 = fig.add_subplot(2, 3, 6)
    ax3.axis("off")
    span_ok = "OK" if geom["span_ok"] else "REVIEW"
    width_ok = "OK" if geom["width_ok"] else "REVIEW"
    text = (
        f"Rod count: {rod_summary['model_rod_count']}\n"
        f"Stock rods: {rod_summary.get('stock_wood_count', rod_summary['equivalent_standard_count'])}\n"
        f"Material score: {rod_summary.get('raw_material_score', rod_summary['material_score'])}\n\n"
        f"Span: {geom['span_mm']:.0f} mm [{span_ok}]\n"
        f"Width: {geom['width_mm']:.0f} mm [{width_ok}]\n\n"
        f"Load case: {governing.name}\n"
        f"Max deflection: {governing.max_vertical_displacement_mm:.3f} mm\n"
        f"Max force: {governing.max_abs_force_n:.1f} N"
    )
    ax3.text(0.02, 0.98, text, va="top", fontsize=12)
    fig.suptitle("Final Structural Review Board", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "final_review_board.png", dpi=dpi)
    plt.close(fig)


def plot_comparison_summary(rows: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    numeric = []
    for row in rows:
        try:
            float(row["old"])
            float(row["new"])
            numeric.append(row)
        except Exception:
            continue
    label_map = {
        "模型杆件数量": "members",
        "等效标准杆数量": "standard rods",
        "最大位移_mm": "max defl.",
        "最大受压杆": "max comp.",
        "最大受拉杆": "max tens.",
        "屈曲风险杆件数量": "buckling risk",
        "高风险节点数量": "risk nodes",
        "材料成本分": "material score",
    }
    labels = [label_map.get(r["metric"], str(r["metric"])) for r in numeric]
    old_vals = [float(r["old"]) for r in numeric]
    new_vals = [float(r["new"]) for r in numeric]
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, old_vals, width, label="old", color="#9e9e9e")
    ax.bar(x + width / 2, new_vals, width, label="new", color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Model Comparison Summary")
    ax.legend()
    ax.grid(True, axis="y", lw=0.2, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_summary.png", dpi=dpi)
    plt.close(fig)


def create_visualizations(
    rods: list[Rod],
    model: TrussModel,
    governing: LoadCaseResult,
    moving_df: pd.DataFrame,
    load_case_df: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any],
    rod_summary: dict[str, Any] | None = None,
    supports: list[dict[str, Any]] | None = None,
    deck_node_ids: list[int] | None = None,
    suggestions: list[dict[str, Any]] | None = None,
    clean_diagnostics: dict[str, Any] | None = None,
    support_load_info: dict[str, Any] | None = None,
    centerline_components: list[dict[str, Any]] | None = None,
    centerline_score: dict[str, Any] | None = None,
) -> list[str]:
    failures: list[str] = []
    dpi = int(config.get("visualization", {}).get("dpi", 180))
    tasks = [
        ("bridge_3views.png", lambda: plot_three_views(rods, output_dir, dpi)),
        ("node_debug.png", lambda: plot_node_debug(model, output_dir, dpi)),
        ("rod_count_diagram.png", lambda: plot_rod_count(rods, output_dir, dpi)),
        ("force_diagram.png", lambda: plot_force_diagram(model, governing, output_dir, dpi)),
        ("deflection_diagram.png", lambda: plot_deflection(model, governing, output_dir, dpi, float(config["visualization"].get("deflection_scale", 20.0)))),
        ("buckling_risk_map.png", lambda: plot_buckling_risk_map(model, governing, output_dir, dpi)),
        ("load_case_comparison.png", lambda: plot_load_case_comparison(load_case_df, output_dir, dpi)),
        ("moving_load_envelope.png", lambda: plot_moving_load(moving_df, output_dir, dpi)),
    ]
    if rod_summary is not None:
        tasks.append(("report_summary.png", lambda: plot_report_summary(rods, model, governing, rod_summary, output_dir, dpi)))
    tasks.append(("top_10_risk_members.png", lambda: plot_top_10_risk_members(governing, output_dir, dpi)))
    if supports is not None and deck_node_ids is not None:
        tasks.append(("support_and_loads.png", lambda: plot_support_and_loads(model, supports, deck_node_ids, output_dir, dpi)))
    tasks.append(("node_connectivity_map.png", lambda: plot_node_connectivity_map(model, output_dir, dpi)))
    if suggestions is not None:
        tasks.append(("fix_suggestions.png", lambda: plot_fix_suggestions(model, suggestions, output_dir, dpi)))
    if rod_summary is not None and suggestions is not None:
        tasks.append(("final_review_board.png", lambda: plot_final_review_board(rods, model, governing, suggestions, rod_summary, geometry_from_rods(rods, config), output_dir, dpi)))
    if clean_diagnostics is not None:
        tasks.append(("clean_centerline_3views.png", lambda: plot_clean_centerline_3views(model, output_dir, dpi)))
        tasks.append(("node_quality_map.png", lambda: plot_node_quality_map(model, clean_diagnostics, output_dir, dpi)))
        tasks.append(("disconnected_members.png", lambda: plot_disconnected_members(model, clean_diagnostics, output_dir, dpi)))
    if support_load_info is not None:
        tasks.append(
            (
                "support_load_manual_check.png",
                lambda: plot_support_load_manual_check(model, supports or [], deck_node_ids or [], support_load_info, output_dir, dpi),
            )
        )
    if centerline_components is not None:
        tasks.append(("connectivity_map.png", lambda: plot_connectivity_map(model, centerline_components, output_dir, dpi)))
        tasks.append(("deck_node_map.png", lambda: plot_deck_node_map(model, deck_node_ids or [], supports or [], output_dir, dpi)))
        tasks.append(("support_node_map.png", lambda: plot_support_node_map(model, supports or [], deck_node_ids or [], output_dir, dpi)))
    if clean_diagnostics is not None and centerline_components is not None and centerline_score is not None:
        tasks.append(
            (
                "centerline_review_board.png",
                lambda: plot_centerline_review_board(
                    model,
                    clean_diagnostics,
                    supports or [],
                    deck_node_ids or [],
                    centerline_components,
                    centerline_score,
                    output_dir,
                    dpi,
                ),
            )
        )
    for name, func in tasks:
        try:
            func()
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            _plot_message(output_dir / name, f"{name} unavailable", str(exc), dpi)
    return failures


def geometry_from_rods(rods: list[Rod], config: dict[str, Any]) -> dict[str, Any]:
    points = np.vstack([np.vstack([r.start, r.end]) for r in rods])
    span = float(np.max(points[:, 0]) - np.min(points[:, 0]))
    width = float(np.max(points[:, 1]) - np.min(points[:, 1]))
    height = float(np.max(points[:, 2]) - np.min(points[:, 2]))
    bridge = config["bridge"]
    return {
        "span_mm": span,
        "width_mm": width,
        "height_mm": height,
        "span_ok": abs(span - float(bridge["target_span_mm"])) <= float(bridge["target_span_tolerance_mm"]),
        "width_ok": float(bridge["min_deck_width_mm"]) <= width <= float(bridge["max_deck_width_mm"]),
    }
