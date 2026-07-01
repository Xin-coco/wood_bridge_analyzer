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


def _load_nodes_members(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(output_dir / "clean_nodes.csv"), pd.read_csv(output_dir / "clean_members.csv")


def _plot_message(path: Path, title: str, message: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", fontsize=14, weight="bold")
    ax.text(0.5, 0.42, message, ha="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_support_and_loads_opensees(output_dir: Path, fixed_nodes: list[int], roller_nodes: list[int], deck_nodes: list[int], dpi: int) -> None:
    nodes, members = _load_nodes_members(output_dir)
    lookup = nodes.set_index("node_id")
    fig, ax = plt.subplots(figsize=(10, 6))
    for row in members.itertuples():
        p1 = lookup.loc[int(row.node_i)]
        p2 = lookup.loc[int(row.node_j)]
        ax.plot([p1.x_mm, p2.x_mm], [p1.y_mm, p2.y_mm], color="#cccccc", lw=0.7)
    for node_id in deck_nodes:
        if node_id in lookup.index:
            p = lookup.loc[node_id]
            ax.scatter(p.x_mm, p.y_mm, color="#f58518", s=55, label="deck/load")
            ax.text(p.x_mm, p.y_mm, f"D{node_id}", fontsize=8)
    for node_id in fixed_nodes:
        if node_id in lookup.index:
            p = lookup.loc[node_id]
            ax.scatter(p.x_mm, p.y_mm, color="#1f77b4", marker="^", s=90, label="fixed")
            ax.text(p.x_mm, p.y_mm, f"F{node_id}", fontsize=8)
    for node_id in roller_nodes:
        if node_id in lookup.index:
            p = lookup.loc[node_id]
            ax.scatter(p.x_mm, p.y_mm, color="#4c78a8", marker="s", s=75, label="roller")
            ax.text(p.x_mm, p.y_mm, f"R{node_id}", fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("OpenSees Supports and Loads")
    ax.set_xlabel("X / mm")
    ax.set_ylabel("Y / mm")
    ax.grid(True, lw=0.2, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "support_and_loads_opensees.png", dpi=dpi)
    plt.close(fig)


def plot_opensees_force_diagram(output_dir: Path, force_df: pd.DataFrame, dpi: int) -> None:
    if force_df.empty:
        _plot_message(output_dir / "opensees_force_diagram.png", "OpenSees force diagram unavailable", "No OpenSees member force results.", dpi)
        return
    nodes, members = _load_nodes_members(output_dir)
    lookup = nodes.set_index("node_id")
    case = force_df.sort_values("axial_force_n", key=lambda s: s.abs(), ascending=False).iloc[0]["case"]
    case_forces = force_df[force_df["case"] == case]
    force_map = {int(r.member_id): float(r.axial_force_n) for r in case_forces.itertuples()}
    max_force = max([abs(v) for v in force_map.values()] + [1.0])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for row in members.itertuples():
        p1 = lookup.loc[int(row.node_i)]
        p2 = lookup.loc[int(row.node_j)]
        force = force_map.get(int(row.member_id), 0.0)
        color = "#d62728" if force >= 0 else "#1f77b4"
        lw = 0.7 + 3.0 * abs(force) / max_force
        ax.plot([p1.x_mm, p2.x_mm], [p1.y_mm, p2.y_mm], [p1.z_mm, p2.z_mm], color=color, lw=lw)
    ax.set_title(f"OpenSees Axial Force: {case}")
    ax.set_xlabel("X / mm")
    ax.set_ylabel("Y / mm")
    ax.set_zlabel("Z / mm")
    fig.tight_layout()
    fig.savefig(output_dir / "opensees_force_diagram.png", dpi=dpi)
    plt.close(fig)


def plot_opensees_deflection_diagram(output_dir: Path, disp_df: pd.DataFrame, dpi: int, scale: float = 20.0) -> None:
    if disp_df.empty:
        _plot_message(output_dir / "opensees_deflection_diagram.png", "OpenSees deflection unavailable", "No OpenSees displacement results.", dpi)
        return
    nodes, members = _load_nodes_members(output_dir)
    lookup = nodes.set_index("node_id")
    case = disp_df.sort_values("uz_mm", key=lambda s: s.abs(), ascending=False).iloc[0]["case"]
    case_disp = disp_df[disp_df["case"] == case].set_index("node_id")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for row in members.itertuples():
        p1 = lookup.loc[int(row.node_i)]
        p2 = lookup.loc[int(row.node_j)]
        d1 = case_disp.loc[int(row.node_i)] if int(row.node_i) in case_disp.index else None
        d2 = case_disp.loc[int(row.node_j)] if int(row.node_j) in case_disp.index else None
        ax.plot([p1.x_mm, p2.x_mm], [p1.y_mm, p2.y_mm], [p1.z_mm, p2.z_mm], color="#999999", lw=0.7, alpha=0.5)
        if d1 is not None and d2 is not None:
            ax.plot(
                [p1.x_mm + d1.ux_mm * scale, p2.x_mm + d2.ux_mm * scale],
                [p1.y_mm + d1.uy_mm * scale, p2.y_mm + d2.uy_mm * scale],
                [p1.z_mm + d1.uz_mm * scale, p2.z_mm + d2.uz_mm * scale],
                color="#e45756",
                lw=1.0,
            )
    ax.set_title(f"OpenSees Deflection x{scale:g}: {case}")
    ax.set_xlabel("X / mm")
    ax.set_ylabel("Y / mm")
    ax.set_zlabel("Z / mm")
    fig.tight_layout()
    fig.savefig(output_dir / "opensees_deflection_diagram.png", dpi=dpi)
    plt.close(fig)


def plot_opensees_reaction_diagram(output_dir: Path, reactions: pd.DataFrame, dpi: int) -> None:
    if reactions.empty:
        _plot_message(output_dir / "opensees_reaction_diagram.png", "OpenSees reactions unavailable", "No OpenSees reaction results.", dpi)
        return
    pivot = reactions.groupby("node_id")["rz_n"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(pivot.index.astype(str), pivot.values, color="#4c78a8")
    ax.set_title("OpenSees Support Reactions Rz")
    ax.set_xlabel("Node ID")
    ax.set_ylabel("Rz / N")
    ax.grid(True, axis="y", lw=0.2, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "opensees_reaction_diagram.png", dpi=dpi)
    plt.close(fig)


def create_opensees_visualizations(output_dir: Path, opensees_result: Any, config: dict[str, Any], fixed_nodes: list[int], roller_nodes: list[int], deck_nodes: list[int]) -> list[str]:
    failures: list[str] = []
    dpi = int(config.get("visualization", {}).get("dpi", 180))
    scale = float(config.get("visualization", {}).get("deflection_scale", 20.0))
    tasks = [
        ("support_and_loads_opensees.png", lambda: plot_support_and_loads_opensees(output_dir, fixed_nodes, roller_nodes, deck_nodes, dpi)),
        ("opensees_force_diagram.png", lambda: plot_opensees_force_diagram(output_dir, opensees_result.member_forces, dpi)),
        ("opensees_deflection_diagram.png", lambda: plot_opensees_deflection_diagram(output_dir, opensees_result.node_displacements, dpi, scale)),
        ("opensees_reaction_diagram.png", lambda: plot_opensees_reaction_diagram(output_dir, opensees_result.reactions, dpi)),
    ]
    for name, func in tasks:
        try:
            func()
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            _plot_message(output_dir / name, f"{name} unavailable", str(exc), dpi)
    return failures
