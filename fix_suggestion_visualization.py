from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import tempfile

import pandas as pd

_cache_root = Path(tempfile.gettempdir()) / "wood_bridge_analyzer_cache"
(_cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
(_cache_root / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _plot_message(path: Path, title: str, message: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=16, weight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=11, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _node_lookup(nodes: pd.DataFrame) -> dict[int, tuple[float, float, float]]:
    if nodes.empty:
        return {}
    return {int(row.node_id): (float(row.x_mm), float(row.y_mm), float(row.z_mm)) for row in nodes.itertuples()}


def _issue_points(diagnosis: dict[str, Any], nodes: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    lookup = _node_lookup(nodes)
    rows = []
    member_lookup = {int(row.member_id): row for row in members.itertuples()} if not members.empty else {}
    for issue in diagnosis.get("issues", []):
        sev = issue.get("severity", "medium")
        label = f"P{issue.get('priority_rank')} {issue.get('issue_type')}"
        for node_id in issue.get("affected_nodes", [])[:20]:
            if int(node_id) in lookup:
                x, y, z = lookup[int(node_id)]
                rows.append({"x": x, "y": y, "z": z, "severity": sev, "label": label})
        for member_id in issue.get("affected_members", [])[:20]:
            member = member_lookup.get(int(member_id))
            if member is None:
                continue
            if int(member.node_i) in lookup and int(member.node_j) in lookup:
                a = lookup[int(member.node_i)]
                b = lookup[int(member.node_j)]
                rows.append({"x": (a[0] + b[0]) / 2, "y": (a[1] + b[1]) / 2, "z": (a[2] + b[2]) / 2, "severity": sev, "label": label})
    return pd.DataFrame(rows)


def _draw_base(ax: Any, nodes: pd.DataFrame, members: pd.DataFrame, x_field: str, y_field: str) -> None:
    lookup = _node_lookup(nodes)
    for row in members.itertuples():
        if int(row.node_i) not in lookup or int(row.node_j) not in lookup:
            continue
        a = lookup[int(row.node_i)]
        b = lookup[int(row.node_j)]
        idx = {"x_mm": 0, "y_mm": 1, "z_mm": 2}
        ax.plot([a[idx[x_field]], b[idx[x_field]]], [a[idx[y_field]], b[idx[y_field]]], color="#c8c8c8", linewidth=0.8, zorder=1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.35)


def plot_problem_map(output_dir: Path, diagnosis: dict[str, Any], dpi: int) -> None:
    nodes = _read_csv(output_dir / "clean_nodes.csv")
    members = _read_csv(output_dir / "clean_members.csv")
    if nodes.empty or members.empty:
        _plot_message(output_dir / "problem_map_3views.png", "Problem map unavailable", "clean_nodes.csv or clean_members.csv is missing.", dpi)
        return
    points = _issue_points(diagnosis, nodes, members)
    views = [("Plan X-Y", "x_mm", "y_mm", "x", "y"), ("Elevation X-Z", "x_mm", "z_mm", "x", "z"), ("Section Y-Z", "y_mm", "z_mm", "y", "z")]
    colors = {"critical": "#d62728", "high": "#ff7f0e", "medium": "#f2c94c", "low": "#2f80ed"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, xf, yf, px, py) in zip(axes, views):
        _draw_base(ax, nodes, members, xf, yf)
        if not points.empty:
            for sev, subset in points.groupby("severity"):
                ax.scatter(subset[px], subset[py], s=36, color=colors.get(sev, "#333333"), label=sev, zorder=3)
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle("V2.2 Problem Map")
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(output_dir / "problem_map_3views.png", dpi=dpi)
    plt.close(fig)


def plot_fix_suggestion_map(output_dir: Path, diagnosis: dict[str, Any], dpi: int) -> None:
    nodes = _read_csv(output_dir / "clean_nodes.csv")
    members = _read_csv(output_dir / "clean_members.csv")
    if nodes.empty or members.empty:
        _plot_message(output_dir / "fix_suggestion_map.png", "Fix map unavailable", "clean centerline files are missing.", dpi)
        return
    points = _issue_points({"issues": diagnosis.get("issues", [])[:10]}, nodes, members)
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_base(ax, nodes, members, "x_mm", "y_mm")
    if not points.empty:
        ax.scatter(points["x"], points["y"], s=60, color="#d62728", zorder=3)
        for _, row in points.head(12).iterrows():
            ax.text(float(row["x"]), float(row["y"]), str(row["label"]).split()[0], fontsize=8, color="#111111")
    ax.set_title("Fix Suggestion Map - Plan View")
    ax.set_xlabel("X mm")
    ax.set_ylabel("Y mm")
    fig.tight_layout()
    fig.savefig(output_dir / "fix_suggestion_map.png", dpi=dpi)
    plt.close(fig)


def plot_priority_ranking(output_dir: Path, diagnosis: dict[str, Any], dpi: int, top_n: int) -> None:
    issues = diagnosis.get("issues", [])[:top_n]
    if not issues:
        _plot_message(output_dir / "priority_ranking.png", "Priority ranking unavailable", "No issues were generated.", dpi)
        return
    severity_score = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    labels = [f"P{i['priority_rank']} {i['issue_type']}" for i in issues]
    values = [severity_score.get(i["severity"], 2) for i in issues]
    colors = [{"critical": "#d62728", "high": "#ff7f0e", "medium": "#f2c94c", "low": "#2f80ed"}.get(i["severity"], "#888888") for i in issues]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(issues))))
    ax.barh(range(len(issues)), values, color=colors)
    ax.set_yticks(range(len(issues)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 4.5)
    ax.set_xlabel("Severity score")
    ax.set_title("Structural Issue Priority Ranking")
    fig.tight_layout()
    fig.savefig(output_dir / "priority_ranking.png", dpi=dpi)
    plt.close(fig)


def plot_before_after_expected_effect(output_dir: Path, diagnosis: dict[str, Any], dpi: int) -> None:
    comparison = _read_csv(output_dir / "comparison_summary.csv")
    if comparison.empty:
        _plot_message(
            output_dir / "before_after_expected_effect.png",
            "Before/after comparison unavailable",
            "Run with --compare-model to visualize measured before/after changes. Current image records expected effects only.",
            dpi,
        )
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = comparison["metric"].astype(str).tolist()
    old = pd.to_numeric(comparison["old"], errors="coerce").fillna(0)
    new = pd.to_numeric(comparison["new"], errors="coerce").fillna(0)
    x = range(len(labels))
    ax.bar([i - 0.18 for i in x], old, width=0.36, label="before", color="#9ecae1")
    ax.bar([i + 0.18 for i in x], new, width=0.36, label="after", color="#fdae6b")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.set_title("Before/After Expected Effect")
    fig.tight_layout()
    fig.savefig(output_dir / "before_after_expected_effect.png", dpi=dpi)
    plt.close(fig)


def create_fix_suggestion_visualizations(output_dir: Path, diagnosis: dict[str, Any], config: dict[str, Any], top_n: int) -> list[str]:
    dpi = int((config.get("visualization", {}) or {}).get("dpi", 180))
    failures: list[str] = []
    tasks = [
        ("problem_map_3views.png", lambda: plot_problem_map(output_dir, diagnosis, dpi)),
        ("fix_suggestion_map.png", lambda: plot_fix_suggestion_map(output_dir, diagnosis, dpi)),
        ("priority_ranking.png", lambda: plot_priority_ranking(output_dir, diagnosis, dpi, top_n)),
        ("before_after_expected_effect.png", lambda: plot_before_after_expected_effect(output_dir, diagnosis, dpi)),
    ]
    for name, fn in tasks:
        try:
            fn()
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            _plot_message(output_dir / name, f"{name} failed", str(exc), dpi)
    return failures
