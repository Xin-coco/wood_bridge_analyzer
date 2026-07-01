from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fem_truss_solver import TrussModel, LoadCaseResult


def generate_fix_suggestions(
    model: TrussModel,
    result: LoadCaseResult,
    diagnostics: dict[str, Any],
    rod_summary: dict[str, Any],
    deck_node_ids: list[int],
    supports: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    suggestions: list[dict[str, Any]] = []
    nodes = model.nodes
    span_mid = (float(np.min(nodes[:, 0])) + float(np.max(nodes[:, 0]))) / 2.0
    member_df = result.member_results

    if not member_df.empty:
        bottom_z = float(np.percentile(nodes[:, 2], 35))
        mid_bottom = []
        for row in member_df.itertuples():
            p1, p2 = nodes[int(row.node_i)], nodes[int(row.node_j)]
            x_mid = (p1[0] + p2[0]) / 2.0
            z_mid = (p1[2] + p2[2]) / 2.0
            if abs(x_mid - span_mid) < 0.18 * max(np.ptp(nodes[:, 0]), 1.0) and z_mid <= bottom_z:
                mid_bottom.append(row.member_id)
        suggestions.append(
            {
                "priority": 1,
                "topic": "中跨下弦",
                "target": mid_bottom[:8] or "中跨下弦区域",
                "suggestion": "中跨下弦是挠度和拉力敏感区，建议检查是否需要双杆并联、缩短节点间距或增加底部拉结。",
            }
        )

        compression = member_df[member_df["force_type"] == "compression"].sort_values("buckling_utilization", ascending=False).head(8)
        suggestions.append(
            {
                "priority": 2,
                "topic": "上弦压杆横向约束",
                "target": compression["member_id"].tolist(),
                "suggestion": "屈曲利用率靠前的压杆建议增加顶部横向联系或侧向约束，避免弱轴失稳。",
            }
        )

    support_like = sorted({int(s["node_id"]) for s in supports})
    suggestions.append(
        {
            "priority": 3,
            "topic": "端部支座底部拉结",
            "target": support_like or "两端支座底部节点",
            "suggestion": "端部支座附近建议设置底部横向拉结，避免支座反力导致桥宽方向张开或节点滑移。",
        }
    )
    suggestions.append(
        {
            "priority": 4,
            "topic": "桥面 X 形拉结",
            "target": deck_node_ids[:12],
            "suggestion": "桥面平面建议设置 X 形拉结或等效剪刀撑，提高偏心行走荷载下的平面稳定性。",
        }
    )
    problem_nodes = diagnostics.get("single_member_nodes", []) + diagnostics.get("abnormal_nodes", [])
    suggestions.append(
        {
            "priority": 5,
            "topic": "金属节点板或双侧夹板",
            "target": sorted(set(problem_nodes))[:12],
            "suggestion": "连接数量异常或受力集中的节点建议使用金属节点板、螺栓夹板或双侧夹板，降低连接滑移。",
        }
    )
    suggestions.append(
        {
            "priority": 6,
            "topic": "超长杆件拆分拼接",
            "target": rod_summary.get("overlength_ids", []),
            "suggestion": "超过 1300mm 的杆件需要拆分为标准杆拼接，并在拼接位置设置可靠夹板和错缝连接。",
        }
    )
    lines = ["# Fix Suggestions", ""]
    for item in sorted(suggestions, key=lambda x: x["priority"]):
        lines.append(f"## {item['priority']}. {item['topic']}")
        lines.append(f"- 目标位置: {item['target'] or '无明显目标'}")
        lines.append(f"- 建议: {item['suggestion']}")
        lines.append("")
    return suggestions, "\n".join(lines)


def write_fix_suggestions(path: Path, suggestions_md: str) -> None:
    path.write_text(suggestions_md, encoding="utf-8")
