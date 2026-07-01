from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fem_truss_solver import TrussModel, LoadCaseResult, load_vector


def _zone_for_member(model: TrussModel, row: Any) -> str:
    p1 = model.nodes[int(row.node_i)]
    p2 = model.nodes[int(row.node_j)]
    z_mid = float((p1[2] + p2[2]) / 2.0)
    z_low = float(np.percentile(model.nodes[:, 2], 35))
    z_high = float(np.percentile(model.nodes[:, 2], 65))
    dz = abs(float(p2[2] - p1[2]))
    dx = abs(float(p2[0] - p1[0]))
    if dz > 0.35 * max(dx, 1.0):
        return "diagonal_or_vertical"
    if z_mid >= z_high:
        return "top_chord"
    if z_mid <= z_low:
        return "bottom_chord"
    return "web_or_deck"


def run_sanity_checks(
    model: TrussModel,
    result: LoadCaseResult,
    config: dict[str, Any],
    deck_node_ids: list[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    force = load_vector(model, config, result.name if result.name != "moving_load_envelope" else "central_person")
    total_vertical_load = float(-np.sum(force[2::3]))
    total_vertical_reaction = float(np.sum(result.reactions[2::3])) if len(result.reactions) else 0.0
    balance_error = abs(total_vertical_reaction - total_vertical_load)
    balance_ratio = balance_error / max(total_vertical_load, 1.0)
    if balance_ratio > 0.08:
        issues.append({"check": "load_reaction_balance", "level": "warning", "message": f"总竖向荷载与支座反力差异 {balance_ratio:.1%}。"})

    if len(result.displacements):
        disp = result.displacements.reshape((-1, 3))
        max_node = int(np.argmax(np.abs(disp[:, 2])))
        x_min, x_max = float(np.min(model.nodes[:, 0])), float(np.max(model.nodes[:, 0]))
        x_ratio = (float(model.nodes[max_node, 0]) - x_min) / max(x_max - x_min, 1.0)
        near_midspan = 0.30 <= x_ratio <= 0.70
        near_deck = max_node in set(deck_node_ids)
        if not near_midspan and not near_deck:
            issues.append({"check": "max_deflection_location", "level": "warning", "message": f"最大位移节点 {max_node} 不在中跨或桥面加载节点附近。"})
    else:
        max_node = None
        x_ratio = None

    max_comp_zone = "none"
    max_tens_zone = "none"
    if not result.member_results.empty:
        comp = result.member_results[result.member_results["force_type"] == "compression"].sort_values("axial_force_n", ascending=True).head(1)
        tens = result.member_results[result.member_results["force_type"] == "tension"].sort_values("axial_force_n", ascending=False).head(1)
        if not comp.empty:
            max_comp_zone = _zone_for_member(model, comp.iloc[0])
            if max_comp_zone not in {"top_chord", "diagonal_or_vertical"}:
                issues.append({"check": "compression_member_zone", "level": "review", "message": f"最大压杆位于 {max_comp_zone}，不符合常见上弦/斜撑受压直觉。"})
        if not tens.empty:
            max_tens_zone = _zone_for_member(model, tens.iloc[0])
            if max_tens_zone != "bottom_chord":
                issues.append({"check": "tension_member_zone", "level": "review", "message": f"最大拉杆位于 {max_tens_zone}，需要人工复核荷载路径。"})

        span = float(np.max(model.nodes[:, 0]) - np.min(model.nodes[:, 0]))
        if result.max_vertical_displacement_mm > max(0.08 * span, float(config["bridge"]["max_deflection_mm"]) * 2.0):
            issues.append({"check": "large_displacement", "level": "danger", "message": "最大位移异常偏大，可能存在机构或支座设置错误。"})
        area = float(config["section"]["width_mm"]) * float(config["section"]["height_mm"])
        very_large_force = area * float(config["materials"]["allowable_compression_mpa"]) * 3.0
        if result.max_abs_force_n > very_large_force:
            issues.append({"check": "large_member_force", "level": "danger", "message": "最大杆力超过许用压应力估算值 3 倍，需复核模型。"})

    if result.condition_number > float(config.get("sanity", {}).get("max_condition_number", 1e12)):
        issues.append({"check": "condition_number", "level": "warning", "message": f"刚度矩阵条件数异常: {result.condition_number:.3e}。"})
    if result.singular:
        issues.append({"check": "singular_stiffness", "level": "warning", "message": "刚度矩阵奇异，当前结果主要用于定位风险，需要人工复核。"})

    summary = {
        "total_vertical_load_n": total_vertical_load,
        "total_vertical_reaction_n": total_vertical_reaction,
        "balance_ratio": balance_ratio,
        "max_displacement_node": max_node,
        "max_displacement_x_ratio": x_ratio,
        "max_compression_zone": max_comp_zone,
        "max_tension_zone": max_tens_zone,
        "condition_number": result.condition_number,
        "requires_manual_review": bool(issues),
        "review_message": "当前计算结果可能受节点识别、支座设置或荷载分配影响，需要人工复核。" if issues else "Sanity checks did not flag major issues.",
    }
    return pd.DataFrame(issues), summary
