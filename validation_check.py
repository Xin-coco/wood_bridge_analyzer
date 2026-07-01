from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fem_truss_solver import LoadCaseResult, TrussModel, load_vector


REVIEW_WARNING = "该结果可能由支座设置、节点聚类、荷载分配或模型识别错误导致，需要人工复核。"


def member_zone(model: TrussModel, node_i: int, node_j: int) -> str:
    p1 = model.nodes[int(node_i)]
    p2 = model.nodes[int(node_j)]
    z_mid = float((p1[2] + p2[2]) / 2.0)
    x_mid = float((p1[0] + p2[0]) / 2.0)
    z_low = float(np.percentile(model.nodes[:, 2], 35))
    z_high = float(np.percentile(model.nodes[:, 2], 65))
    x_min = float(np.min(model.nodes[:, 0]))
    x_max = float(np.max(model.nodes[:, 0]))
    dz = abs(float(p2[2] - p1[2]))
    dx = abs(float(p2[0] - p1[0]))
    end_zone = x_mid <= x_min + 0.18 * max(x_max - x_min, 1.0) or x_mid >= x_max - 0.18 * max(x_max - x_min, 1.0)
    if end_zone:
        return "end_support"
    if dz > 0.35 * max(dx, 1.0):
        return "diagonal_or_vertical"
    if z_mid >= z_high:
        return "top_chord"
    if z_mid <= z_low:
        return "bottom_chord"
    return "web_or_deck"


def run_validation_checks(
    model: TrussModel,
    result: LoadCaseResult,
    config: dict[str, Any],
    deck_node_ids: list[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    force = load_vector(model, config, result.name if result.name != "moving_load_envelope" else "central_person")
    reactions = result.reactions.reshape((-1, 3)) if len(result.reactions) else np.zeros((len(model.nodes), 3))
    total_v_load = float(-np.sum(force[2::3]))
    total_v_reaction = float(np.sum(reactions[:, 2]))
    vertical_error_ratio = abs(total_v_reaction - total_v_load) / max(total_v_load, 1.0)
    if vertical_error_ratio > 0.08:
        checks.append({"check": "vertical_equilibrium", "status": "warning", "message": f"竖向反力与竖向荷载不平衡，差异 {vertical_error_ratio:.1%}。"})

    horizontal_reaction = float(np.linalg.norm(np.sum(reactions[:, :2], axis=0)))
    horizontal_ratio = horizontal_reaction / max(total_v_load, 1.0)
    if horizontal_ratio > 0.25:
        checks.append({"check": "horizontal_reaction", "status": "review", "message": f"水平反力偏大，约为竖向荷载的 {horizontal_ratio:.1%}。"})

    disp = result.displacements.reshape((-1, 3)) if len(result.displacements) else np.zeros((len(model.nodes), 3))
    disp_norm = np.linalg.norm(disp, axis=1)
    max_disp_node = int(np.argmax(np.abs(disp[:, 2]))) if len(disp) else -1
    max_total_disp_node = int(np.argmax(disp_norm)) if len(disp_norm) else -1
    x_min = float(np.min(model.nodes[:, 0]))
    x_max = float(np.max(model.nodes[:, 0]))
    x_ratio = (float(model.nodes[max_disp_node, 0]) - x_min) / max(x_max - x_min, 1.0) if max_disp_node >= 0 else -1.0
    if max_disp_node >= 0 and not (0.30 <= x_ratio <= 0.70 or max_disp_node in set(deck_node_ids)):
        checks.append({"check": "max_deflection_location", "status": "review", "message": f"最大竖向位移节点 {max_disp_node} 不在桥面或中跨附近。"})

    max_comp_member = None
    max_tens_member = None
    max_comp_zone = "none"
    max_tens_zone = "none"
    high_buckling_count = 0
    high_risk_count = 0
    if not result.member_results.empty:
        comp = result.member_results[result.member_results["force_type"] == "compression"].sort_values("axial_force_n", ascending=True).head(1)
        tens = result.member_results[result.member_results["force_type"] == "tension"].sort_values("axial_force_n", ascending=False).head(1)
        high_buckling_count = int((result.member_results["buckling_utilization"] >= 0.6).sum())
        high_risk_count = int((result.member_results["risk_score"] >= 0.6).sum())
        if not comp.empty:
            row = comp.iloc[0]
            max_comp_member = int(row["member_id"])
            max_comp_zone = member_zone(model, int(row["node_i"]), int(row["node_j"]))
            if max_comp_zone not in {"top_chord", "diagonal_or_vertical", "end_support"}:
                checks.append({"check": "compression_zone", "status": "review", "message": f"最大压杆 member {max_comp_member} 位于 {max_comp_zone}，与常见桁架受力路径不一致。"})
        if not tens.empty:
            row = tens.iloc[0]
            max_tens_member = int(row["member_id"])
            max_tens_zone = member_zone(model, int(row["node_i"]), int(row["node_j"]))
            if max_tens_zone not in {"bottom_chord", "diagonal_or_vertical"}:
                checks.append({"check": "tension_zone", "status": "review", "message": f"最大拉杆 member {max_tens_member} 位于 {max_tens_zone}，需要核对荷载路径。"})

        area = float(config["section"]["width_mm"]) * float(config["section"]["height_mm"])
        threshold_force = area * max(float(config["materials"]["allowable_tension_mpa"]), float(config["materials"]["allowable_compression_mpa"])) * 2.0
        if result.max_abs_force_n > threshold_force:
            checks.append({"check": "large_member_force", "status": "danger", "message": f"单根杆件轴力异常大: {result.max_abs_force_n:.1f} N。"})

    span = float(np.max(model.nodes[:, 0]) - np.min(model.nodes[:, 0]))
    max_total_disp = float(np.max(disp_norm)) if len(disp_norm) else 0.0
    if max_total_disp > max(0.05 * span, float(config["bridge"]["max_deflection_mm"])):
        checks.append({"check": "large_node_displacement", "status": "danger", "message": f"节点总位移异常大: node {max_total_disp_node}, {max_total_disp:.2f} mm。"})

    cond_limit = float(config.get("sanity", {}).get("max_condition_number", 1e12))
    if result.condition_number > cond_limit:
        checks.append({"check": "condition_number", "status": "warning", "message": f"全局刚度矩阵条件数异常: {result.condition_number:.3e}。"})
    if result.singular or result.rank < result.free_dof_count:
        checks.append({"check": "mechanism", "status": "warning", "message": "结构存在疑似机构或近似机构，结果需要作为趋势判断而非精确值。"})

    requires_review = bool(checks)
    summary = {
        "total_vertical_load_n": total_v_load,
        "total_vertical_reaction_n": total_v_reaction,
        "vertical_error_ratio": vertical_error_ratio,
        "horizontal_reaction_n": horizontal_reaction,
        "horizontal_reaction_ratio": horizontal_ratio,
        "max_vertical_displacement_node": max_disp_node,
        "max_total_displacement_node": max_total_disp_node,
        "max_total_displacement_mm": max_total_disp,
        "max_displacement_x_ratio": x_ratio,
        "max_compression_member": max_comp_member,
        "max_compression_zone": max_comp_zone,
        "max_tension_member": max_tens_member,
        "max_tension_zone": max_tens_zone,
        "high_buckling_member_count": high_buckling_count,
        "high_risk_member_count": high_risk_count,
        "condition_number": result.condition_number,
        "suspected_mechanism": bool(result.singular or result.rank < result.free_dof_count),
        "requires_manual_review": requires_review,
        "review_message": REVIEW_WARNING if requires_review else "验证检查未发现明显异常。",
    }
    return pd.DataFrame(checks), summary
