from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math

import pandas as pd

from issue_classifier import severity_for_issue, sort_issues
from recommendation_engine import enrich_issue


def _read_csv(path: Path) -> tuple[pd.DataFrame, str | None]:
    if not path.exists():
        return pd.DataFrame(), f"缺少文件: {path.name}"
    if path.stat().st_size == 0:
        return pd.DataFrame(), None
    try:
        return pd.read_csv(path), None
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), None
    except Exception as exc:
        return pd.DataFrame(), f"无法读取 {path.name}: {exc}"


def _read_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, f"缺少文件: {path.name}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return {}, f"无法读取 {path.name}: {exc}"


def _node_position_map(nodes: pd.DataFrame) -> dict[int, tuple[float, float, float]]:
    if nodes.empty:
        return {}
    return {int(row.node_id): (float(row.x_mm), float(row.y_mm), float(row.z_mm)) for row in nodes.itertuples()}


def _member_midpoint(member: Any, positions: dict[int, tuple[float, float, float]]) -> tuple[float, float, float] | None:
    ni = int(getattr(member, "node_i"))
    nj = int(getattr(member, "node_j"))
    if ni not in positions or nj not in positions:
        return None
    a = positions[ni]
    b = positions[nj]
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)


def _issue(
    issue_id: str,
    issue_type: str,
    affected_nodes: list[int] | None,
    affected_members: list[int] | None,
    evidence: dict[str, Any],
    reason: str,
    priority_rank: int,
) -> dict[str, Any]:
    issue = {
        "issue_id": issue_id,
        "issue_type": issue_type,
        "severity": severity_for_issue(issue_type, evidence),
        "affected_nodes": affected_nodes or [],
        "affected_members": affected_members or [],
        "evidence": evidence,
        "reason": reason,
        "recommendation": "",
        "expected_effect": "",
        "material_impact": "",
        "construction_difficulty": "",
        "priority_rank": priority_rank,
    }
    return enrich_issue(issue)


def _load_inputs(output_dir: Path) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    data: dict[str, Any] = {}
    for key, filename in {
        "nodes": "clean_nodes.csv",
        "members": "clean_members.csv",
        "buckling": "buckling_check.csv",
        "load_cases": "load_case_comparison.csv",
        "validation": "validation_check.csv",
        "opensees_displacements": "opensees_node_displacements.csv",
        "opensees_forces": "opensees_member_forces.csv",
        "opensees_reactions": "opensees_reactions.csv",
        "components": "connected_components.csv",
        "support_check": "support_node_check.csv",
        "deck_check": "deck_node_check.csv",
        "cut_plan": "paired_stock_cut_plan.csv",
    }.items():
        df, warning = _read_csv(output_dir / filename)
        data[key] = df
        if warning:
            warnings.append(warning)
    material, warning = _read_json(output_dir / "material_stock_summary.json")
    data["material"] = material
    if warning:
        warnings.append(warning)
    comparison, warning = _read_json(output_dir / "solver_comparison_summary.json")
    data["solver_comparison"] = comparison
    if warning and not (output_dir / "solver_comparison.md").exists():
        warnings.append(warning)
    fem_reliability, warning = _read_json(output_dir / "fem_reliability_summary.json")
    data["fem_reliability"] = fem_reliability
    if warning:
        warnings.append(warning)
    return data, warnings


def _main_span_width(nodes: pd.DataFrame) -> dict[str, float]:
    if nodes.empty:
        return {"span_mm": 0.0, "width_mm": 0.0, "height_mm": 0.0}
    return {
        "span_mm": float(nodes["x_mm"].max() - nodes["x_mm"].min()) if "x_mm" in nodes else 0.0,
        "width_mm": float(nodes["y_mm"].max() - nodes["y_mm"].min()) if "y_mm" in nodes else 0.0,
        "height_mm": float(nodes["z_mm"].max() - nodes["z_mm"].min()) if "z_mm" in nodes else 0.0,
    }


def _diagnose_nodes(data: dict[str, Any], issues: list[dict[str, Any]], priority: int) -> int:
    nodes = data["nodes"]
    members = data["members"]
    if nodes.empty:
        issues.append(_issue("N-001", "fem_not_reliable", [], [], {"missing": "clean_nodes.csv"}, "缺少清理后的节点表，无法确认结构网络。", priority))
        return priority + 1
    member_count_col = "member_count" if "member_count" in nodes.columns else None
    if member_count_col:
        single_nodes = nodes[nodes[member_count_col] <= 1]["node_id"].astype(int).tolist()
        if single_nodes:
            issues.append(
                _issue(
                    "N-101",
                    "single_member_nodes",
                    single_nodes[:20],
                    [],
                    {"single_member_node_count": len(single_nodes), "sample_nodes": single_nodes[:20]},
                    "存在只连接一根杆件的节点，说明中心线端点可能没有汇交，或模型中存在悬空构件。",
                    priority,
                )
            )
            priority += 1
        complex_nodes = nodes[nodes[member_count_col] >= 8]["node_id"].astype(int).tolist()
        if complex_nodes:
            issues.append(
                _issue(
                    "N-102",
                    "complex_nodes",
                    complex_nodes[:20],
                    [],
                    {"complex_node_count": len(complex_nodes), "threshold": 8},
                    "部分节点连接杆件过多，可能存在多个真实节点被错误聚类到同一点。",
                    priority,
                )
            )
            priority += 1
    if not members.empty and member_count_col:
        counts = dict(zip(nodes["node_id"].astype(int), nodes[member_count_col].astype(int)))
        dangling = []
        for row in members.itertuples():
            if counts.get(int(row.node_i), 0) <= 1 or counts.get(int(row.node_j), 0) <= 1:
                dangling.append(int(row.member_id))
        if dangling:
            issues.append(
                _issue(
                    "N-201",
                    "dangling_members",
                    [],
                    dangling[:30],
                    {"dangling_member_count": len(dangling), "sample_members": dangling[:30]},
                    "若杆件至少一端只连接到单杆节点，该杆件很可能没有真正接入桁架传力路径。",
                    priority,
                )
            )
            priority += 1
    components = data["components"]
    if not components.empty:
        comp_count = len(components)
        if comp_count > 1:
            issues.append(
                _issue(
                    "N-301",
                    "disconnected_components",
                    [],
                    [],
                    {"connected_component_count": comp_count, "largest_component_nodes": int(components.get("node_count", pd.Series([0])).max())},
                    "中心线网络不是单一连通体，说明左右桁架、桥面或局部杆件没有形成连续结构体系。",
                    priority,
                )
            )
            priority += 1
    return priority


def _diagnose_analysis_results(data: dict[str, Any], issues: list[dict[str, Any]], priority: int, displacement_limit_mm: float) -> int:
    fem_rel = data.get("fem_reliability") or {}
    if fem_rel and fem_rel.get("fem_reliability_status") != "reliable":
        reasons = fem_rel.get("reasons", [])
        affected_nodes: list[int] = []
        affected_members: list[int] = []
        for reason in reasons:
            affected_nodes.extend(int(x) for x in reason.get("affected_nodes", [])[:10])
            affected_members.extend(int(x) for x in reason.get("affected_members", [])[:10])
        issues.append(
            _issue(
                "A-001",
                "fem_not_reliable",
                sorted(set(affected_nodes))[:30],
                sorted(set(affected_members))[:30],
                {
                    "fem_reliability_status": fem_rel.get("fem_reliability_status"),
                    "reason_count": fem_rel.get("reason_count"),
                    "reasons": [item.get("message") for item in reasons[:8]],
                },
                "FEM 可靠性检查未通过，当前位移、杆力和支座反力只能用于排查，不能作为正式结构结论。",
                priority,
            )
        )
        return priority + 1
    load_cases = data["load_cases"]
    if load_cases.empty or not bool(load_cases.get("success", pd.Series([False])).astype(bool).any()):
        issues.append(
            _issue(
                "A-001",
                "fem_not_reliable",
                [],
                [],
                {"load_case_rows": 0 if load_cases.empty else len(load_cases), "message": "no successful FEM/OpenSees load case"},
                "当前求解结果不可作为可靠结论，通常由支座设置、节点聚类、荷载节点或模型识别问题导致。",
                priority,
            )
        )
        priority += 1
    else:
        max_disp = float(pd.to_numeric(load_cases["max_vertical_displacement_mm"], errors="coerce").max())
        if max_disp > displacement_limit_mm:
            issues.append(
                _issue(
                    "A-101",
                    "excessive_displacement",
                    [],
                    [],
                    {"max_displacement_mm": max_disp, "limit_mm": displacement_limit_mm},
                    "最大竖向位移超过任务书建议控制值，桥面竖向刚度不足。",
                    priority,
                )
            )
            priority += 1
    reactions = data["opensees_reactions"]
    if not reactions.empty:
        rz = float(pd.to_numeric(reactions.get("rz_n", pd.Series(dtype=float)), errors="coerce").sum())
        rx = float(pd.to_numeric(reactions.get("rx_n", pd.Series(dtype=float)), errors="coerce").abs().sum())
        ry = float(pd.to_numeric(reactions.get("ry_n", pd.Series(dtype=float)), errors="coerce").abs().sum())
        if abs(rz) > 1e-6 and (rx + ry) / max(abs(rz), 1.0) > 0.25:
            issues.append(
                _issue(
                    "S-201",
                    "reaction_unbalanced",
                    reactions.get("node_id", pd.Series(dtype=int)).astype(int).tolist()[:20],
                    [],
                    {"vertical_reaction_sum_n": rz, "horizontal_reaction_sum_n": rx + ry, "horizontal_to_vertical_ratio": (rx + ry) / max(abs(rz), 1.0)},
                    "水平反力相对竖向反力偏大，可能存在支座约束、荷载方向或节点几何异常。",
                    priority,
                )
            )
            priority += 1
    support_check = data["support_check"]
    if support_check.empty:
        issues.append(
            _issue("S-101", "support_definition_missing", [], [], {"missing": "support_node_check.csv or confirmed support nodes"}, "支座节点未形成可诊断的人工确认记录。", priority)
        )
        priority += 1
    return priority


def _diagnose_geometry_stability(data: dict[str, Any], issues: list[dict[str, Any]], priority: int) -> int:
    nodes = data["nodes"]
    members = data["members"]
    if nodes.empty or members.empty:
        return priority
    dims = _main_span_width(nodes)
    positions = _node_position_map(nodes)
    member_rows = list(members.itertuples())
    transverse = 0
    x_like = 0
    top_links = 0
    z_threshold = float(nodes["z_mm"].quantile(0.75)) if "z_mm" in nodes else 0.0
    for row in member_rows:
        if int(row.node_i) not in positions or int(row.node_j) not in positions:
            continue
        a = positions[int(row.node_i)]
        b = positions[int(row.node_j)]
        dx = abs(b[0] - a[0])
        dy = abs(b[1] - a[1])
        dz = abs(b[2] - a[2])
        if dy > max(dx, dz, 1.0) and dy > 0.35 * max(dims["width_mm"], 1.0):
            transverse += 1
            if a[2] >= z_threshold and b[2] >= z_threshold:
                top_links += 1
        if dx > 0.15 * max(dims["span_mm"], 1.0) and dy > 0.15 * max(dims["width_mm"], 1.0):
            x_like += 1
    if x_like < 2:
        issues.append(
            _issue(
                "G-101",
                "deck_bracing_missing",
                [],
                [],
                {"diagonal_plan_bracing_like_members": x_like, "span_mm": dims["span_mm"], "width_mm": dims["width_mm"]},
                "模型中缺少明显的桥面平面 X 形拉结，偏心荷载下桥面抗扭可能不足。",
                priority,
            )
        )
        priority += 1
    if top_links < 2:
        issues.append(
            _issue(
                "G-201",
                "top_lateral_bracing_missing",
                [],
                [],
                {"top_transverse_link_count": top_links, "z_top_threshold_mm": z_threshold},
                "顶部横向联系数量偏少，上弦压杆侧向支撑可能不足。",
                priority,
            )
        )
        priority += 1
    if transverse < 3 and dims["width_mm"] > 1000:
        issues.append(
            _issue(
                "G-301",
                "torsion_risk",
                [],
                [],
                {"transverse_link_count": transverse, "width_mm": dims["width_mm"]},
                "宽桥面若横向联系不足，左右桁架难以形成稳定空间盒子，存在扭转趋势。",
                priority,
            )
        )
        priority += 1
    return priority


def _diagnose_buckling_material(data: dict[str, Any], issues: list[dict[str, Any]], priority: int) -> int:
    buckling = data["buckling"]
    if not buckling.empty:
        util_col = "buckling_utilization" if "buckling_utilization" in buckling.columns else None
        if util_col:
            risky = buckling[pd.to_numeric(buckling[util_col], errors="coerce") > 1 / 1.5]
            if not risky.empty:
                top = risky.sort_values(util_col, ascending=False).head(10)
                issues.append(
                    _issue(
                        "B-101",
                        "high_buckling_risk",
                        [],
                        top["member_id"].astype(int).tolist(),
                        {"risk_member_count": len(risky), "max_buckling_utilization": float(pd.to_numeric(top[util_col], errors="coerce").max())},
                        "部分受压杆欧拉屈曲安全储备不足，加载时可能先发生侧向失稳。",
                        priority,
                    )
                )
                priority += 1
    material = data["material"]
    if material:
        oversized = [int(x) for x in material.get("oversized_member_ids", [])]
        if oversized:
            issues.append(
                _issue(
                    "M-101",
                    "oversized_members",
                    [],
                    oversized,
                    {"oversized_member_count": len(oversized), "oversized_member_ids": oversized},
                    "存在超过 1300mm 标准木杆长度的构件，施工时需要拼接或重新设计。",
                    priority,
                )
            )
            priority += 1
        diff = abs(float(material.get("stock_count_difference_vs_manual", 0.0)))
        if diff >= 5:
            issues.append(
                _issue(
                    "M-201",
                    "material_count_difference",
                    [],
                    [],
                    {
                        "model_member_count": material.get("model_member_count"),
                        "program_stock_wood_count": material.get("program_stock_wood_count"),
                        "manual_stock_count": material.get("manual_stock_count"),
                        "stock_wood_count": material.get("stock_wood_count"),
                        "difference": diff,
                    },
                    "程序排料结果和人工复核数量差异较大，说明材料统计仍需要以人工复核表校正。",
                    priority,
                )
            )
            priority += 1
        waste = float(material.get("average_waste_mm", 0.0) or 0.0)
        if waste > 250:
            issues.append(
                _issue(
                    "M-301",
                    "large_waste",
                    [],
                    [],
                    {"average_waste_mm": waste, "total_waste_mm": material.get("total_waste_mm")},
                    "排料平均余料偏大，短杆组合仍可继续优化。",
                    priority,
                )
            )
            priority += 1
    return priority


def run_structural_diagnosis(output_dir: Path, config: dict[str, Any], top_n: int = 10) -> dict[str, Any]:
    data, warnings = _load_inputs(output_dir)
    issues: list[dict[str, Any]] = []
    priority = 1
    displacement_limit = float(((config.get("opensees", {}) or {}).get("analysis", {}) or {}).get("displacement_limit_mm", 500.0))
    priority = _diagnose_nodes(data, issues, priority)
    priority = _diagnose_analysis_results(data, issues, priority, displacement_limit)
    priority = _diagnose_geometry_stability(data, issues, priority)
    priority = _diagnose_buckling_material(data, issues, priority)
    issues = sort_issues(issues)
    for idx, issue in enumerate(issues, start=1):
        issue["priority_rank"] = idx
    critical = sum(1 for issue in issues if issue["severity"] == "critical")
    high = sum(1 for issue in issues if issue["severity"] == "high")
    medium = sum(1 for issue in issues if issue["severity"] == "medium")
    low = sum(1 for issue in issues if issue["severity"] == "low")
    can_build = critical == 0 and high <= 1
    if critical:
        overall = "当前模型不建议直接进入实体建造或承重测试；应先修复关键节点、支座/荷载或求解可靠性问题。"
    elif high:
        overall = "当前模型可作为方案继续推敲，但强烈建议先处理高风险稳定问题。"
    else:
        overall = "当前模型没有发现阻断性问题，但仍需按建议完成节点和施工复核。"
    return {
        "overall_judgement": overall,
        "can_enter_construction": can_build,
        "should_continue_model_revision": not can_build or bool(issues),
        "issue_count": len(issues),
        "severity_counts": {"critical": critical, "high": high, "medium": medium, "low": low},
        "missing_or_unreadable_inputs": warnings,
        "top_issues": issues[:top_n],
        "issues": issues,
        "diagnosis_top_n": top_n,
    }
