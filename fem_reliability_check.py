from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from centerline_validation import connected_components_from_members
from fem_truss_solver import DOF_INDEX, TrussModel, assemble_stiffness


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _manual_nodes(config: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    overrides = config.get("manual_overrides", {}) or {}
    override_supports = overrides.get("support_nodes", {}) or {}
    fixed = list(override_supports.get("fixed") or [])
    roller = list(override_supports.get("roller") or [])
    deck = list(overrides.get("deck_nodes") or [])
    fixed += list(config.get("fixed_nodes") or [])
    roller += list(config.get("roller_nodes") or [])
    deck += list(config.get("deck_nodes") or [])
    supports = config.get("supports", {}) or {}
    fixed += list(supports.get("fixed_nodes") or [])
    roller += list(supports.get("roller_nodes") or [])
    return sorted({int(x) for x in fixed}), sorted({int(x) for x in roller}), sorted({int(x) for x in deck})


def _model_from_clean(nodes_df: pd.DataFrame, members_df: pd.DataFrame) -> TrussModel:
    if nodes_df.empty:
        return TrussModel(np.zeros((0, 3), dtype=float), [], {})
    ordered = nodes_df.sort_values("node_id")
    max_id = int(ordered["node_id"].max())
    nodes = np.zeros((max_id + 1, 3), dtype=float)
    for row in ordered.itertuples():
        nodes[int(row.node_id)] = [float(row.x_mm), float(row.y_mm), float(row.z_mm)]
    members: list[dict[str, Any]] = []
    node_members: dict[int, list[int]] = {}
    for idx, row in enumerate(members_df.itertuples()):
        member = {
            "member_id": int(row.member_id),
            "node_i": int(row.node_i),
            "node_j": int(row.node_j),
            "length_mm": float(getattr(row, "length_mm", 0.0)),
        }
        members.append(member)
        node_members.setdefault(member["node_i"], []).append(idx)
        node_members.setdefault(member["node_j"], []).append(idx)
    return TrussModel(nodes, members, node_members)


def _supports_from_manual(fixed: list[int], roller: list[int]) -> list[dict[str, Any]]:
    supports = [{"node_id": int(n), "restraints": ["Ux", "Uy", "Uz"], "type": "manual_fixed"} for n in fixed]
    supports.extend({"node_id": int(n), "restraints": ["Ux", "Uz"], "type": "manual_roller"} for n in roller)
    return supports


def _reason(
    code: str,
    severity: str,
    message: str,
    affected_nodes: list[int] | None = None,
    affected_members: list[int] | None = None,
    repair: str = "",
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "affected_nodes": affected_nodes or [],
        "affected_members": affected_members or [],
        "repair": repair,
    }


def _stiffness_check(model: TrussModel, config: dict[str, Any], supports: list[dict[str, Any]]) -> dict[str, Any]:
    if len(model.nodes) == 0 or not model.members:
        return {"checked": False, "singular": True, "condition_number": math.inf, "rank": 0, "free_dof_count": 0, "message": "缺少节点或杆件，无法检查刚度矩阵。"}
    if not supports:
        return {"checked": False, "singular": True, "condition_number": math.inf, "rank": 0, "free_dof_count": 0, "message": "未设置支座，无法检查刚度矩阵。"}
    try:
        k_global = assemble_stiffness(model, config)
        fixed: list[int] = []
        for support in supports:
            node_id = int(support["node_id"])
            if not (0 <= node_id < len(model.nodes)):
                continue
            for dof in support.get("restraints", []):
                fixed.append(3 * node_id + DOF_INDEX[dof])
        all_dofs = np.arange(len(model.nodes) * 3)
        free = np.array([int(d) for d in all_dofs if int(d) not in set(fixed)], dtype=int)
        if len(free) == 0:
            return {"checked": True, "singular": True, "condition_number": math.inf, "rank": 0, "free_dof_count": 0, "message": "没有自由自由度，支座约束设置异常。"}
        k_ff = k_global[np.ix_(free, free)]
        rank = int(np.linalg.matrix_rank(k_ff))
        singular = bool(rank < len(free))
        cond = float(np.linalg.cond(k_ff)) if len(free) else math.inf
        return {"checked": True, "singular": singular, "condition_number": cond, "rank": rank, "free_dof_count": int(len(free)), "message": "刚度矩阵检查完成。"}
    except Exception as exc:
        return {"checked": False, "singular": True, "condition_number": math.inf, "rank": 0, "free_dof_count": 0, "message": f"刚度矩阵检查失败: {exc}"}


def evaluate_fem_reliability(
    output_dir: Path,
    config: dict[str, Any],
    opensees_result: Any | None = None,
) -> dict[str, Any]:
    nodes_path = output_dir / "clean_nodes.csv"
    members_path = output_dir / "clean_members.csv"
    nodes_df = _read_csv(nodes_path)
    members_df = _read_csv(members_path)
    reasons: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if nodes_df.empty:
        reasons.append(_reason("missing_clean_nodes", "failed", "clean_nodes.csv 不存在或为空，无法建立 FEM 模型。", repair="先生成并人工验收 clean_nodes.csv。"))
    if members_df.empty:
        reasons.append(_reason("missing_clean_members", "failed", "clean_members.csv 不存在或为空，无法建立 FEM 模型。", repair="先生成并人工验收 clean_members.csv。"))

    model = _model_from_clean(nodes_df, members_df) if not nodes_df.empty and not members_df.empty else TrussModel(np.zeros((0, 3)), [], {})
    fixed_nodes, roller_nodes, deck_nodes = _manual_nodes(config)
    supports = _supports_from_manual(fixed_nodes, roller_nodes)

    if not fixed_nodes:
        reasons.append(_reason("fixed_nodes_missing", "unreliable", "fixed_nodes 未手动指定。", repair="在 config.yaml 的 manual_overrides.support_nodes.fixed 中指定固定铰支座节点，限制 Ux、Uy、Uz。"))
    if not roller_nodes:
        reasons.append(_reason("roller_nodes_missing", "unreliable", "roller_nodes 未手动指定。", repair="在 config.yaml 的 manual_overrides.support_nodes.roller 中指定滚动支座节点，限制 Ux、Uz，释放 Uy。"))
    if not deck_nodes:
        reasons.append(_reason("deck_nodes_missing", "unreliable", "deck_nodes 未手动指定。", repair="在 config.yaml 的 manual_overrides.deck_nodes 中只选择桥面高度范围内、跨度范围内且左右分布均衡的节点。"))

    if not members_df.empty:
        length_col = "length_mm" if "length_mm" in members_df.columns else None
        zero_members = []
        if length_col:
            zero_members = members_df[pd.to_numeric(members_df[length_col], errors="coerce").fillna(0) <= 1e-6]["member_id"].astype(int).tolist()
        duplicate_pairs = []
        seen: dict[tuple[int, int], int] = {}
        for row in members_df.itertuples():
            pair = tuple(sorted((int(row.node_i), int(row.node_j))))
            if pair in seen:
                duplicate_pairs.append((seen[pair], int(row.member_id)))
            else:
                seen[pair] = int(row.member_id)
        if zero_members:
            reasons.append(_reason("zero_length_members", "failed", f"存在 {len(zero_members)} 根零长度杆件。", affected_members=zero_members[:30], repair="删除零长度杆件或修正其端点。"))
        if duplicate_pairs:
            reasons.append(_reason("duplicate_members", "needs_review", f"存在 {len(duplicate_pairs)} 组重复杆件。", affected_members=[m for pair in duplicate_pairs[:20] for m in pair], repair="确认是否为真实并杆；若不是，应删除重复杆件或标记 ignored。"))

    member_counts = {}
    if not nodes_df.empty and "member_count" in nodes_df.columns:
        member_counts = dict(zip(nodes_df["node_id"].astype(int), nodes_df["member_count"].astype(int)))
    elif model.node_members:
        member_counts = {idx: len(model.node_members.get(idx, [])) for idx in range(len(model.nodes))}
    single_nodes = [int(n) for n, c in member_counts.items() if int(c) <= 1]
    if single_nodes:
        severity = "unreliable" if len(single_nodes) > 10 else "needs_review"
        reasons.append(_reason("single_member_nodes", severity, f"模型存在 {len(single_nodes)} 个单杆节点，示例: {single_nodes[:20]}。", affected_nodes=single_nodes[:30], repair="合并真实连接的杆端，检查 manual_node_overrides.csv，避免杆件只靠搭接接触。"))
    dangling_members = []
    for member in model.members:
        if member_counts.get(int(member["node_i"]), 0) <= 1 or member_counts.get(int(member["node_j"]), 0) <= 1:
            dangling_members.append(int(member["member_id"]))
    if dangling_members:
        reasons.append(_reason("dangling_members", "unreliable", f"存在 {len(dangling_members)} 根悬空杆件，示例: {dangling_members[:20]}。", affected_members=dangling_members[:30], repair="检查这些杆件端点是否应与主桁架节点汇交；若为辅助线，应从结构模型中排除。"))

    close_endpoints = _read_csv(output_dir / "close_unclustered_endpoints.csv")
    if not close_endpoints.empty:
        reasons.append(_reason("unconnected_endpoints", "unreliable", f"存在 {len(close_endpoints)} 组距离很近但未聚类的杆端。", repair="在 manual_node_overrides.csv 中强制合并真实连接的杆端；确认交叉杆件是否有金属节点连接。"))

    components: list[dict[str, Any]] = []
    if not nodes_df.empty and not members_df.empty:
        components = connected_components_from_members(nodes_df, members_df)
        if len(components) != 1:
            reasons.append(_reason("multiple_connected_components", "unreliable", f"主结构不是单一连通体，当前共有 {len(components)} 个 connected components。", repair="将真实连接节点强制合并；若某些分量是辅助构件，应标记为 ignored。"))

    support_ids = set(fixed_nodes + roller_nodes)
    invalid_deck = sorted(set(deck_nodes) - set(range(len(model.nodes)))) if deck_nodes else []
    support_overlap = sorted(set(deck_nodes) & support_ids)
    high_deck = []
    if deck_nodes and len(model.nodes):
        z_high = float(np.percentile(model.nodes[:, 2], 80))
        high_deck = [int(n) for n in deck_nodes if 0 <= int(n) < len(model.nodes) and model.nodes[int(n), 2] >= z_high]
    if invalid_deck:
        reasons.append(_reason("deck_nodes_invalid", "unreliable", f"deck_nodes 包含不存在的节点: {invalid_deck[:20]}。", affected_nodes=invalid_deck[:30], repair="重新筛选 deck_nodes，只保留 clean_nodes.csv 中存在的桥面节点。"))
    if support_overlap:
        reasons.append(_reason("deck_support_overlap", "unreliable", f"deck_nodes 误包含支座节点: {support_overlap}。", affected_nodes=support_overlap, repair="不要把支座节点计入桥面加载节点。"))
    if high_deck:
        warnings.append(_reason("deck_high_nodes", "needs_review", f"deck_nodes 可能包含上弦或高点节点: {high_deck[:20]}。", affected_nodes=high_deck[:30], repair="只选择桥面高度范围内的加载节点，不要选择上弦节点。"))

    stiffness = _stiffness_check(model, config, supports)
    cond_limit = float((config.get("sanity", {}) or {}).get("max_condition_number", 1e12))
    if stiffness.get("singular", True):
        reasons.append(_reason("singular_stiffness", "failed", f"全局刚度矩阵奇异或无法检查: {stiffness['message']} rank={stiffness.get('rank')}, free_dof={stiffness.get('free_dof_count')}。", repair="不要随机增加固定约束；优先修复断开的节点、真实支座、桥面 X 形拉结、顶部横向联系和端部底部拉结。"))
    elif float(stiffness.get("condition_number", math.inf)) > cond_limit:
        reasons.append(_reason("ill_conditioned_stiffness", "needs_review", f"全局刚度矩阵条件数异常: {float(stiffness['condition_number']):.3e}，阈值 {cond_limit:.3e}。", repair="检查近似机构、过长杆、支座分布和节点偏心。"))

    opensees_summary = {"requested": False, "available": False, "success": False, "message": "not evaluated"}
    if opensees_result is not None:
        opensees_summary = {
            "requested": bool(getattr(opensees_result, "requested", False)),
            "available": bool(getattr(opensees_result, "available", False)),
            "success": bool(getattr(opensees_result, "success", False)),
            "message": str(getattr(opensees_result, "message", "")),
        }
        if opensees_summary["requested"] and not opensees_summary["success"]:
            reasons.append(_reason("opensees_failed", "failed", f"OpenSeesPy 未成功完成分析: {opensees_summary['message']}", repair="先修复支座、节点连通和荷载节点，再复核 OpenSeesPy 安装和输入单位。"))

    reaction_balance = {"checked": False}
    reactions = _read_csv(output_dir / "opensees_reactions.csv")
    load_cases = _read_csv(output_dir / "opensees_case_summary.csv")
    if not reactions.empty and not load_cases.empty and "total_load_n" in load_cases.columns:
        total_reaction = float(pd.to_numeric(reactions.get("rz_n", pd.Series(dtype=float)), errors="coerce").sum())
        total_load = float(pd.to_numeric(load_cases["total_load_n"], errors="coerce").sum())
        err = abs(total_reaction + total_load) / max(abs(total_load), 1.0)
        reaction_balance = {"checked": True, "total_vertical_reaction_n": total_reaction, "total_load_n": total_load, "error_ratio": err}
        if err > 0.10:
            reasons.append(_reason("reaction_unbalanced", "unreliable", f"支座竖向反力与总荷载不平衡，差异 {err:.1%}。", repair="检查荷载单位 N、长度单位 mm、E 单位 MPa、反力方向和未约束自由度。"))

    severity_values = [item["severity"] for item in reasons]
    if "failed" in severity_values:
        status = "failed"
    elif "unreliable" in severity_values:
        status = "unreliable"
    elif "needs_review" in severity_values or warnings:
        status = "needs_review"
    else:
        status = "reliable"

    return {
        "fem_reliability_status": status,
        "can_use_as_structural_conclusion": status == "reliable",
        "reason_count": len(reasons),
        "reasons": reasons,
        "warnings": warnings,
        "repair_priority": _repair_priority(reasons, components),
        "fixed_nodes": fixed_nodes,
        "roller_nodes": roller_nodes,
        "deck_nodes": deck_nodes,
        "stiffness": stiffness,
        "connected_components": [
            {k: v for k, v in comp.items() if k not in {"node_ids", "member_ids"}}
            for comp in components
        ],
        "opensees": opensees_summary,
        "reaction_balance": reaction_balance,
    }


def _repair_priority(reasons: list[dict[str, Any]], components: list[dict[str, Any]]) -> dict[str, list[str]]:
    first = [
        "修复断开的节点和未连接杆端，更新 manual_node_overrides.csv。",
        "人工确认 fixed_nodes、roller_nodes 和 deck_nodes。",
        "消除刚度矩阵奇异，不要通过随机假约束掩盖结构机构。",
    ]
    for comp in components[:8]:
        first.append(
            f"连通分量 {comp['component_id']}: nodes={comp['node_count']}, members={comp['member_count']}, "
            f"X[{comp['x_min']:.0f},{comp['x_max']:.0f}], Y[{comp['y_min']:.0f},{comp['y_max']:.0f}], Z[{comp['z_min']:.0f},{comp['z_max']:.0f}]。"
        )
    return {
        "first_priority": first,
        "second_priority": [
            "增加桥面 X 形绳索拉结，提高抗扭和偏心荷载稳定性。",
            "增加顶部横向联系，缩短上弦压杆无支撑长度。",
            "增加端部底部拉结和闭合三角形端部门架。",
            "加强中跨下弦节点，优先使用金属节点板或双侧夹板。",
        ],
        "third_priority": [
            "优化材料数量和短杆排料。",
            "复核超长杆件拼接方式。",
            "优化报告图表和展板表达。",
        ],
    }


def write_fem_reliability_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# FEM 可靠性检查报告",
        "",
        f"- fem_reliability_status: {result['fem_reliability_status']}",
        f"- 当前 FEM 结果是否可作为结构结论: {'是' if result['can_use_as_structural_conclusion'] else '否'}",
        f"- fixed_nodes: {result['fixed_nodes'] or '未指定'}",
        f"- roller_nodes: {result['roller_nodes'] or '未指定'}",
        f"- deck_nodes: {result['deck_nodes'] or '未指定'}",
        f"- 刚度矩阵: singular={result['stiffness'].get('singular')}, condition_number={result['stiffness'].get('condition_number')}, rank={result['stiffness'].get('rank')}, free_dof={result['stiffness'].get('free_dof_count')}",
        "",
        "## 不可靠原因",
    ]
    if not result["reasons"]:
        lines.append("- 无阻断原因。")
    for item in result["reasons"]:
        lines.extend(
            [
                f"### {item['code']} ({item['severity']})",
                f"- 原因: {item['message']}",
                f"- 受影响节点: {item['affected_nodes'] or '无'}",
                f"- 受影响杆件: {item['affected_members'] or '无'}",
                f"- 修复建议: {item['repair']}",
                "",
            ]
        )
    if result.get("warnings"):
        lines.append("## 需要人工复核的警告")
        for item in result["warnings"]:
            lines.append(f"- {item['message']} 修复建议: {item['repair']}")
    lines.extend(
        [
            "",
            "## 当前模型是否可以进入正式结构分析",
            "- 可以" if result["can_use_as_structural_conclusion"] else "- 不可以。当前结果只能用于排查，不应作为正式位移、杆力和支座反力结论。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_repair_actions(path: Path, result: dict[str, Any]) -> None:
    p = result["repair_priority"]
    lines = ["# FEM 可靠性修复步骤", ""]
    for title, key in [("第一优先级：必须先修", "first_priority"), ("第二优先级：结构构造加强", "second_priority"), ("第三优先级：材料和表达优化", "third_priority")]:
        lines.extend([f"## {title}", ""])
        for item in p.get(key, []):
            lines.append(f"- {item}")
        lines.append("")
    lines.extend(
        [
            "## 原则",
            "- 不要用随机固定节点来掩盖结构机构。",
            "- 不要把不可靠 FEM 结果作为最终结构结论。",
            "- 不要把近似排料长度用于 FEM；FEM 必须使用 clean_members.csv 中的真实几何长度。",
            "- 真实解决结构机构，应通过增加节点连接、X 形拉结、横向联系、端部拉结等构造方式完成。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def fem_reliability_report_lines(result: dict[str, Any]) -> list[str]:
    if not result:
        return ["", "## FEM 可靠性检查", "- 未生成 FEM 可靠性检查。"]
    lines = [
        "",
        "## FEM 可靠性检查",
        f"- 当前 FEM 可靠性状态: {result['fem_reliability_status']}",
        f"- 是否通过可靠性检查: {'是' if result['can_use_as_structural_conclusion'] else '否'}",
        f"- 当前结果是否可作为结构结论: {'是' if result['can_use_as_structural_conclusion'] else '否；仅供排查，不作为最终受力结论'}",
        "- 不可靠原因:",
    ]
    for item in result["reasons"][:12]:
        lines.append(f"  - {item['code']}: {item['message']} 修复: {item['repair']}")
    lines.extend(["", "### 修复优先级"])
    for item in result["repair_priority"]["first_priority"][:10]:
        lines.append(f"- 第一优先级: {item}")
    for item in result["repair_priority"]["second_priority"]:
        lines.append(f"- 第二优先级: {item}")
    for item in result["repair_priority"]["third_priority"]:
        lines.append(f"- 第三优先级: {item}")
    lines.append("- 下一步修改建议: 先让 FEM 可靠性状态达到 reliable，再输出正式最大位移、杆力和支座反力。")
    return lines


def write_fem_reliability_json(path: Path, result: dict[str, Any]) -> None:
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_nodes_members(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _read_csv(output_dir / "clean_nodes.csv"), _read_csv(output_dir / "clean_members.csv")


def _draw_base(ax: Any, nodes: pd.DataFrame, members: pd.DataFrame, xf: str, yf: str) -> None:
    lookup = {int(r.node_id): (float(r.x_mm), float(r.y_mm), float(r.z_mm)) for r in nodes.itertuples()} if not nodes.empty else {}
    idx = {"x_mm": 0, "y_mm": 1, "z_mm": 2}
    for row in members.itertuples():
        if int(row.node_i) in lookup and int(row.node_j) in lookup:
            a, b = lookup[int(row.node_i)], lookup[int(row.node_j)]
            ax.plot([a[idx[xf]], b[idx[xf]]], [a[idx[yf]], b[idx[yf]]], color="#bbbbbb", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.35)


def _reason_points(result: dict[str, Any], nodes: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    lookup = {int(r.node_id): (float(r.x_mm), float(r.y_mm), float(r.z_mm)) for r in nodes.itertuples()} if not nodes.empty else {}
    member_lookup = {int(r.member_id): r for r in members.itertuples()} if not members.empty else {}
    rows = []
    for reason in result.get("reasons", []):
        for node in reason.get("affected_nodes", [])[:30]:
            if int(node) in lookup:
                x, y, z = lookup[int(node)]
                rows.append({"x": x, "y": y, "z": z, "severity": reason["severity"], "label": reason["code"]})
        for member_id in reason.get("affected_members", [])[:30]:
            row = member_lookup.get(int(member_id))
            if row is not None and int(row.node_i) in lookup and int(row.node_j) in lookup:
                a, b = lookup[int(row.node_i)], lookup[int(row.node_j)]
                rows.append({"x": (a[0] + b[0]) / 2, "y": (a[1] + b[1]) / 2, "z": (a[2] + b[2]) / 2, "severity": reason["severity"], "label": reason["code"]})
    return pd.DataFrame(rows)


def _plot_message(path: Path, title: str, message: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=15, weight="bold")
    ax.text(0.5, 0.4, message, ha="center", va="center", fontsize=11, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def create_fem_reliability_visualizations(output_dir: Path, result: dict[str, Any], config: dict[str, Any]) -> list[str]:
    dpi = int((config.get("visualization", {}) or {}).get("dpi", 180))
    failures: list[str] = []
    nodes, members = _read_nodes_members(output_dir)
    try:
        _plot_reason_map(output_dir / "fem_unreliable_reason_map.png", nodes, members, result, dpi)
        _plot_support_deck(output_dir / "support_constraint_check.png", nodes, members, result, dpi, mode="support")
        _plot_support_deck(output_dir / "deck_load_node_check.png", nodes, members, result, dpi, mode="deck")
        _plot_components(output_dir / "disconnected_component_map.png", nodes, members, result, dpi)
    except Exception as exc:
        failures.append(str(exc))
    return failures


def _plot_reason_map(path: Path, nodes: pd.DataFrame, members: pd.DataFrame, result: dict[str, Any], dpi: int) -> None:
    if nodes.empty or members.empty:
        _plot_message(path, "FEM unreliable reason map unavailable", "clean centerline files are missing.", dpi)
        return
    points = _reason_points(result, nodes, members)
    views = [("Plan X-Y", "x_mm", "y_mm", "x", "y"), ("Elevation X-Z", "x_mm", "z_mm", "x", "z"), ("Section Y-Z", "y_mm", "z_mm", "y", "z")]
    colors = {"failed": "#d62728", "unreliable": "#ff7f0e", "needs_review": "#f2c94c"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, xf, yf, px, py) in zip(axes, views):
        _draw_base(ax, nodes, members, xf, yf)
        if not points.empty:
            for sev, subset in points.groupby("severity"):
                ax.scatter(subset[px], subset[py], s=40, color=colors.get(sev, "#333333"), label=sev)
        ax.set_title(f"{title} - For troubleshooting only")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("FEM Unreliable Reasons - troubleshooting only")
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_support_deck(path: Path, nodes: pd.DataFrame, members: pd.DataFrame, result: dict[str, Any], dpi: int, mode: str) -> None:
    if nodes.empty:
        _plot_message(path, "Node check unavailable", "clean_nodes.csv is missing.", dpi)
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_base(ax, nodes, members, "x_mm", "y_mm")
    if mode == "support":
        fixed = result.get("fixed_nodes", [])
        roller = result.get("roller_nodes", [])
        for ids, marker, color, label in [(fixed, "s", "#d62728", "fixed Ux Uy Uz"), (roller, "^", "#2f80ed", "roller Ux Uz")]:
            pts = nodes[nodes["node_id"].isin(ids)]
            if not pts.empty:
                ax.scatter(pts["x_mm"], pts["y_mm"], s=80, marker=marker, color=color, label=label, zorder=3)
        ax.set_title("Support Constraint Check - troubleshooting only")
    else:
        deck = result.get("deck_nodes", [])
        pts = nodes[nodes["node_id"].isin(deck)]
        if not pts.empty:
            ax.scatter(pts["x_mm"], pts["y_mm"], s=60, color="#27ae60", label="deck load nodes", zorder=3)
        ax.set_title("Deck Load Node Check - troubleshooting only")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_components(path: Path, nodes: pd.DataFrame, members: pd.DataFrame, result: dict[str, Any], dpi: int) -> None:
    if nodes.empty or members.empty:
        _plot_message(path, "Disconnected component map unavailable", "clean centerline files are missing.", dpi)
        return
    comps = connected_components_from_members(nodes, members)
    if not comps:
        _plot_message(path, "Disconnected component map unavailable", "No components found.", dpi)
        return
    node_to_comp = {}
    for comp in comps:
        for node_id in comp["node_ids"]:
            node_to_comp[int(node_id)] = int(comp["component_id"])
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab20")
    lookup = {int(r.node_id): (float(r.x_mm), float(r.y_mm), float(r.z_mm)) for r in nodes.itertuples()}
    for row in members.itertuples():
        if int(row.node_i) in lookup and int(row.node_j) in lookup:
            a, b = lookup[int(row.node_i)], lookup[int(row.node_j)]
            cid = node_to_comp.get(int(row.node_i), 0)
            ax.plot([a[0], b[0]], [a[1], b[1]], color=cmap(cid % 20), linewidth=1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Disconnected Component Map - troubleshooting only")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
