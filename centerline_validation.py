from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fem_truss_solver import DOF_INDEX, TrussModel, assemble_stiffness


RELIABLE_FEM_BLOCK_MESSAGE = "支座节点和桥面加载节点尚未人工确认，当前模型不能进入可靠受力分析。"


def _node_lookup(nodes_df: pd.DataFrame) -> dict[int, np.ndarray]:
    return {
        int(row.node_id): np.array([float(row.x_mm), float(row.y_mm), float(row.z_mm)], dtype=float)
        for row in nodes_df.itertuples()
    }


def connected_components_from_members(nodes_df: pd.DataFrame, members_df: pd.DataFrame) -> list[dict[str, Any]]:
    node_ids = [int(x) for x in nodes_df["node_id"].tolist()]
    adjacency: dict[int, set[int]] = {node_id: set() for node_id in node_ids}
    member_lookup: dict[int, list[int]] = {node_id: [] for node_id in node_ids}
    for row in members_df.itertuples():
        i = int(row.node_i)
        j = int(row.node_j)
        member_id = int(row.member_id)
        adjacency.setdefault(i, set()).add(j)
        adjacency.setdefault(j, set()).add(i)
        member_lookup.setdefault(i, []).append(member_id)
        member_lookup.setdefault(j, []).append(member_id)

    lookup = _node_lookup(nodes_df)
    visited: set[int] = set()
    components: list[dict[str, Any]] = []
    for node_id in node_ids:
        if node_id in visited:
            continue
        stack = [node_id]
        visited.add(node_id)
        comp_nodes: list[int] = []
        comp_members: set[int] = set()
        while stack:
            current = stack.pop()
            comp_nodes.append(current)
            comp_members.update(member_lookup.get(current, []))
            for nxt in adjacency.get(current, set()):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        pts = np.array([lookup[n] for n in comp_nodes if n in lookup], dtype=float)
        if len(pts):
            bbox_min = pts.min(axis=0)
            bbox_max = pts.max(axis=0)
        else:
            bbox_min = bbox_max = np.zeros(3)
        components.append(
            {
                "component_id": len(components) + 1,
                "node_ids": sorted(comp_nodes),
                "member_ids": sorted(comp_members),
                "node_count": len(comp_nodes),
                "member_count": len(comp_members),
                "x_min": float(bbox_min[0]),
                "x_max": float(bbox_max[0]),
                "y_min": float(bbox_min[1]),
                "y_max": float(bbox_max[1]),
                "z_min": float(bbox_min[2]),
                "z_max": float(bbox_max[2]),
            }
        )
    return sorted(components, key=lambda item: (item["node_count"], item["member_count"]), reverse=True)


def centerline_basic_checks(
    nodes_df: pd.DataFrame,
    members_df: pd.DataFrame,
    diagnostics: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    standard_length = float(config["section"]["standard_length_mm"])
    short_limit = float((config.get("centerline_validation", {}) or {}).get("short_member_limit_mm", 0.15 * standard_length))
    long_limit = float((config.get("centerline_validation", {}) or {}).get("long_member_limit_mm", standard_length))

    duplicate_pairs = []
    seen_pairs: dict[tuple[int, int], int] = {}
    zero_length_members: list[int] = []
    short_members: list[int] = []
    long_members: list[int] = []
    for row in members_df.itertuples():
        member_id = int(row.member_id)
        i = int(row.node_i)
        j = int(row.node_j)
        pair = tuple(sorted((i, j)))
        if pair in seen_pairs:
            duplicate_pairs.append({"member_a": seen_pairs[pair], "member_b": member_id, "node_i": pair[0], "node_j": pair[1]})
        else:
            seen_pairs[pair] = member_id
        length = float(getattr(row, "length_mm", getattr(row, "real_length_mm", 0.0)))
        if length <= 1e-6:
            zero_length_members.append(member_id)
        if length < short_limit:
            short_members.append(member_id)
        if length > long_limit + 0.5:
            long_members.append(member_id)

    components = connected_components_from_members(nodes_df, members_df)
    summary = {
        "node_count": int(len(nodes_df)),
        "member_count": int(len(members_df)),
        "single_member_node_count": int(len(diagnostics.get("single_member_nodes", []))),
        "dangling_member_count": int(len(diagnostics.get("dangling_members", []))),
        "unconnected_endpoint_count": int(len(diagnostics.get("close_unclustered_endpoints", []))),
        "duplicate_member_count": int(len(duplicate_pairs)),
        "zero_length_member_count": int(len(zero_length_members)),
        "short_member_count": int(len(short_members)),
        "long_member_count": int(len(long_members)),
        "abnormal_node_count": int(len(diagnostics.get("abnormal_nodes", []))),
        "connected_component_count": int(len(components)),
        "main_component_node_count": int(components[0]["node_count"]) if components else 0,
        "main_component_member_count": int(components[0]["member_count"]) if components else 0,
        "is_single_connected_component": len(components) == 1,
        "zero_length_members": zero_length_members,
        "short_members": short_members,
        "long_members": long_members,
        "duplicate_pairs": duplicate_pairs,
    }
    return summary, components, duplicate_pairs


def check_deck_nodes(model: TrussModel, deck_node_ids: list[int], supports: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    nodes = model.nodes
    support_ids = {int(s["node_id"]) for s in supports}
    valid = [idx for idx in deck_node_ids if 0 <= int(idx) < len(nodes)]
    z_low = float(np.percentile(nodes[:, 2], 15)) if len(nodes) else 0.0
    z_mid = float(np.percentile(nodes[:, 2], 55)) if len(nodes) else 0.0
    z_high = float(np.percentile(nodes[:, 2], 80)) if len(nodes) else 0.0
    x_min = float(np.min(nodes[:, 0])) if len(nodes) else 0.0
    x_max = float(np.max(nodes[:, 0])) if len(nodes) else 0.0
    y_mid = float(np.median(nodes[:, 1])) if len(nodes) else 0.0
    if not valid:
        rows.append({"check": "deck_nodes_present", "status": "fail", "message": "未指定桥面加载节点。"})
    else:
        deck_pts = nodes[valid]
        deck_x_span = float(np.max(deck_pts[:, 0]) - np.min(deck_pts[:, 0])) if len(deck_pts) else 0.0
        span = max(x_max - x_min, 1.0)
        high_nodes = [int(idx) for idx in valid if nodes[int(idx), 2] >= z_high]
        support_overlap = sorted(set(valid) & support_ids)
        left_count = int(np.sum(deck_pts[:, 1] < y_mid)) if len(deck_pts) else 0
        right_count = int(np.sum(deck_pts[:, 1] >= y_mid)) if len(deck_pts) else 0
        imbalance = abs(left_count - right_count) / max(len(valid), 1)
        if deck_x_span < 0.45 * span:
            rows.append({"check": "deck_span_distribution", "status": "review", "message": f"桥面加载节点跨向覆盖不足: {deck_x_span:.1f}mm / {span:.1f}mm。"})
        if high_nodes:
            rows.append({"check": "deck_height", "status": "review", "message": f"桥面节点可能包含上弦或高点节点: {high_nodes[:20]}。"})
        if support_overlap:
            rows.append({"check": "deck_support_overlap", "status": "review", "message": f"桥面加载节点包含支座节点: {support_overlap}。"})
        if imbalance > 0.7 and len(valid) >= 4:
            rows.append({"check": "deck_y_balance", "status": "review", "message": f"桥面节点左右分布不均衡: left={left_count}, right={right_count}。"})
        low_count = int(np.sum(deck_pts[:, 2] <= z_mid)) if len(deck_pts) else 0
        if low_count < max(1, int(0.5 * len(valid))):
            rows.append({"check": "deck_height_range", "status": "review", "message": "多数桥面节点不在模型中低位/桥面高度范围。"})
    if not rows:
        rows.append({"check": "deck_nodes", "status": "pass", "message": "桥面加载节点初步合理。"})
    summary = {
        "deck_node_count": len(valid),
        "deck_nodes_valid": bool(valid) and not any(row["status"] == "fail" for row in rows),
        "deck_nodes_need_review": any(row["status"] in {"review", "fail"} for row in rows),
        "z_low_reference": z_low,
        "z_mid_reference": z_mid,
        "z_high_reference": z_high,
    }
    return pd.DataFrame(rows), summary


def check_support_nodes(model: TrussModel, supports: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    nodes = model.nodes
    support_ids = [int(s["node_id"]) for s in supports if 0 <= int(s["node_id"]) < len(nodes)]
    x_min = float(np.min(nodes[:, 0])) if len(nodes) else 0.0
    x_max = float(np.max(nodes[:, 0])) if len(nodes) else 0.0
    span = max(x_max - x_min, 1.0)
    z_low = float(np.percentile(nodes[:, 2], 20)) if len(nodes) else 0.0
    fixed = [s for s in supports if "Uy" in s.get("restraints", []) and 0 <= int(s["node_id"]) < len(nodes)]
    roller = [s for s in supports if "Uy" not in s.get("restraints", []) and 0 <= int(s["node_id"]) < len(nodes)]
    if not support_ids:
        rows.append({"check": "support_nodes_present", "status": "fail", "message": "未指定支座节点。"})
    if not fixed:
        rows.append({"check": "fixed_support_present", "status": "fail", "message": "缺少固定铰支座节点。"})
    if not roller:
        rows.append({"check": "roller_support_present", "status": "fail", "message": "缺少滚动支座节点。"})
    if support_ids:
        off_end = [idx for idx in support_ids if min(abs(nodes[idx, 0] - x_min), abs(nodes[idx, 0] - x_max)) > 0.18 * span]
        high = [idx for idx in support_ids if nodes[idx, 2] > z_low + 0.15 * max(float(np.ptp(nodes[:, 2])), 1.0)]
        left_count = sum(1 for idx in support_ids if nodes[idx, 0] <= x_min + 0.18 * span)
        right_count = sum(1 for idx in support_ids if nodes[idx, 0] >= x_max - 0.18 * span)
        restrained = set()
        for support in supports:
            for dof in support.get("restraints", []):
                restrained.add(dof)
        if off_end:
            rows.append({"check": "support_end_location", "status": "review", "message": f"部分支座不在桥两端附近: {off_end}。"})
        if high:
            rows.append({"check": "support_low_location", "status": "review", "message": f"部分支座不在低点附近: {high}。"})
        if left_count == 0 or right_count == 0:
            rows.append({"check": "support_left_right", "status": "fail", "message": "支座没有分布在左右两端。"})
        if not {"Ux", "Uy", "Uz"}.issubset(restrained):
            rows.append({"check": "rigid_body_restraint", "status": "fail", "message": f"支座约束自由度不足: {sorted(restrained)}。"})
        if len(support_ids) < 2:
            rows.append({"check": "support_count", "status": "fail", "message": "支座节点过少，至少需要左右两端支承。"})
    if not rows:
        rows.append({"check": "support_nodes", "status": "pass", "message": "支座节点初步合理。"})
    summary = {
        "support_node_count": len(support_ids),
        "fixed_support_count": len(fixed),
        "roller_support_count": len(roller),
        "support_nodes_valid": not any(row["status"] == "fail" for row in rows),
        "support_nodes_need_review": any(row["status"] in {"review", "fail"} for row in rows),
    }
    return pd.DataFrame(rows), summary


def fem_precheck(
    model: TrussModel,
    config: dict[str, Any],
    supports: list[dict[str, Any]],
    deck_node_ids: list[int],
    centerline_summary: dict[str, Any],
    support_summary: dict[str, Any],
    deck_summary: dict[str, Any],
    manual_confirmed: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not manual_confirmed:
        rows.append({"check": "manual_support_load_confirmation", "status": "fail", "message": RELIABLE_FEM_BLOCK_MESSAGE})
    if not centerline_summary.get("is_single_connected_component", False):
        rows.append({"check": "structure_connectivity", "status": "fail", "message": f"结构存在 {centerline_summary.get('connected_component_count')} 个互不连通分量。"})
    if centerline_summary.get("single_member_node_count", 0) > 0:
        rows.append({"check": "single_member_nodes", "status": "fail", "message": f"存在 {centerline_summary['single_member_node_count']} 个单杆节点。"})
    if centerline_summary.get("dangling_member_count", 0) > 0:
        rows.append({"check": "dangling_members", "status": "fail", "message": f"存在 {centerline_summary['dangling_member_count']} 根悬空杆件。"})
    if not support_summary.get("support_nodes_valid", False):
        rows.append({"check": "support_definition", "status": "fail", "message": "支座节点或约束不足。"})
    if not deck_summary.get("deck_nodes_valid", False):
        rows.append({"check": "load_definition", "status": "fail", "message": "桥面加载节点未定义或需复核。"})

    condition_number = float("inf")
    rank = 0
    free_dof_count = 0
    singular = True
    if supports:
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
            free = np.array([d for d in all_dofs if d not in set(fixed)], dtype=int)
            free_dof_count = int(len(free))
            if len(free) == 0:
                rows.append({"check": "free_dof", "status": "fail", "message": "没有自由自由度，支座约束设置异常。"})
            else:
                k_ff = k_global[np.ix_(free, free)]
                rank = int(np.linalg.matrix_rank(k_ff))
                singular = rank < len(free)
                condition_number = float(np.linalg.cond(k_ff))
                if singular:
                    rows.append({"check": "stiffness_rank", "status": "fail", "message": f"全局刚度矩阵奇异: rank={rank}, free_dof={len(free)}。"})
                cond_limit = float(config.get("sanity", {}).get("max_condition_number", 1e12))
                if condition_number > cond_limit:
                    rows.append({"check": "condition_number", "status": "fail", "message": f"刚度矩阵条件数异常: {condition_number:.3e}。"})
        except Exception as exc:
            rows.append({"check": "stiffness_precheck", "status": "fail", "message": f"刚度矩阵前置检查失败: {exc}"})
    else:
        rows.append({"check": "stiffness_precheck", "status": "fail", "message": "未设置支座，无法检查刚度矩阵。"})
    if not rows:
        rows.append({"check": "fem_precheck", "status": "pass", "message": "FEM 前置验收通过。"})
    passed = not any(row["status"] == "fail" for row in rows)
    summary = {
        "fem_precheck_passed": passed,
        "can_enter_reliable_fem": passed,
        "condition_number": condition_number,
        "rank": rank,
        "free_dof_count": free_dof_count,
        "singular": singular,
        "message": "当前中心线模型可以进入可靠 FEM 分析。" if passed else "当前中心线模型不能进入可靠 FEM 分析。",
        "blocking_reasons": [row["message"] for row in rows if row["status"] == "fail"],
    }
    return pd.DataFrame(rows), summary


def score_centerline_model(
    metadata: Any,
    centerline_summary: dict[str, Any],
    support_summary: dict[str, Any],
    deck_summary: dict[str, Any],
    material_summary: dict[str, Any],
    fem_summary: dict[str, Any],
) -> dict[str, Any]:
    geometry = 20 if getattr(metadata, "detected_standard_1_to_10", False) else 15
    node_penalty = min(25, centerline_summary.get("single_member_node_count", 0) * 1 + centerline_summary.get("dangling_member_count", 0) * 2 + max(0, centerline_summary.get("connected_component_count", 1) - 1) * 5)
    node_connection = max(0, 25 - node_penalty)
    support_definition = 20 if support_summary.get("support_nodes_valid", False) and support_summary.get("support_node_count", 0) >= 2 else (8 if support_summary.get("support_node_count", 0) else 0)
    load_definition = 15 if deck_summary.get("deck_nodes_valid", False) and not deck_summary.get("deck_nodes_need_review", False) else (8 if deck_summary.get("deck_node_count", 0) else 0)
    material = 10 if material_summary.get("stock_wood_count", 0) > 0 else 0
    fem = 10 if fem_summary.get("fem_precheck_passed", False) else 0
    total = geometry + node_connection + support_definition + load_definition + material + fem
    return {
        "geometry_score": int(geometry),
        "node_connection_score": int(node_connection),
        "support_definition_score": int(support_definition),
        "load_definition_score": int(load_definition),
        "material_count_score": int(material),
        "fem_solvability_score": int(fem),
        "centerline_model_score": int(total),
        "can_enter_reliable_fem": bool(fem_summary.get("can_enter_reliable_fem", False)),
    }


def write_centerline_validation_report(path: Path, summary: dict[str, Any], components: list[dict[str, Any]], fem_summary: dict[str, Any], score: dict[str, Any]) -> None:
    lines = [
        "# Centerline Validation Report",
        "",
        "## 总览",
        f"- 节点总数: {summary['node_count']}",
        f"- 杆件总数: {summary['member_count']}",
        f"- 单杆节点数量: {summary['single_member_node_count']}",
        f"- 悬空杆件数量: {summary['dangling_member_count']}",
        f"- 未连接杆端数量: {summary['unconnected_endpoint_count']}",
        f"- 重复杆件数量: {summary['duplicate_member_count']}",
        f"- 零长度杆件: {summary['zero_length_members'] or '无'}",
        f"- 超短杆件: {summary['short_members'] or '无'}",
        f"- 超长杆件: {summary['long_members'] or '无'}",
        f"- 连接数异常节点数量: {summary['abnormal_node_count']}",
        f"- connected components: {summary['connected_component_count']}",
        f"- 主结构是否单一连通体: {'是' if summary['is_single_connected_component'] else '否'}",
        f"- centerline_model_score: {score['centerline_model_score']}/100",
        f"- 是否可以进入可靠 FEM 分析: {'是' if fem_summary['can_enter_reliable_fem'] else '否'}",
        "",
        "## FEM 前置验收结论",
        f"- {fem_summary['message']}",
    ]
    if fem_summary.get("blocking_reasons"):
        lines.append("- 阻断原因:")
        for reason in fem_summary["blocking_reasons"]:
            lines.append(f"  - {reason}")
    lines.extend(["", "## 连通分量"])
    for comp in components:
        lines.append(
            f"- component {comp['component_id']}: nodes={comp['node_count']}, members={comp['member_count']}, "
            f"X[{comp['x_min']:.1f}, {comp['x_max']:.1f}], Y[{comp['y_min']:.1f}, {comp['y_max']:.1f}], Z[{comp['z_min']:.1f}, {comp['z_max']:.1f}]"
        )
    if summary.get("duplicate_pairs"):
        lines.extend(["", "## 重复杆件"])
        for item in summary["duplicate_pairs"][:50]:
            lines.append(f"- member {item['member_a']} 与 member {item['member_b']} 重复连接 node {item['node_i']} - node {item['node_j']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_check_markdown(path: Path, title: str, check_df: pd.DataFrame, summary: dict[str, Any]) -> None:
    lines = [f"# {title}", "", "## Summary"]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Checks"])
    for row in check_df.itertuples():
        lines.append(f"- [{row.status}] {row.check}: {row.message}")
    path.write_text("\n".join(lines), encoding="utf-8")
