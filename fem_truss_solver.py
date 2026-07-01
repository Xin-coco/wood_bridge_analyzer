from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from bridge_parser import Rod


DOF_INDEX = {"Ux": 0, "Uy": 1, "Uz": 2}


@dataclass
class TrussModel:
    nodes: np.ndarray
    members: list[dict[str, Any]]
    node_members: dict[int, list[int]]


@dataclass
class LoadCaseResult:
    name: str
    displacements: np.ndarray
    reactions: np.ndarray
    member_results: pd.DataFrame
    max_vertical_displacement_mm: float
    max_abs_force_n: float
    max_reaction_n: float
    singular: bool
    rank: int
    free_dof_count: int
    condition_number: float
    success: bool
    message: str


def build_truss_model(rods: list[Rod], tolerance_mm: float) -> TrussModel:
    nodes: list[np.ndarray] = []
    members: list[dict[str, Any]] = []
    node_members: dict[int, list[int]] = {}

    def node_for(point: np.ndarray) -> int:
        for idx, existing in enumerate(nodes):
            if np.linalg.norm(point - existing) <= tolerance_mm:
                nodes[idx] = (existing + point) / 2.0
                return idx
        nodes.append(point.copy())
        return len(nodes) - 1

    for rod in rods:
        i = node_for(rod.start)
        j = node_for(rod.end)
        if i == j:
            continue
        member_index = len(members)
        members.append({"member_id": rod.id, "node_i": i, "node_j": j, "length_mm": rod.length_mm, "rod": rod})
        node_members.setdefault(i, []).append(member_index)
        node_members.setdefault(j, []).append(member_index)
    return TrussModel(np.array(nodes, dtype=float), members, node_members)


def node_diagnostics(model: TrussModel, rods: list[Rod], tolerance_mm: float) -> dict[str, Any]:
    member_counts = {i: len(model.node_members.get(i, [])) for i in range(len(model.nodes))}
    isolated_nodes = [i for i, count in member_counts.items() if count == 0]
    single_member_nodes = [i for i, count in member_counts.items() if count == 1]
    abnormal_nodes = [i for i, count in member_counts.items() if count <= 1]
    dangling_members = sorted(
        {
            int(m["member_id"])
            for m in model.members
            if member_counts.get(m["node_i"], 0) <= 1 or member_counts.get(m["node_j"], 0) <= 1
        }
    )
    rod_endpoints = []
    for rod in rods:
        rod_endpoints.append((rod.id, "start", rod.start))
        rod_endpoints.append((rod.id, "end", rod.end))
    close_unclustered = []
    lower = tolerance_mm
    upper = max(tolerance_mm * 2.5, tolerance_mm + 1.0)
    for i, a in enumerate(rod_endpoints):
        for b in rod_endpoints[i + 1 :]:
            d = float(np.linalg.norm(a[2] - b[2]))
            if lower < d <= upper:
                close_unclustered.append(
                    {
                        "rod_a": a[0],
                        "end_a": a[1],
                        "rod_b": b[0],
                        "end_b": b[1],
                        "distance_mm": d,
                    }
                )
    return {
        "member_counts": member_counts,
        "isolated_nodes": isolated_nodes,
        "single_member_nodes": single_member_nodes,
        "abnormal_nodes": abnormal_nodes,
        "dangling_members": dangling_members,
        "close_unclustered_endpoints": close_unclustered,
    }


def infer_supports(model: TrussModel) -> list[dict[str, Any]]:
    nodes = model.nodes
    x_min = float(np.min(nodes[:, 0]))
    x_max = float(np.max(nodes[:, 0]))
    span = x_max - x_min
    left_candidates = np.where(nodes[:, 0] <= x_min + 0.08 * max(span, 1.0))[0]
    right_candidates = np.where(nodes[:, 0] >= x_max - 0.08 * max(span, 1.0))[0]

    def lowest_pair(candidates: np.ndarray) -> list[int]:
        ordered = sorted(candidates.tolist(), key=lambda idx: (nodes[idx, 2], abs(nodes[idx, 1] - np.median(nodes[:, 1]))))
        return ordered[:2] if len(ordered) >= 2 else ordered[:1]

    left = lowest_pair(left_candidates)
    right = lowest_pair(right_candidates)
    supports: list[dict[str, Any]] = []
    for idx in left:
        supports.append({"node_id": idx, "restraints": ["Ux", "Uy", "Uz"], "type": "pin"})
    for idx in right:
        supports.append({"node_id": idx, "restraints": ["Ux", "Uz"], "type": "roller"})
    return supports


def resolve_supports(model: TrussModel, config: dict[str, Any]) -> list[dict[str, Any]]:
    overrides = config.get("manual_overrides", {}) or {}
    override_supports = overrides.get("support_nodes", {}) or {}
    fixed_override = override_supports.get("fixed") or []
    roller_override = override_supports.get("roller") or []
    if fixed_override or roller_override:
        supports = []
        for node_id in fixed_override:
            supports.append({"node_id": int(node_id), "restraints": ["Ux", "Uy", "Uz"], "type": "manual_fixed"})
        for node_id in roller_override:
            supports.append({"node_id": int(node_id), "restraints": ["Ux", "Uz"], "type": "manual_roller"})
        return supports
    support_cfg = config.get("supports", {})
    fixed_nodes = support_cfg.get("fixed_nodes") or []
    roller_nodes = support_cfg.get("roller_nodes") or []
    if fixed_nodes or roller_nodes:
        supports = []
        for node_id in fixed_nodes:
            supports.append({"node_id": int(node_id), "restraints": ["Ux", "Uy", "Uz"], "type": "fixed_pin"})
        for node_id in roller_nodes:
            supports.append({"node_id": int(node_id), "restraints": ["Ux", "Uz"], "type": "roller"})
        return supports
    manual = config.get("supports", {}).get("manual") or []
    if manual:
        supports = []
        for item in manual:
            supports.append(
                {
                    "node_id": int(item["node_id"]),
                    "restraints": list(item.get("restraints", ["Ux", "Uy", "Uz"])),
                    "type": item.get("type", "manual"),
                }
            )
        return supports
    return infer_supports(model)


def deck_nodes(model: TrussModel, config: dict[str, Any]) -> np.ndarray:
    nodes = model.nodes
    overrides = config.get("manual_overrides", {}) or {}
    override_deck_nodes = overrides.get("deck_nodes") or []
    if override_deck_nodes:
        return np.array([int(i) for i in override_deck_nodes if 0 <= int(i) < len(nodes)], dtype=int)
    bridge_cfg = config["bridge"]
    filter_cfg = bridge_cfg.get("deck_node_filter") or {}
    explicit = filter_cfg.get("node_ids")
    if explicit:
        return np.array([int(i) for i in explicit if 0 <= int(i) < len(nodes)], dtype=int)
    mask = np.ones(len(nodes), dtype=bool)
    z_range = filter_cfg.get("z_range_mm", bridge_cfg.get("deck_z_range_mm"))
    y_range = filter_cfg.get("y_range_mm", bridge_cfg.get("deck_y_range_mm"))
    x_range = filter_cfg.get("x_range_mm")
    if z_range:
        mask &= (nodes[:, 2] >= z_range[0]) & (nodes[:, 2] <= z_range[1])
    else:
        low = np.percentile(nodes[:, 2], 0)
        high = np.percentile(nodes[:, 2], 55)
        mask &= (nodes[:, 2] >= low - 1e-6) & (nodes[:, 2] <= high + 1e-6)
    if y_range:
        mask &= (nodes[:, 1] >= y_range[0]) & (nodes[:, 1] <= y_range[1])
    if x_range:
        mask &= (nodes[:, 0] >= x_range[0]) & (nodes[:, 0] <= x_range[1])
    selected = np.where(mask)[0]
    if len(selected) == 0:
        return np.arange(len(nodes))
    return selected


def assemble_stiffness(model: TrussModel, config: dict[str, Any]) -> np.ndarray:
    n_dof = len(model.nodes) * 3
    k_global = np.zeros((n_dof, n_dof), dtype=float)
    area = float(config["section"]["width_mm"]) * float(config["section"]["height_mm"])
    elastic = float(config["materials"]["elastic_modulus_mpa"])
    for member in model.members:
        ni = member["node_i"]
        nj = member["node_j"]
        p_i = model.nodes[ni]
        p_j = model.nodes[nj]
        vec = p_j - p_i
        length = float(np.linalg.norm(vec))
        if length <= 0:
            continue
        direction = vec / length
        k = area * elastic / length
        block = k * np.outer(direction, direction)
        ids_i = slice(3 * ni, 3 * ni + 3)
        ids_j = slice(3 * nj, 3 * nj + 3)
        k_global[ids_i, ids_i] += block
        k_global[ids_j, ids_j] += block
        k_global[ids_i, ids_j] -= block
        k_global[ids_j, ids_i] -= block
    return k_global


def load_vector(model: TrussModel, config: dict[str, Any], case: str, position_ratio: float | None = None) -> np.ndarray:
    forces = np.zeros(len(model.nodes) * 3, dtype=float)
    loads = config["loads"]
    g = float(loads["gravity_m_s2"])
    deck = deck_nodes(model, config)
    if len(deck) == 0:
        deck = np.arange(len(model.nodes))

    def add_vertical(node_ids: np.ndarray, total_n: float) -> None:
        if len(node_ids) == 0:
            return
        share = total_n / len(node_ids)
        for node_id in node_ids:
            forces[3 * int(node_id) + 2] -= share

    if loads.get("include_self_weight", True):
        total_mass = len(model.members) * float(config["materials"]["rod_mass_kg"])
        add_vertical(deck, total_mass * g)

    if case == "self_weight":
        pass
    elif case == "central_person":
        x_mid = (np.min(model.nodes[:, 0]) + np.max(model.nodes[:, 0])) / 2.0
        nearest = deck[np.argsort(np.abs(model.nodes[deck, 0] - x_mid))[: max(1, min(4, len(deck)))]]
        add_vertical(nearest, float(loads["central_person_count"]) * float(loads["person_mass_kg"]) * g)
    elif case == "distributed_people":
        add_vertical(deck, float(loads.get("distributed_person_count", 9)) * float(loads["person_mass_kg"]) * g)
    elif case == "eccentric":
        y = model.nodes[deck, 1]
        if loads.get("eccentric_side") == "negative_y":
            side = deck[y <= np.median(y)]
        else:
            side = deck[y >= np.median(y)]
        add_vertical(side, float(loads["distributed_person_count"]) * float(loads["person_mass_kg"]) * g)
    elif case == "moving":
        ratio = 0.5 if position_ratio is None else float(position_ratio)
        x_min = float(np.min(model.nodes[:, 0]))
        x_max = float(np.max(model.nodes[:, 0]))
        x_target = x_min + ratio * (x_max - x_min)
        nearest = deck[np.argsort(np.abs(model.nodes[deck, 0] - x_target))[: max(1, min(4, len(deck)))]]
        add_vertical(nearest, float(loads["moving_person_count"]) * float(loads["person_mass_kg"]) * g)
    return forces


def solve_load_case(model: TrussModel, config: dict[str, Any], supports: list[dict[str, Any]], case: str, position_ratio: float | None = None) -> LoadCaseResult:
    k_global = assemble_stiffness(model, config)
    force = load_vector(model, config, case, position_ratio)
    fixed: list[int] = []
    for support in supports:
        node_id = int(support["node_id"])
        for dof in support["restraints"]:
            fixed.append(3 * node_id + DOF_INDEX[dof])
    fixed = sorted(set(fixed))
    all_dofs = np.arange(len(model.nodes) * 3)
    free = np.array([d for d in all_dofs if d not in fixed], dtype=int)
    disp = np.zeros(len(model.nodes) * 3, dtype=float)
    if len(free) == 0:
        return LoadCaseResult(case, disp, np.zeros_like(disp), pd.DataFrame(), 0.0, 0.0, 0.0, True, 0, 0, float("inf"), False, "没有自由自由度")
    k_ff = k_global[np.ix_(free, free)]
    rank = int(np.linalg.matrix_rank(k_ff))
    singular = rank < len(free)
    condition_number = float(np.linalg.cond(k_ff)) if len(free) else float("inf")
    if singular:
        try:
            disp[free] = np.linalg.lstsq(k_ff, force[free], rcond=None)[0]
            reactions = k_global @ disp - force
            member_df = member_forces(model, config, disp)
            max_vertical = float(np.max(np.abs(disp.reshape((-1, 3))[:, 2])))
            max_force = float(member_df["axial_force_n"].abs().max()) if not member_df.empty else 0.0
            max_reaction = float(np.max(np.linalg.norm(reactions.reshape((-1, 3)), axis=1)))
            return LoadCaseResult(
                case,
                disp,
                reactions,
                member_df,
                max_vertical,
                max_force,
                max_reaction,
                True,
                rank,
                len(free),
                condition_number,
                False,
                f"刚度矩阵奇异: rank={rank}, free_dof={len(free)}。已输出最小二乘近似结果，仅供定位风险；模型可能存在机构、未连接杆件或支座约束不足。",
            )
        except Exception as exc:
            return LoadCaseResult(
                case,
                disp,
                np.zeros_like(disp),
                pd.DataFrame(),
                0.0,
                0.0,
                0.0,
                True,
                rank,
                len(free),
                condition_number,
                False,
                f"刚度矩阵奇异且近似求解失败: {exc}",
            )
    try:
        f_f = force[free]
        disp[free] = np.linalg.solve(k_ff, f_f)
        success = True
        message = "ok"
    except np.linalg.LinAlgError:
        return LoadCaseResult(case, disp, np.zeros_like(disp), pd.DataFrame(), 0.0, 0.0, 0.0, True, rank, len(free), condition_number, False, "刚度矩阵求解失败，结构可能不可解")

    reactions = k_global @ disp - force
    member_df = member_forces(model, config, disp)
    max_vertical = float(np.max(np.abs(disp.reshape((-1, 3))[:, 2])))
    max_force = float(member_df["axial_force_n"].abs().max()) if not member_df.empty else 0.0
    max_reaction = float(np.max(np.linalg.norm(reactions.reshape((-1, 3)), axis=1)))
    return LoadCaseResult(case, disp, reactions, member_df, max_vertical, max_force, max_reaction, singular, rank, len(free), condition_number, success, message)


def member_forces(model: TrussModel, config: dict[str, Any], displacements: np.ndarray) -> pd.DataFrame:
    area = float(config["section"]["width_mm"]) * float(config["section"]["height_mm"])
    elastic = float(config["materials"]["elastic_modulus_mpa"])
    b = float(config["section"]["width_mm"])
    h = float(config["section"]["height_mm"])
    imin = min(h * b**3 / 12.0, b * h**3 / 12.0)
    allow_t = float(config["materials"]["allowable_tension_mpa"])
    allow_c = float(config["materials"]["allowable_compression_mpa"])
    rows = []
    u = displacements.reshape((-1, 3))
    for member in model.members:
        ni = member["node_i"]
        nj = member["node_j"]
        vec = model.nodes[nj] - model.nodes[ni]
        length = float(np.linalg.norm(vec))
        direction = vec / length
        delta = float(np.dot(u[nj] - u[ni], direction))
        axial = area * elastic / length * delta
        stress = axial / area
        pcr = (np.pi**2 * elastic * imin) / (length**2)
        if axial >= 0:
            util = abs(stress) / allow_t if allow_t > 0 else np.inf
            buckling_util = 0.0
            force_type = "tension"
            risk = util
        else:
            util = abs(stress) / allow_c if allow_c > 0 else np.inf
            buckling_util = abs(axial) / pcr if pcr > 0 else np.inf
            force_type = "compression"
            risk = max(util, buckling_util)
        if force_type == "compression":
            if risk >= 1.0:
                risk_level = "危险"
            elif risk >= 0.75:
                risk_level = "高风险"
            elif risk >= 0.45:
                risk_level = "中风险"
            else:
                risk_level = "低风险"
        else:
            risk_level = "拉杆"
        rows.append(
            {
                "member_id": member["member_id"],
                "node_i": ni,
                "node_j": nj,
                "length_mm": length,
                "axial_force_n": axial,
                "force_type": force_type,
                "stress_mpa": stress,
                "stress_utilization": util,
                "euler_pcr_n": pcr,
                "buckling_utilization": buckling_util,
                "risk_score": risk,
                "risk_level": risk_level,
            }
        )
    return pd.DataFrame(rows).sort_values("risk_score", ascending=False)


def nodes_dataframe(model: TrussModel) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"node_id": i, "x_real_mm": p[0], "y_real_mm": p[1], "z_real_mm": p[2], "member_count": len(model.node_members.get(i, []))}
            for i, p in enumerate(model.nodes)
        ]
    )


def members_dataframe(model: TrussModel) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "member_id": m["member_id"],
                "node_i": m["node_i"],
                "node_j": m["node_j"],
                "real_length_mm": m["length_mm"],
                "model_length_mm": m["rod"].model_length_mm,
            }
            for m in model.members
        ]
    )


def geometry_checks(model: TrussModel, rods: list[Rod], config: dict[str, Any]) -> dict[str, Any]:
    nodes = model.nodes
    span = float(np.max(nodes[:, 0]) - np.min(nodes[:, 0]))
    width = float(np.max(nodes[:, 1]) - np.min(nodes[:, 1]))
    height = float(np.max(nodes[:, 2]) - np.min(nodes[:, 2]))
    bridge = config["bridge"]
    low_degree = [int(i) for i, members in model.node_members.items() if len(members) <= 1]
    compression_limit = 1.5 * float(config["section"]["standard_length_mm"])
    return {
        "span_mm": span,
        "width_mm": width,
        "height_mm": height,
        "span_model_mm": span / float(config["model"].get("scale", 10.0)),
        "width_model_mm": width / float(config["model"].get("scale", 10.0)),
        "height_model_mm": height / float(config["model"].get("scale", 10.0)),
        "span_ok": abs(span - float(bridge["target_span_mm"])) <= float(bridge["target_span_tolerance_mm"]),
        "width_ok": float(bridge["min_deck_width_mm"]) <= width <= float(bridge["max_deck_width_mm"]),
        "low_degree_nodes": low_degree,
        "very_long_members": [r.id for r in rods if r.length_mm > compression_limit],
        "missing_deck_x_bracing_todo": True,
        "missing_top_lateral_bracing_todo": True,
    }


def run_standard_cases(model: TrussModel, config: dict[str, Any], supports: list[dict[str, Any]]) -> tuple[dict[str, LoadCaseResult], pd.DataFrame]:
    results = {}
    for case in ["self_weight", "central_person", "distributed_people", "eccentric"]:
        results[case] = solve_load_case(model, config, supports, case)
    moving_rows = []
    steps = int(config["loads"].get("moving_steps", 15))
    for i in range(max(2, steps)):
        ratio = i / (max(2, steps) - 1)
        result = solve_load_case(model, config, supports, "moving", ratio)
        moving_rows.append(
            {
                "position_ratio": ratio,
                "max_vertical_displacement_mm": result.max_vertical_displacement_mm,
                "max_abs_force_n": result.max_abs_force_n,
                "success": result.success,
                "message": result.message,
            }
        )
    return results, pd.DataFrame(moving_rows)
