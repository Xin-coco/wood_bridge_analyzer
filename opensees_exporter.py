from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


UNAVAILABLE_MESSAGE = "OpenSeesPy backend unavailable. Install with: pip install openseespy"


def opensees_requested(config: dict[str, Any]) -> bool:
    return str(config.get("solver_backend", "numpy")).strip().lower() in {"openseespy", "both"}


def load_clean_tables(output_dir: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes_path = output_dir / "clean_nodes.csv"
    members_path = output_dir / "clean_members.csv"
    if not nodes_path.exists() or not members_path.exists():
        raise FileNotFoundError("OpenSeesPy backend requires clean_nodes.csv and clean_members.csv.")
    nodes = pd.read_csv(nodes_path)
    members = pd.read_csv(members_path)
    required_nodes = {"node_id", "x_mm", "y_mm", "z_mm"}
    required_members = {"member_id", "node_i", "node_j"}
    missing_nodes = required_nodes - set(nodes.columns)
    missing_members = required_members - set(members.columns)
    if missing_nodes:
        raise ValueError(f"clean_nodes.csv missing columns: {sorted(missing_nodes)}")
    if missing_members:
        raise ValueError(f"clean_members.csv missing columns: {sorted(missing_members)}")

    area = float((config.get("opensees", {}).get("material", {}) or {}).get("area_mm2", float(config["section"]["width_mm"]) * float(config["section"]["height_mm"])))
    elastic = float((config.get("opensees", {}).get("material", {}) or {}).get("E_MPa", config["materials"]["elastic_modulus_mpa"]))
    if "length_mm" not in members.columns:
        if "real_length_mm" in members.columns:
            members["length_mm"] = members["real_length_mm"]
        else:
            lookup = {int(r.node_id): np.array([r.x_mm, r.y_mm, r.z_mm], dtype=float) for r in nodes.itertuples()}
            members["length_mm"] = [
                float(np.linalg.norm(lookup[int(r.node_j)] - lookup[int(r.node_i)]))
                for r in members.itertuples()
            ]
    if "member_type" not in members.columns:
        members["member_type"] = "wood"
    if "area_mm2" not in members.columns:
        members["area_mm2"] = area
    if "E_MPa" not in members.columns:
        members["E_MPa"] = elastic
    return nodes, members


def support_nodes_from_config(config: dict[str, Any]) -> tuple[list[int], list[int]]:
    overrides = config.get("manual_overrides", {}) or {}
    support_overrides = overrides.get("support_nodes", {}) or {}
    fixed = support_overrides.get("fixed") or config.get("fixed_nodes") or (config.get("supports", {}) or {}).get("fixed_nodes") or []
    roller = support_overrides.get("roller") or config.get("roller_nodes") or (config.get("supports", {}) or {}).get("roller_nodes") or []
    return [int(x) for x in fixed], [int(x) for x in roller]


def deck_nodes_from_config(config: dict[str, Any]) -> list[int]:
    overrides = config.get("manual_overrides", {}) or {}
    if overrides.get("deck_nodes"):
        return [int(x) for x in overrides.get("deck_nodes")]
    if config.get("deck_nodes"):
        return [int(x) for x in config.get("deck_nodes")]
    bridge = config.get("bridge", {}) or {}
    deck_filter = bridge.get("deck_node_filter", {}) or {}
    explicit = deck_filter.get("node_ids") or []
    return [int(x) for x in explicit]


def support_check(nodes: pd.DataFrame, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fixed, roller = support_nodes_from_config(config)
    node_ids = set(int(x) for x in nodes["node_id"].tolist())
    issues = []
    if not fixed:
        issues.append({"status": "fail", "check": "fixed_nodes", "message": "fixed_nodes 未指定。"})
    if not roller:
        issues.append({"status": "fail", "check": "roller_nodes", "message": "roller_nodes 未指定。"})
    invalid = [x for x in fixed + roller if x not in node_ids]
    if invalid:
        issues.append({"status": "fail", "check": "node_ids_exist", "message": f"支座节点不存在: {invalid}"})
    if not issues:
        issues.append({"status": "pass", "check": "support_nodes", "message": "OpenSees 支座节点已指定。"})
    return issues, {"fixed_nodes": fixed, "roller_nodes": roller, "ok": not any(i["status"] == "fail" for i in issues)}


def load_check(nodes: pd.DataFrame, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    deck = deck_nodes_from_config(config)
    node_ids = set(int(x) for x in nodes["node_id"].tolist())
    issues = []
    if not deck:
        issues.append({"status": "fail", "check": "deck_nodes", "message": "deck_nodes 未指定，OpenSees 荷载不能施加。"})
    invalid = [x for x in deck if x not in node_ids]
    if invalid:
        issues.append({"status": "fail", "check": "node_ids_exist", "message": f"桥面加载节点不存在: {invalid}"})
    if not issues:
        issues.append({"status": "pass", "check": "deck_nodes", "message": "OpenSees 桥面加载节点已指定。"})
    return issues, {"deck_nodes": deck, "ok": not any(i["status"] == "fail" for i in issues)}


def write_check_report(path: Path, title: str, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Checks"])
    for row in rows:
        lines.append(f"- [{row['status']}] {row['check']}: {row['message']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def load_case_names(config: dict[str, Any]) -> list[str]:
    opensees = config.get("opensees", {}) or {}
    analysis = opensees.get("analysis", {}) or {}
    return list(analysis.get("load_cases") or ["self_weight", "midspan_person", "group_uniform", "eccentric_walk"])


def load_vector_for_case(nodes: pd.DataFrame, members: pd.DataFrame, deck_nodes: list[int], config: dict[str, Any], case: str) -> tuple[dict[int, tuple[float, float, float]], float]:
    loads = config.get("loads", {}) or {}
    opensees_cfg = config.get("opensees", {}) or {}
    material = opensees_cfg.get("material", {}) or {}
    gravity = float(loads.get("gravity_m_s2", 9.81))
    person_mass = float(loads.get("person_mass_kg", 70.0))
    people = float(loads.get("distributed_person_count", 9))
    deck = [int(x) for x in deck_nodes]
    nodal = {node_id: [0.0, 0.0, 0.0] for node_id in deck}

    def add_vertical(node_ids: list[int], total_n: float) -> None:
        if not node_ids:
            return
        share = -float(total_n) / len(node_ids)
        for node_id in node_ids:
            nodal.setdefault(node_id, [0.0, 0.0, 0.0])
            nodal[node_id][2] += share

    total_load = 0.0
    density = float(material.get("density_kg_per_m3", config["materials"].get("wood_density_kg_m3", 500.0)))
    include_self_weight = bool(loads.get("include_self_weight", True))
    if include_self_weight:
        area = members["area_mm2"].astype(float)
        length = members["length_mm"].astype(float)
        mass_kg = float(np.sum(area * length) * 1e-9 * density)
        weight_n = mass_kg * gravity
        add_vertical(deck, weight_n)
        total_load += weight_n

    if case == "self_weight":
        pass
    elif case == "midspan_person":
        node_lookup = nodes.set_index("node_id")
        deck_sorted = sorted(deck, key=lambda n: abs(float(node_lookup.loc[n, "x_mm"]) - float(nodes["x_mm"].mean())))
        target = deck_sorted[: max(1, min(4, len(deck_sorted)))]
        load_n = person_mass * gravity
        add_vertical(target, load_n)
        total_load += load_n
    elif case == "group_uniform":
        load_n = people * person_mass * gravity
        add_vertical(deck, load_n)
        total_load += load_n
    elif case == "eccentric_walk":
        node_lookup = nodes.set_index("node_id")
        y_values = [float(node_lookup.loc[n, "y_mm"]) for n in deck]
        median_y = float(np.median(y_values)) if y_values else 0.0
        side = [n for n in deck if float(node_lookup.loc[n, "y_mm"]) >= median_y]
        load_n = people * person_mass * gravity
        add_vertical(side or deck, load_n)
        total_load += load_n
    return {node: tuple(values) for node, values in nodal.items()}, total_load
