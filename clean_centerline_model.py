from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fem_truss_solver import TrussModel, build_truss_model, members_dataframe, node_diagnostics, nodes_dataframe


def manual_node_override_path(config: dict[str, Any], output_dir: Path) -> Path:
    centerline_cfg = config.get("centerline_model", {}) or {}
    value = centerline_cfg.get("manual_node_overrides_csv", "manual_node_overrides.csv")
    path = Path(value)
    if path.is_absolute():
        return path
    parent_candidate = output_dir.parent / value
    if parent_candidate.exists():
        return parent_candidate
    return output_dir / value


def write_manual_node_overrides_template(path: Path) -> None:
    if path.exists():
        return
    rows = [
        {
            "action": "merge_nodes",
            "node_ids": "1;2",
            "target_node_id": "1",
            "member_id": "",
            "end": "",
            "node_id": "",
            "note": "action 可选: merge_nodes, force_connect, no_connect。force_connect 需要 member_id/end/node_id。",
        },
        {
            "action": "force_connect",
            "node_ids": "",
            "target_node_id": "",
            "member_id": "3",
            "end": "start",
            "node_id": "1",
            "note": "把指定杆端强制连接到某节点；end 可选 start 或 end。",
        },
        {
            "action": "no_connect",
            "node_ids": "",
            "target_node_id": "",
            "member_id": "",
            "end": "",
            "node_id": "",
            "note": "记录交叉但不连接的位置；当前中心线模型默认不会在交叉处自动增加节点。",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def load_manual_node_overrides(path: Path | None) -> pd.DataFrame:
    columns = ["action", "node_ids", "target_node_id", "member_id", "end", "node_id", "note"]
    if path is None or not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"manual_node_overrides.csv 缺少字段: {sorted(missing)}")
    df["action"] = df["action"].fillna("").astype(str).str.strip()
    return df


def _parse_int_list(value: Any) -> list[int]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    items = text.replace(",", ";").replace("，", ";").split(";")
    return [int(float(item)) for item in items if item.strip()]


class _UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union_to(self, values: list[int], target: int) -> None:
        root = self.find(target)
        for value in values:
            self.parent[self.find(value)] = root


def apply_manual_node_overrides(model: TrussModel, override_df: pd.DataFrame) -> tuple[TrussModel, dict[str, Any]]:
    uf = _UnionFind(len(model.nodes))
    merge_rows: list[dict[str, Any]] = []
    force_rows: list[dict[str, Any]] = []
    no_connect_rows: list[dict[str, Any]] = []
    ignored_rows: list[dict[str, Any]] = []

    for row in override_df.to_dict("records"):
        action = str(row.get("action", "") or "").strip()
        if action == "merge_nodes":
            node_ids = [idx for idx in _parse_int_list(row.get("node_ids")) if 0 <= idx < len(model.nodes)]
            target_raw = row.get("target_node_id")
            target = int(float(target_raw)) if str(target_raw).strip() not in {"", "nan", "None"} else (node_ids[0] if node_ids else -1)
            if node_ids and 0 <= target < len(model.nodes):
                uf.union_to(node_ids, target)
                merge_rows.append({"target_node_id": target, "node_ids": node_ids, "note": row.get("note", "")})
        elif action == "force_connect":
            try:
                member_id = int(float(row.get("member_id")))
                end = str(row.get("end", "")).strip().lower()
                node_id = int(float(row.get("node_id")))
                if end in {"start", "end"} and 0 <= node_id < len(model.nodes):
                    force_rows.append({"member_id": member_id, "end": end, "node_id": node_id, "note": row.get("note", "")})
            except Exception:
                ignored_rows.append({"action": action, "reason": "force_connect 参数无效", "row": row})
        elif action == "no_connect":
            no_connect_rows.append(row)
        elif action:
            ignored_rows.append({"action": action, "reason": "未知 action", "row": row})

    root_to_old: dict[int, list[int]] = {}
    for idx in range(len(model.nodes)):
        root_to_old.setdefault(uf.find(idx), []).append(idx)
    root_to_new: dict[int, int] = {}
    new_nodes: list[np.ndarray] = []
    for root, old_ids in sorted(root_to_old.items(), key=lambda item: min(item[1])):
        root_to_new[root] = len(new_nodes)
        new_nodes.append(np.mean(model.nodes[old_ids], axis=0))
    old_to_new = {old_id: root_to_new[uf.find(old_id)] for old_id in range(len(model.nodes))}

    force_lookup = {(row["member_id"], row["end"]): row["node_id"] for row in force_rows}
    members: list[dict[str, Any]] = []
    node_members: dict[int, list[int]] = {}
    skipped_zero_length: list[int] = []
    for member in model.members:
        member_id = int(member["member_id"])
        node_i = old_to_new[int(member["node_i"])]
        node_j = old_to_new[int(member["node_j"])]
        if (member_id, "start") in force_lookup:
            node_i = old_to_new[force_lookup[(member_id, "start")]]
        if (member_id, "end") in force_lookup:
            node_j = old_to_new[force_lookup[(member_id, "end")]]
        if node_i == node_j:
            skipped_zero_length.append(member_id)
            continue
        p_i = new_nodes[node_i]
        p_j = new_nodes[node_j]
        member_index = len(members)
        clean_member = dict(member)
        clean_member["node_i"] = node_i
        clean_member["node_j"] = node_j
        clean_member["length_mm"] = float(np.linalg.norm(p_j - p_i))
        members.append(clean_member)
        node_members.setdefault(node_i, []).append(member_index)
        node_members.setdefault(node_j, []).append(member_index)

    clean_model = TrussModel(np.array(new_nodes, dtype=float), members, node_members)
    info = {
        "manual_node_overrides_used": bool(merge_rows or force_rows or no_connect_rows),
        "merge_rows": merge_rows,
        "force_connect_rows": force_rows,
        "no_connect_rows": no_connect_rows,
        "ignored_rows": ignored_rows,
        "skipped_zero_length_members_after_overrides": skipped_zero_length,
        "initial_node_count": len(model.nodes),
        "clean_node_count": len(clean_model.nodes),
        "initial_member_count": len(model.members),
        "clean_member_count": len(clean_model.members),
    }
    return clean_model, info


def build_clean_centerline_model(
    rods: list[Any],
    config: dict[str, Any],
    output_dir: Path,
    tolerance_mm: float,
) -> tuple[TrussModel, dict[str, Any], dict[str, Any], pd.DataFrame]:
    initial_model = build_truss_model(rods, tolerance_mm)
    template_path = output_dir / "manual_node_overrides_template.csv"
    write_manual_node_overrides_template(template_path)
    override_path = manual_node_override_path(config, output_dir)
    override_df = load_manual_node_overrides(override_path if override_path.exists() else None)
    clean_model, override_info = apply_manual_node_overrides(initial_model, override_df)
    diagnostics = node_diagnostics(clean_model, rods, tolerance_mm)
    override_info["manual_node_overrides_csv_used"] = str(override_path) if override_path.exists() else "无"
    override_info["manual_node_overrides_template"] = str(template_path)
    return clean_model, diagnostics, override_info, override_df


def write_clean_centerline_outputs(model: TrussModel, output_dir: Path) -> None:
    nodes_dataframe(model).rename(columns={"x_real_mm": "x_mm", "y_real_mm": "y_mm", "z_real_mm": "z_mm"}).to_csv(
        output_dir / "clean_nodes.csv", index=False
    )
    members = members_dataframe(model).rename(columns={"real_length_mm": "length_mm"})
    if "member_type" not in members.columns:
        members["member_type"] = "wood"
    if "area_mm2" not in members.columns:
        members["area_mm2"] = 2400.0
    if "E_MPa" not in members.columns:
        members["E_MPa"] = 10000.0
    members.to_csv(output_dir / "clean_members.csv", index=False)


def write_node_quality_report(path: Path, diagnostics: dict[str, Any], override_info: dict[str, Any]) -> None:
    counts = diagnostics.get("member_counts", {})
    abnormal_nodes = diagnostics.get("abnormal_nodes", [])
    close = diagnostics.get("close_unclustered_endpoints", [])
    lines = [
        "# Node Quality Report",
        "",
        "## 中心线模型",
        f"- 初始节点数: {override_info['initial_node_count']}",
        f"- clean 节点数: {override_info['clean_node_count']}",
        f"- 初始杆件数: {override_info['initial_member_count']}",
        f"- clean 杆件数: {override_info['clean_member_count']}",
        f"- manual_node_overrides.csv: {override_info['manual_node_overrides_csv_used']}",
        f"- 人工合并节点: {override_info['merge_rows'] or '无'}",
        f"- 强制连接杆端: {override_info['force_connect_rows'] or '无'}",
        f"- 指定交叉不连接: {len(override_info['no_connect_rows'])} 条",
        f"- 修正后退化为零长度的杆件: {override_info['skipped_zero_length_members_after_overrides'] or '无'}",
        "",
        "## 节点质量问题",
        f"- 孤立节点: {diagnostics.get('isolated_nodes', []) or '无'}",
        f"- 单杆节点: {diagnostics.get('single_member_nodes', []) or '无'}",
        f"- 悬空杆件: {diagnostics.get('dangling_members', []) or '无'}",
        f"- 未连接但距离接近的杆端数量: {len(close)}",
        f"- 连接数异常节点: {abnormal_nodes or '无'}",
        "",
        "## 前 30 个问题节点",
    ]
    for node_id in abnormal_nodes[:30]:
        lines.append(f"- node {node_id}: 连接杆件数 {counts.get(node_id, 0)}")
    if not abnormal_nodes:
        lines.append("- 无")
    if close:
        lines.extend(["", "## 前 30 组近距离未连接杆端"])
        for item in close[:30]:
            lines.append(
                f"- member {item['rod_a']} {item['end_a']} 与 member {item['rod_b']} {item['end_b']}: {item['distance_mm']:.1f} mm"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def manual_fem_inputs_confirmed(config: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    overrides = config.get("manual_overrides", {}) or {}
    support_overrides = overrides.get("support_nodes", {}) or {}
    manual_fixed = support_overrides.get("fixed") or []
    manual_roller = support_overrides.get("roller") or []
    manual_deck = overrides.get("deck_nodes") or []

    support_cfg = config.get("supports", {}) or {}
    fixed_nodes = support_cfg.get("fixed_nodes") or []
    roller_nodes = support_cfg.get("roller_nodes") or []
    manual_supports = support_cfg.get("manual") or []

    deck_filter = (config.get("bridge", {}) or {}).get("deck_node_filter", {}) or {}
    deck_filter_nodes = deck_filter.get("node_ids") or []

    fixed_confirmed = bool(manual_fixed or fixed_nodes or manual_supports)
    roller_confirmed = bool(manual_roller or roller_nodes or manual_supports)
    deck_confirmed = bool(manual_deck or deck_filter_nodes)
    info = {
        "fixed_nodes_confirmed": fixed_confirmed,
        "roller_nodes_confirmed": roller_confirmed,
        "deck_nodes_confirmed": deck_confirmed,
        "manual_fixed_nodes": manual_fixed or fixed_nodes,
        "manual_roller_nodes": manual_roller or roller_nodes,
        "manual_deck_nodes": manual_deck or deck_filter_nodes,
        "message": "已人工确认支座和桥面加载节点。" if fixed_confirmed and roller_confirmed and deck_confirmed else "请先人工确认支座和桥面加载节点。",
    }
    return fixed_confirmed and roller_confirmed and deck_confirmed, info


def confidence_scores(
    metadata: Any,
    material_summary: dict[str, Any],
    diagnostics: dict[str, Any],
    support_load_info: dict[str, Any],
    fem_ran: bool,
    governing: Any,
) -> dict[str, int]:
    def clamp(value: int) -> int:
        return max(0, min(100, int(value)))

    geometry = 75 + (15 if getattr(metadata, "detected_standard_1_to_10", False) else 0)
    material = 75 + (15 if material_summary.get("manual_overrides_csv_used") != "无" else 0)
    node_network = 100 - 4 * len(diagnostics.get("single_member_nodes", [])) - 2 * len(diagnostics.get("close_unclustered_endpoints", []))
    support = 100 if support_load_info.get("fixed_nodes_confirmed") and support_load_info.get("roller_nodes_confirmed") else 25
    load = 100 if support_load_info.get("deck_nodes_confirmed") else 25
    fem = 80 if fem_ran and getattr(governing, "success", False) else (35 if fem_ran else 0)
    return {
        "geometry_recognition_score": clamp(geometry),
        "material_count_score": clamp(material),
        "node_network_score": clamp(node_network),
        "support_definition_score": clamp(support),
        "load_definition_score": clamp(load),
        "fem_result_score": clamp(fem),
    }
