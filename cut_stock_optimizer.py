from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import math
import pandas as pd


@dataclass
class MaterialPiece:
    piece_id: str
    member_id: int
    source_member_ids: list[int]
    length_mm: float
    required_length_mm: float
    note: str


def write_manual_overrides_template(path: Path) -> None:
    if path.exists():
        return
    rows = [
        {
            "member_id": "",
            "action": "keep",
            "corrected_length_mm": "",
            "group_id": "",
            "note": "action 可选: keep, ignore, merge, shorten, duplicate_of, non_structural",
        }
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def load_manual_member_overrides(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["member_id", "action", "corrected_length_mm", "group_id", "note"])
    df = pd.read_csv(path)
    required = {"member_id", "action", "corrected_length_mm", "group_id", "note"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manual_member_overrides.csv 缺少字段: {sorted(missing)}")
    df["action"] = df["action"].fillna("keep").astype(str).str.strip()
    return df


def apply_manual_member_overrides(rods: list[Any], override_df: pd.DataFrame) -> tuple[pd.DataFrame, list[MaterialPiece], dict[str, Any]]:
    override_by_id = {}
    for row in override_df.to_dict("records"):
        if pd.isna(row.get("member_id")) or str(row.get("member_id")).strip() == "":
            continue
        override_by_id[int(row["member_id"])] = row

    inventory_rows: list[dict[str, Any]] = []
    raw_pieces: list[MaterialPiece] = []
    merge_groups: dict[str, list[tuple[Any, dict[str, Any]]]] = {}
    ignored_ids: list[int] = []
    duplicate_ids: list[int] = []
    shortened_ids: list[int] = []

    for rod in rods:
        override = override_by_id.get(int(rod.id), {})
        action = str(override.get("action", "keep") or "keep").strip()
        corrected = override.get("corrected_length_mm", "")
        corrected_length = float(corrected) if str(corrected).strip() not in {"", "nan", "None"} else float(rod.length_mm)
        group_id = str(override.get("group_id", "") or "").strip()
        note = str(override.get("note", "") or "").strip()
        include = True
        effective_length = corrected_length

        if action in {"ignore", "non_structural"}:
            include = False
            ignored_ids.append(int(rod.id))
        elif action == "duplicate_of":
            include = False
            duplicate_ids.append(int(rod.id))
        elif action == "shorten":
            include = True
            shortened_ids.append(int(rod.id))
        elif action == "merge":
            include = False
            if not group_id:
                group_id = f"merge_{rod.id}"
            merge_groups.setdefault(group_id, []).append((rod, override))
        elif action == "keep":
            include = True
        else:
            include = True
            note = (note + " | " if note else "") + f"未知 action={action}，按 keep 处理"

        inventory_rows.append(
            {
                "member_id": int(rod.id),
                "original_length_mm": float(rod.length_mm),
                "corrected_length_mm": effective_length,
                "action": action,
                "group_id": group_id,
                "included_in_stock_count": include,
                "note": note,
            }
        )
        if include:
            raw_pieces.append(
                MaterialPiece(
                    piece_id=f"M{rod.id}",
                    member_id=int(rod.id),
                    source_member_ids=[int(rod.id)],
                    length_mm=effective_length,
                    required_length_mm=effective_length,
                    note=note,
                )
            )

    for group_id, items in merge_groups.items():
        source_ids = [int(rod.id) for rod, _ in items]
        length = sum(
            float(row.get("corrected_length_mm"))
            if str(row.get("corrected_length_mm", "")).strip() not in {"", "nan", "None"}
            else float(rod.length_mm)
            for rod, row in items
        )
        raw_pieces.append(
            MaterialPiece(
                piece_id=f"G{group_id}",
                member_id=source_ids[0],
                source_member_ids=source_ids,
                length_mm=length,
                required_length_mm=length,
                note=f"merge group {group_id}: {source_ids}",
            )
        )

    summary = {
        "manual_override_rows": len(override_by_id),
        "ignored_member_ids": ignored_ids,
        "duplicate_member_ids": duplicate_ids,
        "shortened_member_ids": shortened_ids,
        "merge_group_count": len(merge_groups),
    }
    return pd.DataFrame(inventory_rows), raw_pieces, summary


def split_overlength_piece(piece: MaterialPiece, standard_length_mm: float, allow_splicing: bool) -> tuple[list[MaterialPiece], list[dict[str, Any]]]:
    if piece.required_length_mm <= standard_length_mm + 0.5:
        piece.required_length_mm = min(piece.required_length_mm, standard_length_mm)
        piece.length_mm = min(piece.length_mm, standard_length_mm)
        return [piece], []
    issue = {
        "member_id": piece.member_id,
        "source_member_ids": piece.source_member_ids,
        "length_mm": piece.required_length_mm,
        "status": "需要拼接或重新设计" if allow_splicing else "设计问题：不允许拼接",
        "suggested_split": "",
    }
    if not allow_splicing:
        return [], [issue]
    segment_count = int(math.ceil(piece.required_length_mm / standard_length_mm))
    base = piece.required_length_mm / segment_count
    pieces = []
    for idx in range(segment_count):
        segment_length = min(standard_length_mm, base)
        pieces.append(
            MaterialPiece(
                piece_id=f"{piece.piece_id}_S{idx + 1}",
                member_id=piece.member_id,
                source_member_ids=piece.source_member_ids,
                length_mm=segment_length,
                required_length_mm=segment_length,
                note=(piece.note + " | " if piece.note else "") + f"overlength split {idx + 1}/{segment_count}",
            )
        )
    issue["suggested_split"] = " + ".join(f"{p.required_length_mm:.1f}mm" for p in pieces)
    return pieces, [issue]


def first_fit_decreasing(
    pieces: list[MaterialPiece],
    standard_length_mm: float,
    saw_kerf_mm: float,
    trim_allowance_mm: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    def required_with_allowance(piece: MaterialPiece) -> float:
        length = float(piece.required_length_mm)
        if length + trim_allowance_mm <= standard_length_mm + 1e-6:
            return length + trim_allowance_mm
        return length

    bins: list[dict[str, Any]] = []
    sorted_pieces = sorted(pieces, key=required_with_allowance, reverse=True)
    for piece in sorted_pieces:
        required = required_with_allowance(piece)
        placed = False
        for bin_item in bins:
            kerf = saw_kerf_mm if bin_item["pieces"] else 0.0
            if bin_item["used_mm"] + kerf + required <= standard_length_mm + 1e-6:
                bin_item["pieces"].append((piece, kerf, required))
                bin_item["used_mm"] += kerf + required
                placed = True
                break
        if not placed:
            if required > standard_length_mm + 1e-6:
                raise ValueError(f"构件 {piece.piece_id} 长度 {required:.1f}mm 超过标准木杆，无法排料")
            bins.append({"used_mm": required, "pieces": [(piece, 0.0, required)]})

    rows = []
    for stock_idx, bin_item in enumerate(bins, start=1):
        for order, (piece, kerf, required) in enumerate(bin_item["pieces"], start=1):
            rows.append(
                {
                    "stock_id": stock_idx,
                    "piece_order": order,
                    "piece_id": piece.piece_id,
                    "member_id": piece.member_id,
                    "source_member_ids": ";".join(str(x) for x in piece.source_member_ids),
                    "piece_length_mm": piece.length_mm,
                    "trim_allowance_mm": trim_allowance_mm,
                    "required_length_mm": required,
                    "kerf_before_mm": kerf,
                    "stock_used_mm": bin_item["used_mm"],
                    "stock_waste_mm": standard_length_mm - bin_item["used_mm"],
                    "note": piece.note,
                }
            )
    summary = {
        "stock_wood_count": len(bins),
        "total_piece_count": len(pieces),
        "total_required_length_mm": sum(float(p.required_length_mm) + trim_allowance_mm for p in pieces),
        "total_waste_mm": sum(standard_length_mm - b["used_mm"] for b in bins),
        "saw_kerf_mm": saw_kerf_mm,
        "trim_allowance_mm": trim_allowance_mm,
    }
    return pd.DataFrame(rows), summary


def optimize_cut_stock(rods: list[Any], config: dict[str, Any], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    material_cfg = config.get("material_count", {}) or {}
    standard_length = float(config["section"]["standard_length_mm"])
    saw_kerf = float(material_cfg.get("saw_kerf_mm", 3.0))
    trim_allowance = float(material_cfg.get("trim_allowance_mm", 5.0))
    allow_splicing = bool(material_cfg.get("allow_splicing", True))
    manual_path_value = material_cfg.get("manual_member_overrides_csv", "manual_member_overrides.csv")
    manual_path = Path(manual_path_value)
    if not manual_path.is_absolute():
        manual_path = output_dir.parent / manual_path
        if not manual_path.exists():
            manual_path = output_dir / manual_path_value

    write_manual_overrides_template(output_dir / "manual_member_overrides_template.csv")
    override_df = load_manual_member_overrides(manual_path if manual_path.exists() else None)
    corrected_inventory, raw_pieces, manual_summary = apply_manual_member_overrides(rods, override_df)

    pieces: list[MaterialPiece] = []
    overlength_issues: list[dict[str, Any]] = []
    for piece in raw_pieces:
        split_pieces, issues = split_overlength_piece(piece, standard_length, allow_splicing)
        pieces.extend(split_pieces)
        overlength_issues.extend(issues)
    cut_plan_df, cut_summary = first_fit_decreasing(pieces, standard_length, saw_kerf, trim_allowance)
    summary = {
        **cut_summary,
        **manual_summary,
        "manual_overrides_csv_used": str(manual_path) if manual_path.exists() else "无",
        "allow_splicing": allow_splicing,
        "structural_member_count": len(raw_pieces),
        "raw_material_score": 20 + (int(config["bridge"]["base_rod_count"]) - cut_summary["stock_wood_count"]),
        "capped_material_score": min(20, 20 + (int(config["bridge"]["base_rod_count"]) - cut_summary["stock_wood_count"])),
        "user_manual_count": int(material_cfg.get("user_manual_count", 46)),
    }
    return corrected_inventory, cut_plan_df, summary, overlength_issues


def write_cut_plan_markdown(path: Path, cut_plan_df: pd.DataFrame, summary: dict[str, Any], overlength_issues: list[dict[str, Any]]) -> None:
    lines = [
        "# Cut Plan",
        "",
        f"- stock_wood_count: {summary['stock_wood_count']}",
        f"- saw_kerf_mm: {summary['saw_kerf_mm']}",
        f"- trim_allowance_mm: {summary['trim_allowance_mm']}",
        f"- total_waste_mm: {summary['total_waste_mm']:.1f}",
        "",
        "## 超长杆件",
    ]
    if not overlength_issues:
        lines.append("- 无")
    else:
        for issue in overlength_issues:
            lines.append(
                f"- member {issue['member_id']}: {issue['length_mm']:.1f}mm，{issue['status']}，建议拆分: {issue['suggested_split'] or '无'}"
            )
    lines.append("")
    lines.append("## 排料明细")
    for stock_id, group in cut_plan_df.groupby("stock_id"):
        pieces = ", ".join(f"{row.piece_id}({row.required_length_mm:.1f}mm)" for row in group.itertuples())
        waste = float(group.iloc[0]["stock_waste_mm"])
        lines.append(f"- stock {stock_id}: {pieces}; waste {waste:.1f}mm")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_material_count_comparison(path: Path, model_member_count: int, raw_equivalent_count: int, material_summary: dict[str, Any], overlength_issues: list[dict[str, Any]]) -> None:
    lines = [
        "# Material Count Comparison",
        "",
        f"- 程序原始识别 model_member_count: {model_member_count}",
        f"- 程序原始等效标准杆: {raw_equivalent_count}",
        f"- 用户人工统计: {material_summary['user_manual_count']}",
        f"- 修正后 structural_member_count: {material_summary['structural_member_count']}",
        f"- 修正后 stock_wood_count: {material_summary['stock_wood_count']}",
        f"- raw_material_score: {material_summary['raw_material_score']}",
        f"- capped_material_score: {material_summary['capped_material_score']}",
        "",
        "## 差异原因",
        "- 原始等效标准杆按每个模型构件单独计算，会把多段短杆各算成一根标准木杆。",
        "- V1.6.1 使用 1300mm 标准木杆排料，多段短杆可以来自同一根标准木杆。",
        "- manual_member_overrides.csv 可进一步排除辅助线、重复杆件，或修正有效长度。",
        "- 超长构件需要拼接或重新设计，不能静默只按 ceil 统计。",
        "",
        "## 超长杆件",
    ]
    if not overlength_issues:
        lines.append("- 无")
    else:
        for issue in overlength_issues:
            lines.append(f"- member {issue['member_id']}: {issue['length_mm']:.1f}mm，{issue['status']}")
    lines.extend(
        [
            "",
            "## 建议采用数字",
            "- 材料成本分建议采用 stock_wood_count，而不是 model_member_count。",
            "- 如果课程评分材料成本分上限为 20 分，则采用 capped_material_score；否则可展示 raw_material_score。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
