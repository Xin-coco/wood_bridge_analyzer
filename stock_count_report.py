from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import pandas as pd


def write_paired_cut_plan_markdown(path: Path, plan_df: pd.DataFrame) -> None:
    lines = ["# 近似长度两两配对排料方案", ""]
    counted = plan_df[plan_df["pairing_type"] != "ignored"] if not plan_df.empty else pd.DataFrame()
    if counted.empty:
        lines.append("- 无可用排料结果。")
    else:
        for row in counted.itertuples():
            member_ids = [x for x in str(row.member_ids).split(";") if x]
            lengths = [float(x) for x in str(row.member_lengths_mm).split(";") if x]
            pieces = " + ".join(f"构件 {member_ids[i]}（{lengths[i]:.0f}mm）" for i in range(len(member_ids)))
            lines.append(
                f"第 {int(row.stock_id)} 根标准木杆：{pieces} = {float(row.total_used_mm):.0f}mm，余料 {float(row.waste_mm):.0f}mm"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_material_stock_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# 标准木杆近似排料统计",
        "",
        f"- 模型识别杆件数量: {summary['model_member_count']}",
        f"- 有效木杆结构段数: {summary['effective_wood_segment_count']}",
        f"- 参与材料统计杆件数量: {summary['included_member_count']}",
        f"- 取近似后的长度种类: {summary['rounded_length_type_count']}",
        f"- 配对成功的标准木杆数量: {summary['paired_stock_count']}",
        f"- 单独占用标准木杆数量: {summary['single_stock_count']}",
        f"- 超长杆件数量: {summary['oversized_member_count']}",
        f"- 最终 stock_wood_count: {summary['stock_wood_count']}",
        f"- 人工统计 manual_stock_count: {summary['manual_stock_count']}",
        f"- 程序统计与人工统计差异: {summary['stock_count_difference_vs_manual']}",
        f"- raw_material_score: {summary['raw_material_score']}",
        f"- capped_material_score: {summary['capped_material_score']}",
        f"- score_policy_note: {summary['score_policy_note']}",
        f"- 余料总量: {summary['total_waste_mm']:.1f} mm",
        f"- 平均余料: {summary['average_waste_mm']:.1f} mm",
        f"- 配对成功率: {summary['pair_success_rate']:.1%}",
        f"- 无法配对杆件列表: {summary['unpaired_member_ids'] or '无'}",
        f"- 超长杆件列表: {summary['oversized_member_ids'] or '无'}",
        f"- 是否建议采用该统计结果: {'建议采用' if summary['recommended_to_use'] else '不建议采用'}",
        "",
        "## 重要限制",
        "- 近似排料统计用于材料成本估算，不用于 OpenSeesPy 结构计算。",
        "- OpenSeesPy 仍使用 clean_members.csv 中的真实几何长度。",
        "- 长度近似和两两配对会受到锯缝、端部削角、搭接长度和施工误差影响。",
        "- 如果程序统计与人工统计差异较大，应优先检查非结构杆件、重复杆件、超长杆件拆分、多段短杆合并、取整步长和 max_pieces_per_stock 设置。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_material_stock_json(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def stock_summary_report_lines(summary: dict[str, Any]) -> list[str]:
    if not summary:
        return [
            "",
            "## 标准木杆近似排料统计",
            "- 未启用或未生成 V2.1 近似排料统计。",
        ]
    return [
        "",
        "## 标准木杆近似排料统计",
        f"- 是否启用长度近似: {summary.get('rounding_enabled', True)}",
        f"- 取整步长: {summary['round_step_mm']} mm",
        f"- 配对容差: {summary['pair_tolerance_mm']} mm",
        f"- 是否考虑锯缝: {summary['consider_saw_kerf']}",
        f"- 有效木杆件段数: {summary['effective_wood_segment_count']}",
        f"- 最终标准木杆数量: {summary['stock_wood_count']}",
        f"- 与人工统计 {summary['manual_stock_count']} 根的差异: {summary['stock_count_difference_vs_manual']}",
        f"- raw_material_score: {summary['raw_material_score']}",
        f"- capped_material_score: {summary['capped_material_score']}",
        f"- score_policy_note: {summary['score_policy_note']}",
        f"- 余料总量: {summary['total_waste_mm']:.1f} mm",
        f"- 平均余料: {summary['average_waste_mm']:.1f} mm",
        f"- 配对成功率: {summary['pair_success_rate']:.1%}",
        f"- 无法配对杆件列表: {summary['unpaired_member_ids'] or '无'}",
        f"- 超长杆件列表: {summary['oversized_member_ids'] or '无'}",
        f"- 推荐采用的材料统计结果: stock_wood_count = {summary['stock_wood_count']}",
        "",
        "### V2.1 统计限制",
        "- 近似排料统计用于材料成本估算，不用于 OpenSeesPy 结构计算。",
        "- OpenSeesPy 仍使用真实几何长度。",
        "- 长度近似和两两配对会受到锯缝、端部削角、搭接长度、施工误差影响。",
        "- 若程序统计与人工统计差异较大，应检查非结构杆件、重复杆件、超长杆件、多段短杆、取整步长和 max_pieces_per_stock。",
    ]
