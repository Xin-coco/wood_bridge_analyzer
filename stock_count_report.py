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
        "## 统计口径",
        f"- model_member_count（模型中识别到的杆件段数）: {summary['model_member_count']} 段",
        f"- structural_member_count（实际参与结构分析/材料统计的有效木杆段数）: {summary['structural_member_count']} 段",
        f"- length_rounded_member_count（长度近似后参与排料杆件）: {summary['included_member_count']} 段",
        f"- program_stock_wood_count（程序两两配对排料结果）: {summary['program_stock_wood_count']} 根",
        f"- manual_stock_count（人工复核标准木杆数量）: {summary['manual_stock_count']} 根",
        f"- stock_wood_count（最终用于材料成本分的标准木杆数量）: {summary['stock_wood_count']} 根",
        f"- stock_count_source: {summary['stock_count_source']}",
        "",
        "模型杆件段数不等于标准木杆使用数量。程序识别的 66 段表示桥体中的构件段数，人工统计的约 46 根表示经过裁切排料后需要领取的 1300mm 标准木杆数量。材料成本分应以后者为准。",
        "",
        "## 排料结果",
        f"- 取近似后的长度种类: {summary['rounded_length_type_count']}",
        f"- 配对成功的标准木杆数量: {summary['paired_stock_count']}",
        f"- 单独占用标准木杆数量: {summary['single_stock_count']}",
        f"- 超长杆件数量: {summary['oversized_member_count']}",
        f"- 程序排料与人工统计差异: {summary['stock_count_difference_vs_manual']}",
        f"- 模型段数与最终标准木杆数差异: {summary['model_vs_stock_difference']}",
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
        "模型杆件段数不等于标准木杆使用数量。程序识别的 66 段表示桥体中的构件段数，人工统计的约 46 根表示经过裁切排料后需要领取的 1300mm 标准木杆数量。材料成本分应以后者为准。",
        "",
        f"- model_member_count（模型识别杆件段数）: {summary['model_member_count']} 段",
        f"- structural_member_count（有效结构木杆段数）: {summary['structural_member_count']} 段",
        f"- length_rounded_member_count（长度近似后参与排料杆件）: {summary['included_member_count']} 段",
        f"- program_stock_wood_count（程序排料后标准木杆数量）: {summary['program_stock_wood_count']} 根",
        f"- manual_stock_count（人工复核标准木杆数量）: {summary['manual_stock_count']} 根",
        f"- stock_wood_count（最终用于材料成本分）: {summary['stock_wood_count']} 根",
        f"- stock_count_source: {summary['stock_count_source']}",
        f"- 是否启用长度近似: {summary.get('rounding_enabled', True)}",
        f"- 取整步长: {summary['round_step_mm']} mm",
        f"- 配对容差: {summary['pair_tolerance_mm']} mm",
        f"- 是否考虑锯缝: {summary['consider_saw_kerf']}",
        f"- 程序排料与人工统计 {summary['manual_stock_count']} 根的差异: {summary['stock_count_difference_vs_manual']}",
        f"- 模型段数与最终标准木杆数差异: {summary['model_vs_stock_difference']}",
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
        "### 差异原因",
        "- 一根 1300mm 标准木杆可以裁切成两段或多段。",
        "- 程序识别的是桥体构件段数，人工统计的是原材领取数量。",
        "- 两根短杆相加接近 1300mm 时，应视为只消耗一根标准木杆。",
        "- 因此模型杆件段数通常会大于实际标准木杆数量。",
        "",
        "### V2.1 统计限制",
        "- 近似排料统计用于材料成本估算，不用于 OpenSeesPy 结构计算。",
        "- OpenSeesPy 仍使用真实几何长度。",
        "- 长度近似和两两配对会受到锯缝、端部削角、搭接长度、施工误差影响。",
        "- 若程序统计与人工统计差异较大，应检查非结构杆件、重复杆件、超长杆件、多段短杆、取整步长和 max_pieces_per_stock。",
    ]


def write_material_count_comparison_v21(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# 材料统计口径对比",
        "",
        "## 三种数量不能混用",
        f"- 模型识别杆件段数 model_member_count: {summary['model_member_count']} 段",
        f"- 有效结构木杆段数 structural_member_count: {summary['structural_member_count']} 段",
        f"- 程序排料后标准木杆数量 program_stock_wood_count: {summary['program_stock_wood_count']} 根",
        f"- 人工统计标准木杆数量 manual_stock_count: {summary['manual_stock_count']} 根",
        f"- 最终建议采用 stock_wood_count: {summary['stock_wood_count']} 根",
        f"- 当前采用来源 stock_count_source: {summary['stock_count_source']}",
        "",
        "## 差异数量",
        f"- model_member_count - stock_wood_count = {summary['model_vs_stock_difference']} 段/根",
        f"- program_stock_wood_count - manual_stock_count = {summary['stock_count_difference_vs_manual']} 根",
        "",
        "## 差异原因",
        "- 66 段表示桥体中的木杆构件段数，不表示领取了 66 根 1300mm 标准木杆。",
        "- 人工统计的约 46 根表示经过裁切排料后实际需要领取的标准木杆数量。",
        "- 一根 1300mm 标准木杆可以裁切成两段或多段，因此模型构件段数通常大于原材领取数量。",
        "- 两根短杆相加接近 1300mm 时，应视为只消耗一根标准木杆。",
        "- 若程序排料结果和人工统计仍有差异，应复核非结构杆件、重复杆件、超长杆件拆分、施工搭接和是否允许三段以上排料。",
        "",
        "## 材料成本分",
        "- 材料成本分必须按 stock_wood_count 计算，而不是按 model_member_count 计算。",
        f"- raw_material_score: {summary['raw_material_score']}",
        f"- capped_material_score: {summary['capped_material_score']}",
        f"- score_policy_note: {summary['score_policy_note']}",
        "",
        "## 最终建议采用的材料统计口径",
        "- 对外汇报时写“模型识别杆件段数”和“标准木杆领取数量”两个数字。",
        "- 成本分采用 stock_wood_count；结构检查仍采用 structural_member_count 和真实几何长度。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
