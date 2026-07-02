from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import pandas as pd


def write_structural_diagnosis_json(path: Path, diagnosis: dict[str, Any]) -> None:
    path.write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")


def _fmt_list(values: Any) -> str:
    if not values:
        return "无"
    if isinstance(values, list):
        return ", ".join(str(v) for v in values[:20])
    return str(values)


def write_structural_diagnosis_markdown(path: Path, diagnosis: dict[str, Any]) -> None:
    lines = [
        "# 模型问题诊断与修改建议",
        "",
        "## 当前模型总体判断",
        f"- {diagnosis['overall_judgement']}",
        f"- 是否建议进入实体建造: {'是' if diagnosis['can_enter_construction'] else '否'}",
        f"- 是否建议继续修改模型: {'是' if diagnosis['should_continue_model_revision'] else '否'}",
        f"- 问题总数: {diagnosis['issue_count']}",
        f"- 严重程度统计: {diagnosis['severity_counts']}",
    ]
    if diagnosis.get("missing_or_unreadable_inputs"):
        lines.extend(["", "## 无法完成的诊断项"])
        for item in diagnosis["missing_or_unreadable_inputs"]:
            lines.append(f"- {item}")
    lines.extend(["", "## 最严重的 5 个问题"])
    for issue in diagnosis["issues"][:5]:
        lines.extend(
            [
                f"### P{issue['priority_rank']} {issue['severity']} / {issue['issue_type']}",
                f"- affected_nodes: {_fmt_list(issue['affected_nodes'])}",
                f"- affected_members: {_fmt_list(issue['affected_members'])}",
                f"- evidence: {issue['evidence']}",
                f"- reason: {issue['reason']}",
                f"- recommendation: {issue['recommendation']}",
                f"- expected_effect: {issue['expected_effect']}",
                f"- material_impact: {issue['material_impact']}",
                f"- construction_difficulty: {issue['construction_difficulty']}",
                "",
            ]
        )
    lines.extend(["", "## 所有问题列表"])
    for issue in diagnosis["issues"]:
        lines.append(
            f"- P{issue['priority_rank']} [{issue['severity']}] {issue['issue_type']}: {issue['reason']} 建议: {issue['recommendation']}"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "- 如果出现 critical 或多个 high 问题，当前模型不应直接作为承重测试依据。",
            "- 修改建议优先使用任务书允许的标准木杆、金属节点板、双侧夹板和绳索拉结，不建议使用胶粘剂。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_fix_suggestions_outputs(output_dir: Path, diagnosis: dict[str, Any]) -> None:
    issues = diagnosis["issues"]
    rows = []
    for issue in issues:
        rows.append(
            {
                "priority": issue["priority_rank"],
                "issue_type": issue["issue_type"],
                "severity": issue["severity"],
                "affected_members": ";".join(str(x) for x in issue["affected_members"]),
                "affected_nodes": ";".join(str(x) for x in issue["affected_nodes"]),
                "recommendation": issue["recommendation"],
                "material_impact": issue["material_impact"],
                "expected_effect": issue["expected_effect"],
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "fix_suggestions.csv", index=False)
    groups = [
        ("第一优先级：必须改", {"critical"}),
        ("第二优先级：强烈建议改", {"high"}),
        ("第三优先级：优化表达和施工", {"medium", "low"}),
    ]
    lines = ["# 修改建议优先级", ""]
    for title, severities in groups:
        lines.extend([f"## {title}", ""])
        selected = [issue for issue in issues if issue["severity"] in severities]
        if not selected:
            lines.append("- 无")
        for issue in selected:
            lines.extend(
                [
                    f"### P{issue['priority_rank']} {issue['issue_type']}",
                    f"- 问题位置: 节点 {_fmt_list(issue['affected_nodes'])}；杆件 {_fmt_list(issue['affected_members'])}",
                    f"- 问题原因: {issue['reason']}",
                    f"- 建议做法: {issue['recommendation']}",
                    f"- 对木杆数量的影响: {issue['material_impact']}",
                    f"- 对结构稳定性的预期影响: {issue['expected_effect']}",
                    f"- 施工难度: {issue['construction_difficulty']}",
                    "",
                ]
            )
    (output_dir / "fix_suggestions.md").write_text("\n".join(lines), encoding="utf-8")


def write_model_test_summary(path: Path, diagnosis: dict[str, Any]) -> None:
    issue_types = {issue["issue_type"] for issue in diagnosis["issues"]}
    lines = [
        "# Model Test Summary",
        "",
        f"- 是否满足 0.5m 位移控制: {'需复核' if 'fem_not_reliable' in issue_types else ('否' if 'excessive_displacement' in issue_types else '未发现超限')}",
        f"- 是否存在高风险压杆: {'是' if 'high_buckling_risk' in issue_types else '未检出或数据不足'}",
        f"- 是否存在节点问题: {'是' if {'single_member_nodes', 'dangling_members', 'disconnected_components'} & issue_types else '未检出'}",
        f"- 是否存在支座问题: {'是' if {'support_definition_missing', 'support_instability', 'reaction_unbalanced'} & issue_types else '未检出或数据不足'}",
        f"- 是否存在材料统计问题: {'是' if {'oversized_members', 'material_count_difference', 'large_waste'} & issue_types else '未检出'}",
        f"- 当前模型是否建议继续建造: {'是' if diagnosis['can_enter_construction'] else '否'}",
        f"- 总体判断: {diagnosis['overall_judgement']}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_board_structure_advice(path: Path, diagnosis: dict[str, Any]) -> None:
    top = diagnosis["issues"][:3]
    top_text = "、".join(issue["issue_type"] for issue in top) if top else "未发现主要问题"
    lines = [
        "# Board Structure Advice",
        "",
        "## 100字结构问题总结",
        f"当前模型主要问题集中在 {top_text}。若存在 critical 问题，应先复核节点汇交、支座节点和桥面加载节点，再进入承重分析或实体测试。",
        "",
        "## 100字受力路径分析",
        "竖向荷载应由桥面节点传入横梁、上下弦和斜撑，再传至两端支座。若中心线不连通、单杆节点过多或支座未确认，受力路径会中断，位移和杆力排序不具备可靠性。",
        "",
        "## 100字修改策略",
        "优先修复 critical 问题，再处理抗扭和压杆稳定。建议用桥面 X 形绳索、顶部横向联系、端部闭合三角形和金属节点板提高整体性，避免使用胶粘剂。",
        "",
        "## 100字材料优化说明",
        "材料分应按 1300mm 标准木杆领取数量计算，不按模型构件段数计算。短杆应成对排料，超长杆需拆分并设置可靠拼接节点，非结构杆件和重复杆件不应计入材料数量。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def diagnosis_report_lines(diagnosis: dict[str, Any]) -> list[str]:
    if not diagnosis:
        return ["", "## 当前模型问题诊断与修改建议", "- 未生成 V2.2 结构诊断。"]
    lines = [
        "",
        "## 当前模型问题诊断与修改建议",
        f"- 当前模型总体评价: {diagnosis['overall_judgement']}",
        f"- 是否建议进入实体建造: {'是' if diagnosis['can_enter_construction'] else '否'}",
        f"- 问题数量: {diagnosis['issue_count']}；严重程度: {diagnosis['severity_counts']}",
        "- 是否满足任务书主要指标: 若 FEM 被阻断，则位移和杆力指标需复核；材料数量按 stock_wood_count 统计。",
        "",
        "### 主要结构问题",
    ]
    for issue in diagnosis["issues"][: min(5, len(diagnosis["issues"]))]:
        lines.append(f"- P{issue['priority_rank']} [{issue['severity']}] {issue['issue_type']}: {issue['reason']}")
    lines.extend(["", "### 建议优先修改顺序"])
    for issue in diagnosis["issues"][: min(int(diagnosis.get("diagnosis_top_n", 10)), len(diagnosis["issues"]))]:
        lines.append(
            f"- P{issue['priority_rank']} {issue['recommendation']} 影响: {issue['expected_effect']} 材料: {issue['material_impact']}"
        )
    lines.extend(
        [
            "",
            "### 下一步建模建议",
            "- 先修复 critical 问题，再讨论杆件加密或材料优化。",
            "- 修改后重新运行 V2.2，并比较最大位移、危险杆件数量、节点问题数量和 stock_wood_count。",
        ]
    )
    return lines
