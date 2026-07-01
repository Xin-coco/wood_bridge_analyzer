from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def top_member_ids(member_df: pd.DataFrame, n: int = 10) -> list[int]:
    if member_df.empty:
        return []
    return [int(x) for x in member_df.sort_values("risk_score", ascending=False).head(n)["member_id"].tolist()]


def write_manual_review_checklist(
    path: Path,
    diagnostics: dict[str, Any],
    issues: dict[str, Any],
    supports: list[dict[str, Any]],
    deck_node_ids: list[int],
    governing: Any,
    suggestions: list[dict[str, Any]],
) -> None:
    high_risk = top_member_ids(governing.member_results)
    support_nodes = [int(s["node_id"]) for s in supports]
    build_focus = [s for s in suggestions if int(s.get("priority", 99)) <= 4]
    lines = [
        "# Manual Review Checklist",
        "",
        "## 需要人工确认的节点",
        f"- 单杆或异常连接节点: {sorted(set(diagnostics.get('single_member_nodes', []) + diagnostics.get('abnormal_nodes', []))) or '无'}",
        f"- 距离很近但未聚类的杆端: {len(diagnostics.get('close_unclustered_endpoints', []))} 处，详见 `close_unclustered_endpoints.csv`。",
        "",
        "## 可能被误识别的杆件",
        f"- 可能辅助线/节点块: {issues.get('可能被误识别为木杆的辅助线或节点块') or '无'}",
        f"- 超短杆件: {issues.get('超短杆件') or '无'}",
        f"- 疑似重复杆件: {issues.get('疑似重复杆件') or '无'}",
        "",
        "## 支座节点确认",
        f"- 当前支座节点: {support_nodes}",
        "- 在 Rhino 中确认这些节点是否确实位于两端最低支承位置，并确认滚动端方向是否符合实际加载。",
        "",
        "## 桥面加载节点确认",
        f"- 当前桥面加载节点: {deck_node_ids}",
        "- 检查这些节点是否位于真实桥面受力范围，避免把上弦或非桥面节点纳入人群荷载。",
        "",
        "## Rhino 模型中需要检查的高风险杆件",
        f"- Top 风险杆件: {high_risk or '无'}",
        "- 检查其端点是否汇交、是否超长拼接、是否有侧向约束，尤其关注受压杆弱轴方向。",
        "",
        "## 实体建造重点加固位置",
    ]
    for item in build_focus:
        lines.append(f"- P{item['priority']} {item['topic']}: {item['target'] or '无明显目标'}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_reinforcement_priority(
    path: Path,
    suggestions: list[dict[str, Any]],
    rod_summary: dict[str, Any],
    validation_summary: dict[str, Any],
) -> None:
    high_risk = validation_summary.get("high_risk_member_count", 0)
    mechanism = validation_summary.get("suspected_mechanism", False)
    groups = [
        ("第一优先级：必须修改，否则承重测试风险高", [s for s in suggestions if s["priority"] in {1, 2, 3}]),
        ("第二优先级：建议修改，可明显提高稳定性", [s for s in suggestions if s["priority"] in {4, 5}]),
        ("第三优先级：表达和施工优化", [s for s in suggestions if s["priority"] >= 6]),
    ]
    lines = ["# Reinforcement Priority", ""]
    for title, items in groups:
        lines.append(f"## {title}")
        if not items:
            lines.append("- 暂无。")
        for item in items:
            rod_effect = "可能增加 1-4 根标准杆" if item["priority"] in {1, 2, 4} else ("不一定增加木杆，主要增加节点板/夹具" if item["priority"] == 5 else "超长杆拆分会增加拼接件，等效杆数量不变或略增")
            stability_effect = "显著提高整体稳定性" if item["priority"] <= 2 or mechanism or high_risk else "提高局部可靠性和施工容错"
            lines.extend(
                [
                    f"- 问题位置: {item['target'] or item['topic']}",
                    f"  问题原因: {item['suggestion']}",
                    f"  建议做法: 围绕 `{item['topic']}` 优先补杆、加横向联系或加强节点连接。",
                    f"  对木杆数量的影响: {rod_effect}。",
                    f"  对结构稳定性的预期影响: {stability_effect}。",
                ]
            )
        lines.append("")
    lines.append(f"当前等效标准杆数量为 {rod_summary['equivalent_standard_count']}，材料成本分为 {rod_summary['material_score']}。")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_board_text(
    path: Path,
    geom: dict[str, Any],
    rod_summary: dict[str, Any],
    governing: Any,
    conservative: dict[str, float],
    validation_summary: dict[str, Any],
) -> None:
    stock_count = rod_summary.get("stock_wood_count", rod_summary["equivalent_standard_count"])
    material_score = rod_summary.get("raw_material_score", rod_summary["material_score"])
    lines = [
        "# Board Text",
        "",
        "## 150字结构分析说明",
        f"本桥采用单跨木桁架逻辑，实桥估计跨度 {geom['span_mm']:.0f}mm、宽度 {geom['width_mm']:.0f}mm。模型按三维铰接桁架计算，杆件仅承受轴力。控制工况为 {governing.name}，理想最大竖向位移 {governing.max_vertical_displacement_mm:.2f}mm，保守修正后约 {conservative['conservative_max_displacement_mm']:.2f}mm。计算结果显示结构主要风险集中在压杆侧向稳定、桥面平面拉结和中跨下弦受力。",
        "",
        "## 100字材料统计说明",
        f"程序识别模型杆件 {rod_summary['model_rod_count']} 根，经 1300mm 标准杆排料后需领取 {stock_count} 根。任务书基准为 60 根，当前原始材料成本分估算为 {material_score}。超长杆件为 {rod_summary['overlength_ids'] or '无'}，后续应优先检查拼接位置和节点板可靠性。",
        "",
        "## 100字受力路径说明",
        f"竖向荷载经桥面节点传入上下弦和斜撑，支座反力与总荷载差异约 {validation_summary['vertical_error_ratio']:.1%}。最大压杆区域判断为 {validation_summary['max_compression_zone']}，最大拉杆区域判断为 {validation_summary['max_tension_zone']}。这说明模型受力路径基本可读，但若存在奇异刚度或高条件数，仍需核对节点汇交与支座设置。",
        "",
        "## 100字加固策略说明",
        "加固优先从中跨下弦、上弦压杆侧向约束和桥面 X 形拉结开始；端部支座区域应补底部拉结，避免宽度方向张开。高风险节点建议采用金属节点板或双侧夹板，提高连接抗滑移能力。所有建议应先在 Rhino 中复核节点编号，再落实到实体建造。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def comparison_rows(old: dict[str, Any], new: dict[str, Any]) -> list[dict[str, Any]]:
    def metric(name: str, old_value: Any, new_value: Any) -> dict[str, Any]:
        try:
            delta = float(new_value) - float(old_value)
        except Exception:
            delta = ""
        return {"metric": name, "old": old_value, "new": new_value, "delta": delta}

    return [
        metric("模型杆件数量", old["rod_summary"]["model_rod_count"], new["rod_summary"]["model_rod_count"]),
        metric("等效标准杆数量", old["rod_summary"]["equivalent_standard_count"], new["rod_summary"]["equivalent_standard_count"]),
        metric("最大位移_mm", old["governing"].max_vertical_displacement_mm, new["governing"].max_vertical_displacement_mm),
        metric("最大受压杆", old["validation_summary"].get("max_compression_member"), new["validation_summary"].get("max_compression_member")),
        metric("最大受拉杆", old["validation_summary"].get("max_tension_member"), new["validation_summary"].get("max_tension_member")),
        metric("屈曲风险杆件数量", old["validation_summary"].get("high_buckling_member_count"), new["validation_summary"].get("high_buckling_member_count")),
        metric("高风险节点数量", len(old["diagnostics"].get("abnormal_nodes", [])), len(new["diagnostics"].get("abnormal_nodes", []))),
        metric("材料成本分", old["rod_summary"]["material_score"], new["rod_summary"]["material_score"]),
    ]


def write_comparison_report(path: Path, rows: list[dict[str, Any]], old_name: str, new_name: str) -> None:
    lines = ["# Comparison Report", "", f"- 修改前模型: `{old_name}`", f"- 修改后模型: `{new_name}`", "", "## 指标变化"]
    improved = []
    for row in rows:
        lines.append(f"- {row['metric']}: {row['old']} -> {row['new']}；变化 {row['delta']}")
        if row["metric"] in {"最大位移_mm", "屈曲风险杆件数量", "高风险节点数量"}:
            try:
                if float(row["delta"]) < 0:
                    improved.append(row["metric"])
            except Exception:
                pass
    lines.extend(["", "## 判断", f"- 改善项: {improved or '无明显改善'}"])
    lines.append("- 若最大位移、屈曲风险杆件数量和问题节点数量下降，同时材料分没有明显恶化，可认为修改方向更稳健。")
    path.write_text("\n".join(lines), encoding="utf-8")
