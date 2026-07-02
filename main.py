from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Any

_cache_root = Path(tempfile.gettempdir()) / "wood_bridge_analyzer_cache"
_cache_root.mkdir(parents=True, exist_ok=True)
(_cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
(_cache_root / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg"))

try:
    import numpy as np
    import pandas as pd
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("缺少运行依赖。请在 wood_bridge_analyzer 目录运行: pip install -r requirements.txt") from exc

from bridge_parser import ModelMetadata, parse_3dm_model
from centerline_validation import (
    RELIABLE_FEM_BLOCK_MESSAGE,
    centerline_basic_checks,
    check_deck_nodes,
    check_support_nodes,
    fem_precheck,
    score_centerline_model,
    write_centerline_validation_report,
    write_check_markdown,
)
from clean_centerline_model import (
    build_clean_centerline_model,
    confidence_scores,
    manual_fem_inputs_confirmed,
    write_clean_centerline_outputs,
    write_node_quality_report,
)
from cut_stock_optimizer import optimize_cut_stock, write_cut_plan_markdown, write_material_count_comparison
from fem_truss_solver import (
    LoadCaseResult,
    build_truss_model,
    deck_nodes,
    geometry_checks,
    members_dataframe,
    node_diagnostics,
    nodes_dataframe,
    resolve_supports,
    run_standard_cases,
)
from fix_suggestions import generate_fix_suggestions, write_fix_suggestions
from fix_suggestion_report import (
    diagnosis_report_lines,
    write_board_structure_advice,
    write_fix_suggestions_outputs,
    write_model_test_summary,
    write_structural_diagnosis_json,
    write_structural_diagnosis_markdown,
)
from fix_suggestion_visualization import create_fix_suggestion_visualizations
from opensees_backend import run_opensees_backend
from opensees_exporter import deck_nodes_from_config, opensees_requested, support_nodes_from_config
from opensees_postprocess import create_opensees_visualizations
from material_length_rounding import load_material_members, material_stock_config, rounded_member_lengths
from rod_counter import inventory_dataframe, summarize_rods, write_inventory_summary
from sanity_check import run_sanity_checks
from solver_comparison import compare_solvers
from stock_count_report import (
    stock_summary_report_lines,
    write_material_count_comparison_v21,
    write_material_stock_json,
    write_material_stock_summary,
    write_paired_cut_plan_markdown,
)
from stock_count_visualization import create_stock_count_visualizations
from stock_pairing_optimizer import optimize_pairing
from structural_diagnosis import run_structural_diagnosis
from validation_check import REVIEW_WARNING, run_validation_checks
from v17_reports import (
    comparison_rows,
    write_board_text,
    write_comparison_report,
    write_manual_review_checklist,
    write_reinforcement_priority,
)
from visualization import create_visualizations, plot_comparison_summary


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(config: dict[str, Any], args: Any) -> dict[str, Any]:
    stock_cfg = config.setdefault("material_stock_counting", {})
    rounding = stock_cfg.setdefault("length_rounding", {})
    pairing = stock_cfg.setdefault("pairing", {})
    manual = stock_cfg.setdefault("manual_compare", {})
    if getattr(args, "enable_stock_pairing", False):
        stock_cfg["enabled"] = True
        pairing["enabled"] = True
        rounding["enabled"] = True
    if getattr(args, "round_step", None) is not None:
        rounding["step_mm"] = float(args.round_step)
    if getattr(args, "pair_tolerance", None) is not None:
        pairing["pair_tolerance_mm"] = float(args.pair_tolerance)
    if getattr(args, "manual_stock_count", None) is not None:
        manual["manual_stock_count"] = int(args.manual_stock_count)
    diagnosis_cfg = config.setdefault("structural_diagnosis", {})
    if getattr(args, "generate_diagnosis", False):
        diagnosis_cfg["enabled"] = True
    if getattr(args, "generate_fix_suggestions", False):
        diagnosis_cfg["enabled"] = True
        diagnosis_cfg["generate_fix_suggestions"] = True
    if getattr(args, "diagnosis_top_n", None) is not None:
        diagnosis_cfg["top_n"] = int(args.diagnosis_top_n)
    return config


def log_step(message: str) -> None:
    print(f"[wood-bridge-analyzer] {message}")


def choose_governing(results: dict[str, Any]):
    successful = [r for r in results.values() if r.success]
    if successful:
        return max(successful, key=lambda r: (r.max_vertical_displacement_mm, r.max_abs_force_n))
    return max(results.values(), key=lambda r: (r.max_vertical_displacement_mm, r.max_abs_force_n))


def load_case_summary_dataframe(results: dict[str, Any], moving_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, result in results.items():
        rows.append(
            {
                "case": name,
                "success": result.success,
                "message": result.message,
                "singular": result.singular,
                "rank": result.rank,
                "free_dof_count": result.free_dof_count,
                "condition_number": result.condition_number,
                "max_vertical_displacement_mm": result.max_vertical_displacement_mm,
                "max_abs_force_n": result.max_abs_force_n,
                "max_reaction_n": result.max_reaction_n,
            }
        )
    if not moving_df.empty:
        worst = moving_df.sort_values(["max_vertical_displacement_mm", "max_abs_force_n"], ascending=False).head(1)
        rows.append(
            {
                "case": "moving_load_envelope",
                "success": bool(worst.iloc[0]["success"]),
                "message": str(worst.iloc[0]["message"]),
                "singular": not bool(worst.iloc[0]["success"]),
                "rank": "",
                "free_dof_count": "",
                "condition_number": "",
                "max_vertical_displacement_mm": float(worst.iloc[0]["max_vertical_displacement_mm"]),
                "max_abs_force_n": float(worst.iloc[0]["max_abs_force_n"]),
                "max_reaction_n": "",
            }
        )
    return pd.DataFrame(rows)


def manual_confirmation_result(node_count: int, message: str) -> LoadCaseResult:
    displacements = np.zeros(node_count * 3, dtype=float)
    reactions = np.zeros(node_count * 3, dtype=float)
    return LoadCaseResult(
        "manual_confirmation_required",
        displacements,
        reactions,
        pd.DataFrame(),
        0.0,
        0.0,
        0.0,
        True,
        0,
        node_count * 3,
        float("inf"),
        False,
        message,
    )


def apply_member_overrides(rods: list[Any], config: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    overrides = config.get("manual_overrides", {}) or {}
    ignored = set(int(x) for x in (overrides.get("ignored_members") or []))
    force_include = set(int(x) for x in (overrides.get("force_include_members") or []))
    effective_ignored = ignored - force_include
    filtered = [rod for rod in rods if rod.id not in effective_ignored]
    return filtered, {
        "ignored_members": sorted(ignored),
        "force_include_members": sorted(force_include),
        "effective_ignored_members": sorted(effective_ignored),
        "removed_count": len(rods) - len(filtered),
        "manual_support_nodes": overrides.get("support_nodes", {}) or {},
        "manual_deck_nodes": overrides.get("deck_nodes") or [],
        "manual_used": bool(ignored or force_include or (overrides.get("deck_nodes") or []) or any((overrides.get("support_nodes", {}) or {}).values())),
    }


def conservative_results(governing: Any, config: dict[str, Any]) -> dict[str, float]:
    factors = config.get("conservative_factors", {}) or {}
    slip = float(factors.get("connection_slip_factor", 1.0))
    construction = float(factors.get("construction_error_factor", 1.0))
    material = float(factors.get("material_uncertainty_factor", 1.0))
    buckling = float(factors.get("buckling_safety_factor", 1.0))
    displacement_factor = slip * construction
    force_factor = construction * material
    buckling_factor = force_factor * buckling
    max_buckling = 0.0
    max_risk = 0.0
    if not governing.member_results.empty:
        max_buckling = float(governing.member_results["buckling_utilization"].max())
        max_risk = float(governing.member_results["risk_score"].max())
    return {
        "displacement_factor": displacement_factor,
        "force_factor": force_factor,
        "buckling_factor": buckling_factor,
        "ideal_max_displacement_mm": governing.max_vertical_displacement_mm,
        "conservative_max_displacement_mm": governing.max_vertical_displacement_mm * displacement_factor,
        "ideal_max_abs_force_n": governing.max_abs_force_n,
        "conservative_max_abs_force_n": governing.max_abs_force_n * force_factor,
        "ideal_max_buckling_utilization": max_buckling,
        "conservative_max_buckling_utilization": max_buckling * buckling_factor,
        "ideal_max_risk_score": max_risk,
        "conservative_max_risk_score": max_risk * max(force_factor, buckling_factor),
    }


def recognition_issues(rod_summary: dict[str, Any], diagnostics: dict[str, Any], skipped: list[str], rods: list[Any], config: dict[str, Any]) -> dict[str, Any]:
    standard_len = float(config["section"]["standard_length_mm"])
    very_short = [r.id for r in rods if r.length_mm < 0.15 * standard_len]
    possible_aux = [
        r.id
        for r in rods
        if r.source_type.lower().endswith("curve") and (r.length_mm < 0.2 * standard_len or any(k in r.layer.lower() for k in ["axis", "guide", "helper", "辅助", "轴线"]))
    ]
    return {
        "超短杆件": very_short or rod_summary.get("short_ids", []),
        "超长杆件": rod_summary.get("overlength_ids", []),
        "疑似重复杆件": rod_summary.get("duplicate_pairs", []),
        "孤立杆件": diagnostics.get("dangling_members", []),
        "未连接杆端": diagnostics.get("close_unclustered_endpoints", []),
        "节点连接数量异常": diagnostics.get("abnormal_nodes", []),
        "可能不是木杆的对象": skipped[:30],
        "可能被误识别为木杆的辅助线或节点块": possible_aux,
    }


def confidence_rating(issues: dict[str, Any], sanity_summary: dict[str, Any], manual_info: dict[str, Any], governing: Any) -> str:
    score = 100
    if issues.get("可能不是木杆的对象"):
        score -= 10
    if issues.get("节点连接数量异常"):
        score -= 15
    if issues.get("未连接杆端"):
        score -= 10
    if sanity_summary.get("requires_manual_review"):
        score -= 20
    if governing.singular:
        score -= 20
    if manual_info.get("manual_used"):
        score += 5
    if score >= 80:
        return f"较高 ({score}/100)"
    if score >= 60:
        return f"中等 ({score}/100)"
    return f"较低 ({score}/100)"


def _row_value(df: pd.DataFrame, col: str, default: str = "无") -> str:
    if df.empty:
        return default
    return str(df.iloc[0][col])


def opensees_report_lines(opensees_result: Any, comparison_summary: dict[str, Any], visualization_failures: list[str], config: dict[str, Any]) -> list[str]:
    opensees_cfg = config.get("opensees", {}) or {}
    material = opensees_cfg.get("material", {}) or {}
    analysis_cfg = opensees_cfg.get("analysis", {}) or {}
    lines = [
        "",
        "## V2.0 OpenSeesPy 分析",
        f"- solver_backend: {config.get('solver_backend', 'numpy')}",
        f"- 是否请求 OpenSeesPy: {bool(opensees_result and opensees_result.requested)}",
        f"- OpenSeesPy 是否可用: {bool(opensees_result and opensees_result.available)}",
        f"- OpenSeesPy 是否成功运行: {bool(opensees_result and opensees_result.success)}",
        f"- 状态说明: {opensees_result.message if opensees_result else '未请求 OpenSeesPy backend'}",
        f"- 单元类型: {opensees_cfg.get('element_type', 'truss')}",
        f"- 材料参数: E={material.get('E_MPa', config['materials']['elastic_modulus_mpa'])} MPa, A={material.get('area_mm2', float(config['section']['width_mm']) * float(config['section']['height_mm']))} mm2",
        f"- 变形限值: {analysis_cfg.get('displacement_limit_mm', config['bridge']['max_deflection_mm'])} mm",
        f"- OpenSees 图像生成问题: {visualization_failures or '无'}",
    ]
    if opensees_result and not opensees_result.case_summary.empty:
        limit = float(analysis_cfg.get("displacement_limit_mm", config["bridge"]["max_deflection_mm"]))
        lines.extend(["", "### OpenSees 荷载工况结果"])
        for row in opensees_result.case_summary.itertuples():
            ok = float(row.max_vertical_displacement_mm) <= limit
            lines.append(
                f"- {row.case}: success={row.success}, max_uz={row.max_vertical_displacement_mm:.3f} mm, "
                f"max tension member={row.max_tension_member}, max compression member={row.max_compression_member}, "
                f"Rz sum={row.total_vertical_reaction_n:.2f} N, load/reaction diff={row.vertical_balance_error_n:.2f} N, "
                f"deflection={'满足' if ok else '超过'}"
            )
    else:
        lines.append("- 各荷载工况最大位移/杆力: 无 OpenSees 结果。")
    lines.extend(
        [
            "",
            "### 与 numpy FEM 对比",
            f"- comparison_available: {comparison_summary.get('comparison_available', False)}",
            f"- max_error_percent: {comparison_summary.get('max_error_percent')}",
            f"- acceptable: {comparison_summary.get('acceptable', False)}",
            f"- 对比结论: {comparison_summary.get('message', '未生成对比。')}",
        ]
    )
    if comparison_summary.get("message") == "两个求解器结果差异较大，需复核节点、支座、荷载分配和单位设置。":
        lines.append("- 两个求解器结果差异较大，需复核节点、支座、荷载分配和单位设置。")
    lines.extend(
        [
            "",
            "### OpenSees 加固判断",
            "- 优先复核 OpenSees 最大受压杆、最大受拉杆和反力集中支座附近节点。",
            "- 若 OpenSees 与 numpy 差异超过 10%，先复核单位、节点编号、支座约束和荷载节点，再讨论结构加固。",
        ]
    )
    return lines


def run_v21_material_stock_counting(output_dir: Path, config: dict[str, Any], material_summary: dict[str, Any], rod_summary: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    cfg = material_stock_config(config)
    if not cfg["enabled"]:
        return {}, []
    members = load_material_members(output_dir)
    rounded = rounded_member_lengths(members, config)
    rounded, plan, oversized, stock_summary = optimize_pairing(rounded, config)
    rounded.to_csv(output_dir / "rounded_member_lengths.csv", index=False)
    plan.to_csv(output_dir / "paired_stock_cut_plan.csv", index=False)
    oversized.to_csv(output_dir / "oversized_members.csv", index=False)
    write_paired_cut_plan_markdown(output_dir / "paired_stock_cut_plan.md", plan)
    write_material_stock_summary(output_dir / "material_stock_summary.md", stock_summary)
    write_material_stock_json(output_dir / "material_stock_summary.json", stock_summary)
    write_material_count_comparison_v21(output_dir / "material_count_comparison.md", stock_summary)
    failures = create_stock_count_visualizations(rounded, plan, stock_summary, output_dir, config)

    material_summary["v16_stock_wood_count"] = material_summary.get("stock_wood_count")
    material_summary["v16_raw_material_score"] = material_summary.get("raw_material_score")
    material_summary["v16_capped_material_score"] = material_summary.get("capped_material_score")
    material_summary["stock_wood_count"] = stock_summary["stock_wood_count"]
    material_summary["raw_material_score"] = stock_summary["raw_material_score"]
    material_summary["capped_material_score"] = stock_summary["capped_material_score"]
    material_summary["score_policy_note"] = stock_summary["score_policy_note"]
    material_summary["user_manual_count"] = stock_summary["manual_stock_count"]
    material_summary["v21_material_stock_summary_json"] = str(output_dir / "material_stock_summary.json")
    rod_summary["stock_wood_count"] = stock_summary["stock_wood_count"]
    rod_summary["raw_material_score"] = stock_summary["raw_material_score"]
    rod_summary["capped_material_score"] = stock_summary["capped_material_score"]
    rod_summary["material_score"] = stock_summary["raw_material_score"]
    return stock_summary, failures


def run_v22_structural_diagnosis(output_dir: Path, config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    diagnosis_cfg = config.get("structural_diagnosis", {}) or {}
    if not bool(diagnosis_cfg.get("enabled", True)):
        return {}, []
    top_n = int(diagnosis_cfg.get("top_n", 10))
    diagnosis = run_structural_diagnosis(output_dir, config, top_n=top_n)
    write_structural_diagnosis_json(output_dir / "structural_diagnosis.json", diagnosis)
    write_structural_diagnosis_markdown(output_dir / "structural_diagnosis.md", diagnosis)
    write_fix_suggestions_outputs(output_dir, diagnosis)
    write_model_test_summary(output_dir / "model_test_summary.md", diagnosis)
    write_board_structure_advice(output_dir / "board_structure_advice.md", diagnosis)
    visualization_failures = create_fix_suggestion_visualizations(output_dir, diagnosis, config, top_n)
    if visualization_failures:
        diagnosis["visualization_failures"] = visualization_failures
        write_structural_diagnosis_json(output_dir / "structural_diagnosis.json", diagnosis)
    report_path = output_dir / "analysis_report.md"
    if report_path.exists():
        with open(report_path, "a", encoding="utf-8") as f:
            f.write("\n".join(diagnosis_report_lines(diagnosis)))
            if visualization_failures:
                f.write("\n- V2.2 图像生成问题: " + str(visualization_failures))
            f.write("\n")
    return diagnosis, visualization_failures


def write_report(
    output_dir: Path,
    model_path: Path,
    metadata: ModelMetadata,
    rod_summary: dict[str, Any],
    geom: dict[str, Any],
    diagnostics: dict[str, Any],
    skipped: list[str],
    supports: list[dict[str, Any]],
    deck_node_ids: list[int],
    load_case_df: pd.DataFrame,
    moving_df: pd.DataFrame,
    governing: Any,
    sanity_df: pd.DataFrame,
    sanity_summary: dict[str, Any],
    validation_df: pd.DataFrame,
    validation_summary: dict[str, Any],
    conservative: dict[str, float],
    issues: dict[str, Any],
    suggestions: list[dict[str, Any]],
    manual_info: dict[str, Any],
    visualization_failures: list[str],
    material_summary: dict[str, Any],
    overlength_material_issues: list[dict[str, Any]],
    clean_info: dict[str, Any],
    support_load_info: dict[str, Any],
    confidence_breakdown: dict[str, int],
    centerline_validation: dict[str, Any],
    centerline_score: dict[str, Any],
    fem_precheck_summary: dict[str, Any],
    opensees_result: Any,
    solver_comparison_summary: dict[str, Any],
    opensees_visualization_failures: list[str],
    v21_stock_summary: dict[str, Any],
    v21_visualization_failures: list[str],
    fem_ran: bool,
    config: dict[str, Any],
) -> None:
    member_df = governing.member_results
    max_tension = member_df[member_df["force_type"] == "tension"].sort_values("axial_force_n", ascending=False).head(1) if not member_df.empty else pd.DataFrame()
    max_compression = member_df[member_df["force_type"] == "compression"].sort_values("axial_force_n", ascending=True).head(1) if not member_df.empty else pd.DataFrame()
    max_risk = member_df.sort_values("risk_score", ascending=False).head(1) if not member_df.empty else pd.DataFrame()
    buckling_top10 = member_df[member_df["force_type"] == "compression"].sort_values("buckling_utilization", ascending=False).head(10) if not member_df.empty else pd.DataFrame()
    moving_worst = moving_df.sort_values(["max_vertical_displacement_mm", "max_abs_force_n"], ascending=False).head(1) if not moving_df.empty else pd.DataFrame()
    max_deflection_limit = float(config["bridge"]["max_deflection_mm"])
    top_risk = member_df.sort_values("risk_score", ascending=False).head(10) if not member_df.empty else pd.DataFrame()
    problem_nodes = sorted(set(diagnostics.get("abnormal_nodes", []) + diagnostics.get("single_member_nodes", [])))[:10]
    confidence = confidence_rating(issues, sanity_summary, manual_info, governing)

    lines = [
        "# Wood Bridge Analysis Report",
        "",
        "## 基本信息",
        f"- 模型文件: `{model_path}`",
        f"- Rhino 文件单位: {metadata.rhino_unit}",
        f"- 默认/采用比例: 1:{metadata.scale:g}",
        f"- 模型尺寸 X/Y/Z: {geom['span_model_mm']:.1f} / {geom['width_model_mm']:.1f} / {geom['height_model_mm']:.1f} mm",
        f"- 实桥尺寸 X/Y/Z: {geom['span_mm']:.1f} / {geom['width_mm']:.1f} / {geom['height_mm']:.1f} mm",
        f"- 是否满足 3.9m 单跨要求: {'满足' if geom['span_ok'] else '需复核'}",
        f"- 是否满足 0.65m-1.3m 桥面宽度要求: {'满足' if geom['width_ok'] else '需复核'}",
        f"- 单位/比例检查: {'检测到 1:10 标准木杆模型' if metadata.detected_standard_1_to_10 else '未检测到典型 1:10 标准杆长度分布'}",
        f"- 程序识别可信度评价: {confidence}",
        f"- 人工校核模式: {'已使用 manual_overrides' if manual_info['manual_used'] else '未使用，采用自动识别'}",
        f"- 自动识别和人工指定的差异: 忽略杆件 {manual_info['effective_ignored_members']}；人工桥面节点 {manual_info['manual_deck_nodes'] or '无'}；人工支座 {manual_info['manual_support_nodes'] or '无'}",
        f"- 图像生成问题: {visualization_failures or '无'}",
        "",
        "## V1.6.3 模型层级说明",
        "- 展示模型: Rhino 3dm 中的实体、曲线、节点块等原始对象，用于表达外观，不直接等同于结构计算杆件。",
        "- 施工材料统计模型: 识别出的有效木杆构件经过 manual_member_overrides.csv 修正后，进入 1300mm 标准木杆排料统计。",
        "- 结构中心线计算模型: clean_centerline_model 将每根有效木杆简化为一条中心线，并输出 clean_nodes.csv 与 clean_members.csv。",
        f"- FEM 求解模型: {'已进入简化 3D 桁架 FEM' if fem_ran else '未进入 FEM；请先人工确认支座和桥面加载节点'}。",
        f"- clean 节点数/杆件数: {clean_info['clean_node_count']} / {clean_info['clean_member_count']}",
        f"- manual_node_overrides.csv: {clean_info['manual_node_overrides_csv_used']}",
        f"- 支座与加载节点确认: {support_load_info['message']}",
        "",
        "## V1.6.3 中心线模型验收",
        f"- centerline_model_score: {centerline_score['centerline_model_score']}/100",
        f"- 当前模型是否可以进入可靠 FEM 分析: {'可以' if centerline_score['can_enter_reliable_fem'] else '不可以'}",
        f"- connected components: {centerline_validation['connected_component_count']}",
        f"- 主结构是否单一连通体: {'是' if centerline_validation['is_single_connected_component'] else '否'}",
        f"- 单杆节点数量: {centerline_validation['single_member_node_count']}",
        f"- 悬空杆件数量: {centerline_validation['dangling_member_count']}",
        f"- 未连接杆端数量: {centerline_validation['unconnected_endpoint_count']}",
        f"- 重复杆件数量: {centerline_validation['duplicate_member_count']}",
        f"- 零长度杆件数量: {centerline_validation['zero_length_member_count']}",
        f"- 超短杆件数量: {centerline_validation['short_member_count']}",
        f"- 超长杆件数量: {centerline_validation['long_member_count']}",
        f"- FEM 前置验收: {fem_precheck_summary['message']}",
        f"- FEM 阻断原因: {fem_precheck_summary['blocking_reasons'] or '无'}",
        f"- 下一步: {'可进入 V1.7 或 OpenSeesPy 后端' if centerline_score['can_enter_reliable_fem'] else '先修复节点连接、支座定义和桥面加载节点'}",
        "",
        "## V1.6.2/V1.6.3 可信度评分",
        f"- geometry_recognition_score: {confidence_breakdown['geometry_recognition_score']}/100",
        f"- material_count_score: {confidence_breakdown['material_count_score']}/100",
        f"- node_network_score: {confidence_breakdown['node_network_score']}/100",
        f"- support_definition_score: {confidence_breakdown['support_definition_score']}/100",
        f"- load_definition_score: {confidence_breakdown['load_definition_score']}/100",
        f"- fem_result_score: {confidence_breakdown['fem_result_score']}/100",
        "",
        "## 木杆统计",
        f"- 模型识别杆件段数 model_member_count: {rod_summary['model_rod_count']} 段",
        f"- 有效结构木杆段数 structural_member_count: {material_summary['structural_member_count']} 段",
        f"- 长度近似后参与排料杆件: {v21_stock_summary.get('included_member_count', material_summary['structural_member_count'])} 段",
        f"- 程序排料后标准木杆数量 program_stock_wood_count: {v21_stock_summary.get('program_stock_wood_count', material_summary['stock_wood_count'])} 根",
        f"- 人工复核标准木杆数量 manual_stock_count: {material_summary['user_manual_count']} 根",
        f"- 最终标准木杆使用数量 stock_wood_count: {material_summary['stock_wood_count']} 根",
        f"- 程序原始等效标准杆: {rod_summary['equivalent_standard_count']}",
        f"- raw_material_score: {material_summary['raw_material_score']}",
        f"- capped_material_score: {material_summary['capped_material_score']}",
        "- 材料成本分按标准木杆使用数量 stock_wood_count 计算，不按模型杆件段数计算。",
        f"- 60 根限制: {'超过' if material_summary['stock_wood_count'] > rod_summary['base_count'] else '未超过'}",
        f"- 超长杆件 ID: {rod_summary['overlength_ids'] or '无'}",
        f"- 短杆件 ID: {rod_summary['short_ids'] or '无'}",
        f"- 疑似重复杆件: {rod_summary['duplicate_pairs'] or '无'}",
        f"- 人工修正表: {material_summary['manual_overrides_csv_used']}",
        "",
        "## V1.6.1 材料统计修正说明",
        "- 模型杆件段数不等于标准木杆使用数量。程序识别的 66 段表示桥体中的构件段数，人工统计的约 46 根表示经过裁切排料后需要领取的 1300mm 标准木杆数量。材料成本分应以后者为准。",
        "- 任务书材料评分应使用 stock_wood_count，而不是 model_member_count。",
        "- 原始等效标准杆按每个模型构件单独折算，会把多段短杆各算成一根标准木杆，因此会高估用料。",
        "- cut_plan.csv / cut_plan.md 已按 1300mm 标准木杆、锯缝和预留量进行 first-fit decreasing 排料。",
        f"- 差异对照: 程序原始识别 {rod_summary['model_rod_count']} 段；程序原始等效标准杆 {rod_summary['equivalent_standard_count']} 根；程序排料后标准木杆 {v21_stock_summary.get('program_stock_wood_count', material_summary['stock_wood_count'])} 根；人工复核 {material_summary['user_manual_count']} 根；最终采用 stock_wood_count {material_summary['stock_wood_count']} 根。",
        "- 最终建议：材料成本分采用 stock_wood_count；如果课程材料分上限为 20，则展示 capped_material_score。",
    ]
    lines.extend(stock_summary_report_lines(v21_stock_summary))
    if v21_visualization_failures:
        lines.append(f"- V2.1 图像生成问题: {v21_visualization_failures}")
    lines.extend([
        "",
        "## 超长杆件处理",
        "",
        "## 节点聚类检查",
        f"- 节点数量: 见 `nodes.csv`",
        f"- 杆件网络: 见 `members.csv`",
        f"- 节点聚类容差: {float(config['model'].get('node_cluster_tolerance_model_mm', config['model'].get('endpoint_cluster_tolerance_model_mm', 5.0))):.1f} mm model",
        f"- 孤立节点: {diagnostics['isolated_nodes'] or '无'}",
        f"- 单杆节点: {diagnostics['single_member_nodes'][:30]}{' ...' if len(diagnostics['single_member_nodes']) > 30 else ''}",
        f"- 悬空杆件: {diagnostics['dangling_members'][:30]}{' ...' if len(diagnostics['dangling_members']) > 30 else ''}",
        f"- 距离很近但未聚类的杆端数量: {len(diagnostics['close_unclustered_endpoints'])}",
        f"- 节点连接数量异常位置: {diagnostics['abnormal_nodes'][:30]}{' ...' if len(diagnostics['abnormal_nodes']) > 30 else ''}",
        f"- 超长压杆候选: {geom['very_long_members'] or '无'}",
        "",
        "## 支座与桥面加载节点",
        f"- 实际采用支座节点: {supports}",
        f"- 实际采用桥面加载节点: {deck_node_ids}",
        f"- fixed_nodes_confirmed: {support_load_info['fixed_nodes_confirmed']}",
        f"- roller_nodes_confirmed: {support_load_info['roller_nodes_confirmed']}",
        f"- deck_nodes_confirmed: {support_load_info['deck_nodes_confirmed']}",
        f"- V1.6.3 FEM 前置验收: {'已通过' if fem_ran else '未通过；本次不进行可靠 FEM 计算'}",
        "",
        "## 有限元分析结果",
        f"- 控制工况: {governing.name}",
        f"- 求解状态: {'成功' if governing.success else '不可解或需复核'}；{governing.message}",
        f"- 最大位移: {governing.max_vertical_displacement_mm:.3f} mm，限值 {max_deflection_limit:.1f} mm，结果: {'超过' if governing.max_vertical_displacement_mm > max_deflection_limit else '未超过'}",
        f"- 最大受拉杆: member {_row_value(max_tension, 'member_id')}，轴力 {float(max_tension.iloc[0]['axial_force_n']) if not max_tension.empty else 0:.2f} N",
        f"- 最大受压杆: member {_row_value(max_compression, 'member_id')}，轴力 {float(max_compression.iloc[0]['axial_force_n']) if not max_compression.empty else 0:.2f} N",
        f"- 最大节点反力: {governing.max_reaction_n:.2f} N",
        f"- 屈曲风险最高杆件: member {_row_value(max_risk, 'member_id')}，risk {float(max_risk.iloc[0]['risk_score']) if not max_risk.empty else 0:.3f}",
        f"- 最危险移动荷载位置: span ratio {float(moving_worst.iloc[0]['position_ratio']) if not moving_worst.empty else 0:.2f}",
        f"- 最危险荷载工况: {governing.name}",
    ])
    lines.extend(opensees_report_lines(opensees_result, solver_comparison_summary, opensees_visualization_failures, config))
    lines.extend([
        "",
        "## 保守修正结果",
        f"- 理想最大位移: {conservative['ideal_max_displacement_mm']:.3f} mm；保守最大位移: {conservative['conservative_max_displacement_mm']:.3f} mm",
        f"- 理想最大杆力: {conservative['ideal_max_abs_force_n']:.2f} N；保守最大杆力: {conservative['conservative_max_abs_force_n']:.2f} N",
        f"- 理想最大屈曲利用率: {conservative['ideal_max_buckling_utilization']:.3f}；保守最大屈曲利用率: {conservative['conservative_max_buckling_utilization']:.3f}",
        f"- 修正系数: 位移 x{conservative['displacement_factor']:.2f}，杆力 x{conservative['force_factor']:.2f}，屈曲 x{conservative['buckling_factor']:.2f}",
        "",
        "## 结果合理性检查",
        f"- 总竖向荷载: {sanity_summary['total_vertical_load_n']:.2f} N；竖向反力合计: {sanity_summary['total_vertical_reaction_n']:.2f} N；差异: {sanity_summary['balance_ratio']:.1%}",
        f"- 最大位移节点: {sanity_summary['max_displacement_node']}；跨向位置 ratio: {sanity_summary['max_displacement_x_ratio']}",
        f"- 最大压杆区域判断: {sanity_summary['max_compression_zone']}；最大拉杆区域判断: {sanity_summary['max_tension_zone']}",
        f"- 刚度矩阵条件数: {sanity_summary['condition_number']:.3e}",
        f"- 复核提示: {sanity_summary['review_message']}",
        "",
        "## V1.7 验证结论",
        f"- 竖向荷载/反力差异: {validation_summary['vertical_error_ratio']:.1%}",
        f"- 水平反力比: {validation_summary['horizontal_reaction_ratio']:.1%}",
        f"- 最大竖向位移节点: {validation_summary['max_vertical_displacement_node']}；最大总位移节点: {validation_summary['max_total_displacement_node']}",
        f"- 最大压杆: member {validation_summary['max_compression_member']}，区域 {validation_summary['max_compression_zone']}",
        f"- 最大拉杆: member {validation_summary['max_tension_member']}，区域 {validation_summary['max_tension_zone']}",
        f"- 疑似机构或近似机构: {'是' if validation_summary['suspected_mechanism'] else '否'}",
        f"- 验证复核提示: {validation_summary['review_message']}",
        "- 当前 FEM 结果不可靠。需要先修复节点连接、支座设置和桥面加载节点后，再进行受力分析。",
        "",
        "## 荷载工况对比",
    ])
    overlength_insert_at = lines.index("## 超长杆件处理") + 1
    overlength_lines = [""]
    if overlength_material_issues:
        for issue in overlength_material_issues:
            overlength_lines.append(
                f"- member {issue['member_id']}: {issue['length_mm']:.1f}mm，{issue['status']}，建议拆分: {issue['suggested_split'] or '无'}"
            )
    else:
        overlength_lines.append("- 无")
    lines[overlength_insert_at:overlength_insert_at] = overlength_lines
    for row in load_case_df.itertuples():
        lines.append(
            f"- {row.case}: {'成功' if row.success else '失败'}，最大竖向位移 {float(row.max_vertical_displacement_mm):.3f} mm，最大杆力 {float(row.max_abs_force_n):.2f} N，说明: {row.message}"
        )
    lines.extend(["", "## 屈曲风险最高 10 根压杆"])
    if buckling_top10.empty:
        lines.append("- 无可用压杆屈曲结果。")
    else:
        for row in buckling_top10.itertuples():
            lines.append(
                f"- member {int(row.member_id)}: 轴力 {row.axial_force_n:.2f} N，Pcr {row.euler_pcr_n:.2f} N，屈曲利用率 {row.buckling_utilization:.3f}，等级 {row.risk_level}"
            )
    lines.extend(["", "## 前 10 根危险杆件"])
    if top_risk.empty:
        lines.append("- 无可用杆件风险结果。")
    else:
        for row in top_risk.itertuples():
            lines.append(f"- member {int(row.member_id)}: {row.force_type}，轴力 {row.axial_force_n:.2f} N，risk {row.risk_score:.3f}，等级 {row.risk_level}")
    lines.extend(["", "## 前 10 个问题节点"])
    if not problem_nodes:
        lines.append("- 未发现明显问题节点。")
    else:
        for node_id in problem_nodes:
            lines.append(f"- node {node_id}: 连接杆件数 {diagnostics['member_counts'].get(node_id, 0)}")
    lines.extend(["", "## 模型识别错误提示"])
    for label, value in issues.items():
        lines.append(f"- {label}: {value if value else '无'}")
    if not sanity_df.empty:
        lines.extend(["", "## Sanity Check 具体问题"])
        for row in sanity_df.itertuples():
            lines.append(f"- [{row.level}] {row.check}: {row.message}")
    if not validation_df.empty:
        lines.extend(["", "## Validation Check 具体问题"])
        for row in validation_df.itertuples():
            lines.append(f"- [{row.status}] {row.check}: {row.message}")
        lines.append(f"- {REVIEW_WARNING}")
    lines.extend(
        [
            "",
            "## 建议加固位置",
        ]
    )
    for item in sorted(suggestions, key=lambda x: x["priority"]):
        lines.append(f"- P{item['priority']} {item['topic']}: {item['suggestion']} 目标: {item['target'] or '无明显目标'}")
    panel_summary = (
        f"本桥模型按 1:{metadata.scale:g} 换算为实桥进行三维铰接桁架分析，实桥估计跨度 {geom['span_mm']:.0f}mm、宽度 {geom['width_mm']:.0f}mm。"
        f"程序识别模型木杆构件 {rod_summary['model_rod_count']} 段，最终按 {material_summary['stock_wood_count']} 根 1300mm 标准木杆计材料分，原始材料分 {material_summary['raw_material_score']}。"
        f"控制工况为 {governing.name}，理想最大竖向位移 {governing.max_vertical_displacement_mm:.2f}mm，保守修正后约 {conservative['conservative_max_displacement_mm']:.2f}mm。"
        f"最危险杆件为 member {_row_value(max_risk, 'member_id')}。建议优先复核节点连接、桥面 X 形拉结、上弦压杆侧向约束和中跨下弦加强。"
        "该结果基于简化轴力桁架模型，连接滑移、施工误差和材料缺陷仍需实体加载试验修正。"
    )
    lines.extend(
        [
            "",
            "## 建议优先修改顺序",
            "1. 先处理单杆节点、悬空杆件和未聚类杆端。",
            "2. 再加强风险最高的压杆侧向约束和中跨下弦。",
            "3. 然后补齐桥面 X 形拉结和端部支座底部拉结。",
            "4. 最后复核超长杆件拼接和金属节点板布置。",
            "",
            "## 展板用 200 字结构分析总结",
            panel_summary,
            "",
            "## 程序假设和局限性",
            "该分析是基于简化三维桁架有限元模型，节点刚度、木材缺陷、连接滑移、绳索预拉力和施工误差都需要通过实体测试修正。",
            "V1.7 默认所有杆件只承受轴力，节点按铰接处理；Rhino 中复杂 Brep/Extrusion 的中心线由包围盒长轴估算，非直杆或倾斜截面可能需要人工复核。",
            "若某个荷载工况显示刚度矩阵奇异，通常意味着结构存在机构、节点未正确聚类、杆件未闭合或支座约束不足。",
            "",
            "## 识别警告与功能说明",
        ]
    )
    lines.extend([f"- {msg}" for msg in metadata.messages])
    lines.extend([f"- {item}" for item in skipped[:80]] or ["- 无"])
    if len(skipped) > 80:
        lines.append(f"- 另有 {len(skipped) - 80} 条跳过信息未列出")
    (output_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def analyze_model(model_path: str | Path, config: dict[str, Any], output_dir: Path, label: str = "model") -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    log_step(f"[{label}] 读取 Rhino 3dm 模型并识别木杆")
    rods, skipped, metadata = parse_3dm_model(model_path, config)
    rods, manual_info = apply_member_overrides(rods, config)
    pd.DataFrame([metadata.dimensions_record()]).to_csv(output_dir / "model_dimensions.csv", index=False)

    log_step(f"[{label}] 生成木杆清单和统计摘要")
    standard_length = float(config["section"]["standard_length_mm"])
    inventory = inventory_dataframe(rods, standard_length)
    inventory.to_csv(output_dir / "rod_inventory.csv", index=False)
    rod_summary = summarize_rods(rods, config)
    write_inventory_summary(str(output_dir / "rod_inventory_summary.md"), rod_summary)
    corrected_inventory, cut_plan_df, material_summary, overlength_material_issues = optimize_cut_stock(rods, config, output_dir)
    rod_summary["stock_wood_count"] = material_summary["stock_wood_count"]
    rod_summary["raw_material_score"] = material_summary["raw_material_score"]
    rod_summary["capped_material_score"] = material_summary["capped_material_score"]
    rod_summary["material_score"] = material_summary["raw_material_score"]
    corrected_inventory.to_csv(output_dir / "corrected_rod_inventory.csv", index=False)
    cut_plan_df.to_csv(output_dir / "cut_plan.csv", index=False)
    write_cut_plan_markdown(output_dir / "cut_plan.md", cut_plan_df, material_summary, overlength_material_issues)
    write_material_count_comparison(
        output_dir / "material_count_comparison.md",
        rod_summary["model_rod_count"],
        rod_summary["equivalent_standard_count"],
        material_summary,
        overlength_material_issues,
    )

    log_step(f"[{label}] 聚类杆端并建立节点/杆件网络")
    tolerance_model = float(config["model"].get("node_cluster_tolerance_model_mm", config["model"].get("endpoint_cluster_tolerance_model_mm", 5.0)))
    tolerance = tolerance_model * float(config["model"].get("scale", 10.0))
    model, diagnostics, clean_info, node_override_df = build_clean_centerline_model(rods, config, output_dir, tolerance)
    write_clean_centerline_outputs(model, output_dir)
    nodes_dataframe(model).to_csv(output_dir / "nodes.csv", index=False)
    members_dataframe(model).to_csv(output_dir / "members.csv", index=False)
    write_node_quality_report(output_dir / "node_quality_report.md", diagnostics, clean_info)
    node_override_df.to_csv(output_dir / "manual_node_overrides_used.csv", index=False)
    pd.DataFrame(diagnostics["close_unclustered_endpoints"]).to_csv(output_dir / "close_unclustered_endpoints.csv", index=False)
    geom = geometry_checks(model, rods, config)

    log_step(f"[{label}] 执行 V2.1 标准木杆近似长度排料统计")
    v21_stock_summary, v21_visualization_failures = run_v21_material_stock_counting(output_dir, config, material_summary, rod_summary)

    log_step(f"[{label}] 检查支座和桥面加载节点是否已人工确认")
    manual_ready, support_load_info = manual_fem_inputs_confirmed(config)
    supports = resolve_supports(model, config) if manual_ready else []
    deck_node_ids = deck_nodes(model, config).astype(int).tolist() if manual_ready else []

    log_step(f"[{label}] 执行中心线模型验收和 FEM 前置检查")
    clean_nodes_df = pd.read_csv(output_dir / "clean_nodes.csv")
    clean_members_df = pd.read_csv(output_dir / "clean_members.csv")
    centerline_validation, centerline_components, duplicate_pairs = centerline_basic_checks(clean_nodes_df, clean_members_df, diagnostics, config)
    pd.DataFrame(centerline_components).drop(columns=["node_ids", "member_ids"], errors="ignore").to_csv(output_dir / "connected_components.csv", index=False)
    pd.DataFrame(duplicate_pairs).to_csv(output_dir / "duplicate_members.csv", index=False)
    deck_check_df, deck_check_summary = check_deck_nodes(model, deck_node_ids, supports)
    support_check_df, support_check_summary = check_support_nodes(model, supports)
    deck_check_df.to_csv(output_dir / "deck_node_check.csv", index=False)
    support_check_df.to_csv(output_dir / "support_node_check.csv", index=False)
    write_check_markdown(output_dir / "deck_node_check.md", "Deck Node Check", deck_check_df, deck_check_summary)
    write_check_markdown(output_dir / "support_node_check.md", "Support Node Check", support_check_df, support_check_summary)
    precheck_df, precheck_summary = fem_precheck(
        model,
        config,
        supports,
        deck_node_ids,
        centerline_validation,
        support_check_summary,
        deck_check_summary,
        manual_ready,
    )
    precheck_df.to_csv(output_dir / "fem_precheck.csv", index=False)
    fem_ready = bool(precheck_summary["fem_precheck_passed"])
    centerline_score = score_centerline_model(metadata, centerline_validation, support_check_summary, deck_check_summary, material_summary, precheck_summary)
    pd.DataFrame([centerline_score]).to_csv(output_dir / "centerline_model_score.csv", index=False)
    write_centerline_validation_report(output_dir / "centerline_validation_report.md", centerline_validation, centerline_components, precheck_summary, centerline_score)

    if fem_ready:
        supports = resolve_supports(model, config)
        deck_node_ids = deck_nodes(model, config).astype(int).tolist()
        log_step(f"[{label}] 已人工确认支座/加载节点，运行简化 3D 桁架有限元荷载工况")
        results, moving_df = run_standard_cases(model, config, supports)
        moving_df.to_csv(output_dir / "moving_load_envelope.csv", index=False)
        for name, result in results.items():
            result.member_results.to_csv(output_dir / f"member_forces_{name}.csv", index=False)
        load_case_df = load_case_summary_dataframe(results, moving_df)
        load_case_df.to_csv(output_dir / "load_case_comparison.csv", index=False)

        governing = choose_governing(results)
        governing.member_results.to_csv(output_dir / "member_forces_governing.csv", index=False)
        if governing.member_results.empty:
            pd.DataFrame().to_csv(output_dir / "buckling_check.csv", index=False)
        else:
            governing.member_results[governing.member_results["force_type"] == "compression"].to_csv(output_dir / "buckling_check.csv", index=False)
    else:
        message = precheck_summary["message"]
        if not manual_ready:
            message = RELIABLE_FEM_BLOCK_MESSAGE
        log_step(f"[{label}] {message} 本次跳过 FEM 求解")
        governing = manual_confirmation_result(len(model.nodes), message)
        results = {"manual_confirmation_required": governing}
        moving_df = pd.DataFrame(columns=["position_ratio", "max_vertical_displacement_mm", "max_abs_force_n", "success", "message"])
        moving_df.to_csv(output_dir / "moving_load_envelope.csv", index=False)
        load_case_df = load_case_summary_dataframe(results, moving_df)
        load_case_df.to_csv(output_dir / "load_case_comparison.csv", index=False)
        governing.member_results.to_csv(output_dir / "member_forces_governing.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "buckling_check.csv", index=False)

    opensees_result = None
    opensees_visualization_failures: list[str] = []
    solver_comparison_df = pd.DataFrame()
    solver_comparison_summary = {
        "comparison_available": False,
        "max_error_percent": None,
        "acceptable": False,
        "message": "solver_backend is numpy; OpenSeesPy comparison not requested.",
    }
    if opensees_requested(config):
        log_step(f"[{label}] V2.0 OpenSeesPy backend requested")
        opensees_result = run_opensees_backend(output_dir, config)
        fixed_for_plot, roller_for_plot = support_nodes_from_config(config)
        deck_for_plot = deck_nodes_from_config(config)
        opensees_visualization_failures = create_opensees_visualizations(output_dir, opensees_result, config, fixed_for_plot, roller_for_plot, deck_for_plot)
        if str(config.get("solver_backend", "numpy")).strip().lower() == "both":
            solver_comparison_df, solver_comparison_summary = compare_solvers(results, opensees_result, output_dir)
    else:
        log_step(f"[{label}] solver_backend=numpy，跳过 OpenSeesPy backend")

    log_step(f"[{label}] 执行结果验证、合理性检查和保守修正")
    sanity_df, sanity_summary = run_sanity_checks(model, governing, config, deck_node_ids)
    sanity_df.to_csv(output_dir / "sanity_check.csv", index=False)
    validation_df, validation_summary = run_validation_checks(model, governing, config, deck_node_ids)
    validation_df.to_csv(output_dir / "validation_check.csv", index=False)
    conservative = conservative_results(governing, config)
    pd.DataFrame([conservative]).to_csv(output_dir / "conservative_results.csv", index=False)
    issues = recognition_issues(rod_summary, diagnostics, skipped, rods, config)
    issue_rows = [{"issue_type": key, "value": str(value)} for key, value in issues.items()]
    pd.DataFrame(issue_rows).to_csv(output_dir / "recognition_issues.csv", index=False)
    confidence_breakdown = confidence_scores(metadata, material_summary, diagnostics, support_load_info, fem_ready, governing)
    pd.DataFrame([confidence_breakdown]).to_csv(output_dir / "confidence_scores.csv", index=False)

    log_step(f"[{label}] 生成自动加固建议")
    suggestions, suggestions_md = generate_fix_suggestions(model, governing, diagnostics, rod_summary, deck_node_ids, supports)
    write_fix_suggestions(output_dir / "fix_suggestions.md", suggestions_md)
    write_manual_review_checklist(output_dir / "manual_review_checklist.md", diagnostics, issues, supports, deck_node_ids, governing, suggestions)
    write_reinforcement_priority(output_dir / "reinforcement_priority.md", suggestions, rod_summary, validation_summary)
    write_board_text(output_dir / "board_text.md", geom, rod_summary, governing, conservative, validation_summary)

    log_step(f"[{label}] 生成图像")
    visualization_failures = create_visualizations(
        rods,
        model,
        governing,
        moving_df,
        load_case_df,
        output_dir,
        config,
        rod_summary=rod_summary,
        supports=supports,
        deck_node_ids=deck_node_ids,
        suggestions=suggestions,
        clean_diagnostics=diagnostics,
        support_load_info=support_load_info,
        centerline_components=centerline_components,
        centerline_score=centerline_score,
    )

    log_step(f"[{label}] 生成 Markdown 报告")
    write_report(
        output_dir,
        Path(model_path),
        metadata,
        rod_summary,
        geom,
        diagnostics,
        skipped,
        supports,
        deck_node_ids,
        load_case_df,
        moving_df,
        governing,
        sanity_df,
        sanity_summary,
        validation_df,
        validation_summary,
        conservative,
        issues,
        suggestions,
        manual_info,
        visualization_failures,
        material_summary,
        overlength_material_issues,
        clean_info,
        support_load_info,
        confidence_breakdown,
        centerline_validation,
        centerline_score,
        precheck_summary,
        opensees_result,
        solver_comparison_summary,
        opensees_visualization_failures,
        v21_stock_summary,
        v21_visualization_failures,
        fem_ready,
        config,
    )
    log_step(f"[{label}] 生成 V2.2 结构问题诊断与修改建议")
    structural_diagnosis, diagnosis_visualization_failures = run_v22_structural_diagnosis(output_dir, config)
    return {
        "model_path": Path(model_path),
        "output_dir": output_dir,
        "rods": rods,
        "metadata": metadata,
        "rod_summary": rod_summary,
        "material_summary": material_summary,
        "model": model,
        "diagnostics": diagnostics,
        "geom": geom,
        "supports": supports,
        "deck_node_ids": deck_node_ids,
        "governing": governing,
        "sanity_df": sanity_df,
        "sanity_summary": sanity_summary,
        "validation_df": validation_df,
        "validation_summary": validation_summary,
        "conservative": conservative,
        "suggestions": suggestions,
        "issues": issues,
        "manual_info": manual_info,
        "clean_info": clean_info,
        "support_load_info": support_load_info,
        "confidence_breakdown": confidence_breakdown,
        "centerline_validation": centerline_validation,
        "centerline_score": centerline_score,
        "fem_precheck_summary": precheck_summary,
        "opensees_result": opensees_result,
        "solver_comparison_df": solver_comparison_df,
        "solver_comparison_summary": solver_comparison_summary,
        "v21_stock_summary": v21_stock_summary,
        "structural_diagnosis": structural_diagnosis,
        "diagnosis_visualization_failures": diagnosis_visualization_failures,
        "fem_ran": fem_ready,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a Rhino 3dm wood truss bridge model.")
    parser.add_argument("--model", required=True, help="Path to Rhino .3dm model")
    parser.add_argument("--compare-model", default=None, help="Optional second Rhino .3dm model for before/after comparison")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--enable-stock-pairing", action="store_true", help="Enable V2.1 approximate stock pairing material count")
    parser.add_argument("--round-step", type=float, default=None, help="Length rounding step in mm for V2.1 stock pairing")
    parser.add_argument("--pair-tolerance", type=float, default=None, help="Pair tolerance in mm for V2.1 stock pairing")
    parser.add_argument("--manual-stock-count", type=int, default=None, help="Manual stock count for comparison in V2.1 report")
    parser.add_argument("--generate-diagnosis", action="store_true", help="Generate V2.2 structural diagnosis report")
    parser.add_argument("--generate-fix-suggestions", action="store_true", help="Generate V2.2 prioritized fix suggestions")
    parser.add_argument("--diagnosis-top-n", type=int, default=None, help="Number of top V2.2 issues to include in report")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_step("读取配置")
    config = apply_cli_overrides(load_config(args.config), args)
    primary = analyze_model(args.model, config, output_dir, label="primary")
    print(f"分析完成。输出目录: {output_dir.resolve()}")
    print(f"识别模型杆件段数: {primary['rod_summary']['model_rod_count']} 段；最终标准木杆: {primary['material_summary']['stock_wood_count']} 根")
    print(f"控制工况: {primary['governing'].name}；最大竖向位移: {primary['governing'].max_vertical_displacement_mm:.3f} mm")

    if args.compare_model:
        log_step("执行修改前后模型对比")
        compare_dir = output_dir / "compare_model_outputs"
        new = analyze_model(args.compare_model, config, compare_dir, label="compare")
        rows = comparison_rows(primary, new)
        pd.DataFrame(rows).to_csv(output_dir / "comparison_summary.csv", index=False)
        write_comparison_report(output_dir / "comparison_report.md", rows, str(args.model), str(args.compare_model))
        plot_comparison_summary(rows, output_dir, int(config.get("visualization", {}).get("dpi", 180)))
        print(f"对比完成。报告: {(output_dir / 'comparison_report.md').resolve()}")


if __name__ == "__main__":
    main()
