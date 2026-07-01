from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


CASE_MAP = {
    "self_weight": "self_weight",
    "midspan_person": "central_person",
    "group_uniform": "distributed_people",
    "eccentric_walk": "eccentric",
}


def _safe_pct(new: float, old: float) -> float:
    return abs(new - old) / max(abs(old), 1.0) * 100.0


def _numpy_case_row(numpy_results: dict[str, Any], numpy_case: str) -> dict[str, Any] | None:
    result = numpy_results.get(numpy_case)
    if result is None or not getattr(result, "success", False):
        return None
    member_df = result.member_results
    max_tension = ""
    max_compression = ""
    if not member_df.empty:
        tension = member_df[member_df["force_type"] == "tension"].sort_values("axial_force_n", ascending=False).head(1)
        compression = member_df[member_df["force_type"] == "compression"].sort_values("axial_force_n", ascending=True).head(1)
        max_tension = int(tension.iloc[0]["member_id"]) if not tension.empty else ""
        max_compression = int(compression.iloc[0]["member_id"]) if not compression.empty else ""
    return {
        "max_vertical_displacement_mm": float(result.max_vertical_displacement_mm),
        "max_tension_member": max_tension,
        "max_compression_member": max_compression,
        "total_vertical_reaction_n": float(result.reactions[2::3].sum()) if len(result.reactions) else 0.0,
        "max_abs_force_n": float(result.max_abs_force_n),
    }


def compare_solvers(numpy_results: dict[str, Any], opensees_result: Any, output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if opensees_result is None or not getattr(opensees_result, "success", False) or opensees_result.case_summary.empty:
        summary = {
            "comparison_available": False,
            "max_error_percent": None,
            "acceptable": False,
            "message": "OpenSeesPy result unavailable; solver comparison skipped.",
        }
        write_solver_comparison(output_dir / "solver_comparison.md", pd.DataFrame(rows), summary)
        return pd.DataFrame(rows), summary

    for op_case, np_case in CASE_MAP.items():
        op_rows = opensees_result.case_summary[opensees_result.case_summary["case"] == op_case]
        np_row = _numpy_case_row(numpy_results, np_case)
        if op_rows.empty or np_row is None:
            continue
        op_row = op_rows.iloc[0].to_dict()
        disp_err = _safe_pct(float(op_row["max_vertical_displacement_mm"]), np_row["max_vertical_displacement_mm"])
        force_err = _safe_pct(float(op_row["max_abs_force_n"]), np_row["max_abs_force_n"])
        reaction_err = _safe_pct(float(op_row["total_vertical_reaction_n"]), np_row["total_vertical_reaction_n"])
        rows.append(
            {
                "opensees_case": op_case,
                "numpy_case": np_case,
                "numpy_max_vertical_displacement_mm": np_row["max_vertical_displacement_mm"],
                "opensees_max_vertical_displacement_mm": float(op_row["max_vertical_displacement_mm"]),
                "displacement_error_percent": disp_err,
                "numpy_max_abs_force_n": np_row["max_abs_force_n"],
                "opensees_max_abs_force_n": float(op_row["max_abs_force_n"]),
                "force_error_percent": force_err,
                "numpy_total_vertical_reaction_n": np_row["total_vertical_reaction_n"],
                "opensees_total_vertical_reaction_n": float(op_row["total_vertical_reaction_n"]),
                "reaction_error_percent": reaction_err,
                "numpy_max_tension_member": np_row["max_tension_member"],
                "opensees_max_tension_member": op_row.get("max_tension_member", ""),
                "numpy_max_compression_member": np_row["max_compression_member"],
                "opensees_max_compression_member": op_row.get("max_compression_member", ""),
                "opensees_load_reaction_diff_n": float(op_row.get("vertical_balance_error_n", 0.0)),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        summary = {
            "comparison_available": False,
            "max_error_percent": None,
            "acceptable": False,
            "message": "No matching successful numpy/OpenSees load cases for comparison.",
        }
    else:
        max_error = float(df[["displacement_error_percent", "force_error_percent", "reaction_error_percent"]].max().max())
        acceptable = max_error <= 10.0
        summary = {
            "comparison_available": True,
            "max_error_percent": max_error,
            "acceptable": acceptable,
            "message": "两个求解器结果差异在 10% 以内。" if acceptable else "两个求解器结果差异较大，需复核节点、支座、荷载分配和单位设置。",
        }
    df.to_csv(output_dir / "solver_comparison.csv", index=False)
    write_solver_comparison(output_dir / "solver_comparison.md", df, summary)
    return df, summary


def write_solver_comparison(path: Path, df: pd.DataFrame, summary: dict[str, Any]) -> None:
    lines = [
        "# Solver Comparison",
        "",
        f"- comparison_available: {summary['comparison_available']}",
        f"- max_error_percent: {summary['max_error_percent']}",
        f"- acceptable: {summary['acceptable']}",
        f"- message: {summary['message']}",
        "",
    ]
    if not df.empty:
        lines.append("## Load Cases")
        for row in df.itertuples():
            lines.append(
                f"- {row.opensees_case} vs {row.numpy_case}: disp error {row.displacement_error_percent:.2f}%, "
                f"force error {row.force_error_percent:.2f}%, reaction error {row.reaction_error_percent:.2f}%"
            )
    path.write_text("\n".join(lines), encoding="utf-8")
