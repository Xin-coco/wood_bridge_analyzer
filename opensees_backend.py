from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import traceback

import numpy as np
import pandas as pd

from opensees_exporter import (
    UNAVAILABLE_MESSAGE,
    load_case_names,
    load_clean_tables,
    load_check,
    load_vector_for_case,
    support_check,
    support_nodes_from_config,
    deck_nodes_from_config,
    write_check_report,
)


@dataclass
class OpenSeesRunResult:
    requested: bool
    available: bool
    success: bool
    message: str
    case_summary: pd.DataFrame
    node_displacements: pd.DataFrame
    member_forces: pd.DataFrame
    reactions: pd.DataFrame
    log_path: Path


def _write_log(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_empty_result_files(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_df = pd.DataFrame(columns=["case", "success", "status_code", "total_vertical_load_n", "total_vertical_reaction_n", "vertical_balance_error_n", "max_vertical_displacement_mm", "max_abs_force_n", "max_tension_member", "max_compression_member"])
    disp_df = pd.DataFrame(columns=["case", "node_id", "ux_mm", "uy_mm", "uz_mm", "u_abs_mm"])
    force_df = pd.DataFrame(columns=["case", "member_id", "node_i", "node_j", "length_mm", "axial_force_n", "force_type", "stress_mpa"])
    reaction_df = pd.DataFrame(columns=["case", "node_id", "rx_n", "ry_n", "rz_n"])
    case_df.to_csv(output_dir / "opensees_case_summary.csv", index=False)
    disp_df.to_csv(output_dir / "opensees_node_displacements.csv", index=False)
    force_df.to_csv(output_dir / "opensees_member_forces.csv", index=False)
    reaction_df.to_csv(output_dir / "opensees_reactions.csv", index=False)
    return case_df, disp_df, force_df, reaction_df


def _load_ops():
    try:
        import openseespy.opensees as ops  # type: ignore
        return ops, None
    except Exception as exc:  # pragma: no cover - depends on optional package
        return None, exc


def _define_model(ops: Any, nodes: pd.DataFrame, members: pd.DataFrame, config: dict[str, Any]) -> None:
    ops.wipe()
    ops.model("basic", "-ndm", int((config.get("opensees", {}) or {}).get("ndm", 3)), "-ndf", int((config.get("opensees", {}) or {}).get("ndf", 3)))
    for row in nodes.itertuples():
        ops.node(int(row.node_id) + 1, float(row.x_mm), float(row.y_mm), float(row.z_mm))
    mat_tag = 1
    elastic = float((config.get("opensees", {}).get("material", {}) or {}).get("E_MPa", config["materials"]["elastic_modulus_mpa"]))
    ops.uniaxialMaterial("Elastic", mat_tag, elastic)
    use_coro = bool((config.get("opensees", {}) or {}).get("use_corotational_truss", False))
    element_name = "corotTruss" if use_coro else "truss"
    for row in members.itertuples():
        area = float(row.area_mm2)
        ops.element(element_name, int(row.member_id) + 1, int(row.node_i) + 1, int(row.node_j) + 1, area, mat_tag)


def _apply_supports(ops: Any, fixed_nodes: list[int], roller_nodes: list[int]) -> None:
    for node_id in fixed_nodes:
        ops.fix(int(node_id) + 1, 1, 1, 1)
    for node_id in roller_nodes:
        ops.fix(int(node_id) + 1, 1, 0, 1)


def _analyze_case(
    ops: Any,
    nodes: pd.DataFrame,
    members: pd.DataFrame,
    config: dict[str, Any],
    fixed_nodes: list[int],
    roller_nodes: list[int],
    deck_nodes: list[int],
    case: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _define_model(ops, nodes, members, config)
    _apply_supports(ops, fixed_nodes, roller_nodes)
    nodal_loads, total_load = load_vector_for_case(nodes, members, deck_nodes, config, case)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for node_id, load in nodal_loads.items():
        ops.load(int(node_id) + 1, float(load[0]), float(load[1]), float(load[2]))
    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    status = int(ops.analyze(1))
    ops.reactions()

    disp_rows = []
    for row in nodes.itertuples():
        node_id = int(row.node_id)
        disp = ops.nodeDisp(node_id + 1)
        disp_rows.append({"case": case, "node_id": node_id, "ux_mm": disp[0], "uy_mm": disp[1], "uz_mm": disp[2], "u_abs_mm": float(np.linalg.norm(disp))})

    reaction_rows = []
    for node_id in sorted(set(fixed_nodes + roller_nodes)):
        reaction = ops.nodeReaction(node_id + 1)
        reaction_rows.append({"case": case, "node_id": node_id, "rx_n": reaction[0], "ry_n": reaction[1], "rz_n": reaction[2]})

    force_rows = []
    for row in members.itertuples():
        member_id = int(row.member_id)
        try:
            response = ops.eleResponse(member_id + 1, "axialForce")
            axial = float(response[0] if isinstance(response, (list, tuple)) else response)
        except Exception:
            try:
                forces = ops.eleForce(member_id + 1)
                axial = float(forces[0]) if forces else 0.0
            except Exception:
                axial = 0.0
        force_rows.append(
            {
                "case": case,
                "member_id": member_id,
                "node_i": int(row.node_i),
                "node_j": int(row.node_j),
                "length_mm": float(row.length_mm),
                "axial_force_n": axial,
                "force_type": "tension" if axial >= 0 else "compression",
                "stress_mpa": axial / max(float(row.area_mm2), 1e-9),
            }
        )

    disp_df = pd.DataFrame(disp_rows)
    force_df = pd.DataFrame(force_rows)
    reaction_df = pd.DataFrame(reaction_rows)
    max_disp = float(disp_df["uz_mm"].abs().max()) if not disp_df.empty else 0.0
    max_force = float(force_df["axial_force_n"].abs().max()) if not force_df.empty else 0.0
    total_reaction_z = float(reaction_df["rz_n"].sum()) if not reaction_df.empty else 0.0
    summary = {
        "case": case,
        "success": status == 0,
        "status_code": status,
        "total_vertical_load_n": total_load,
        "total_vertical_reaction_n": total_reaction_z,
        "vertical_balance_error_n": total_reaction_z - total_load,
        "max_vertical_displacement_mm": max_disp,
        "max_abs_force_n": max_force,
        "max_tension_member": int(force_df.sort_values("axial_force_n", ascending=False).iloc[0]["member_id"]) if not force_df.empty else "",
        "max_compression_member": int(force_df.sort_values("axial_force_n", ascending=True).iloc[0]["member_id"]) if not force_df.empty else "",
    }
    return summary, disp_df, force_df, reaction_df


def run_opensees_backend(output_dir: Path, config: dict[str, Any]) -> OpenSeesRunResult:
    log_path = output_dir / "opensees_analysis_log.txt"
    logs = ["OpenSeesPy backend start."]
    try:
        nodes, members = load_clean_tables(output_dir, config)
        support_rows, support_summary = support_check(nodes, config)
        load_rows, load_summary = load_check(nodes, config)
        write_check_report(output_dir / "opensees_support_check.md", "OpenSees Support Check", support_rows, support_summary)
        write_check_report(output_dir / "opensees_load_check.md", "OpenSees Load Check", load_rows, load_summary)
    except Exception as exc:
        logs.append(f"OpenSeesPy input export/check failed: {exc}")
        logs.append(traceback.format_exc())
        _write_log(log_path, logs)
        case_df, disp_df, force_df, reaction_df = _write_empty_result_files(output_dir)
        return OpenSeesRunResult(True, False, False, f"OpenSeesPy input export/check failed: {exc}", case_df, disp_df, force_df, reaction_df, log_path)

    ops, import_error = _load_ops()
    if ops is None:
        message = UNAVAILABLE_MESSAGE
        logs.append(message)
        logs.append(str(import_error))
        _write_log(log_path, logs)
        case_df, disp_df, force_df, reaction_df = _write_empty_result_files(output_dir)
        return OpenSeesRunResult(True, False, False, message, case_df, disp_df, force_df, reaction_df, log_path)

    try:
        if not support_summary["ok"]:
            message = "OpenSeesPy analysis stopped: support nodes are not manually specified or invalid."
            logs.append(message)
            _write_log(log_path, logs)
            case_df, disp_df, force_df, reaction_df = _write_empty_result_files(output_dir)
            return OpenSeesRunResult(True, True, False, message, case_df, disp_df, force_df, reaction_df, log_path)
        if not load_summary["ok"]:
            message = "OpenSeesPy analysis stopped: deck_nodes are not manually specified or invalid."
            logs.append(message)
            _write_log(log_path, logs)
            case_df, disp_df, force_df, reaction_df = _write_empty_result_files(output_dir)
            return OpenSeesRunResult(True, True, False, message, case_df, disp_df, force_df, reaction_df, log_path)
        fixed_nodes, roller_nodes = support_nodes_from_config(config)
        deck_nodes = deck_nodes_from_config(config)
        summaries = []
        disp_frames = []
        force_frames = []
        reaction_frames = []
        for case in load_case_names(config):
            logs.append(f"Running case: {case}")
            summary, disp_df, force_df, reaction_df = _analyze_case(ops, nodes, members, config, fixed_nodes, roller_nodes, deck_nodes, case)
            summaries.append(summary)
            disp_frames.append(disp_df)
            force_frames.append(force_df)
            reaction_frames.append(reaction_df)
            logs.append(f"Case {case}: success={summary['success']}, max_uz={summary['max_vertical_displacement_mm']:.6g} mm")
        case_df = pd.DataFrame(summaries)
        disp_all = pd.concat(disp_frames, ignore_index=True) if disp_frames else pd.DataFrame()
        force_all = pd.concat(force_frames, ignore_index=True) if force_frames else pd.DataFrame()
        reaction_all = pd.concat(reaction_frames, ignore_index=True) if reaction_frames else pd.DataFrame()
        case_df.to_csv(output_dir / "opensees_case_summary.csv", index=False)
        disp_all.to_csv(output_dir / "opensees_node_displacements.csv", index=False)
        force_all.to_csv(output_dir / "opensees_member_forces.csv", index=False)
        reaction_all.to_csv(output_dir / "opensees_reactions.csv", index=False)
        success = bool(case_df["success"].all()) if not case_df.empty else False
        message = "OpenSeesPy analysis completed." if success else "OpenSeesPy analysis completed with failed load cases."
        logs.append(message)
        _write_log(log_path, logs)
        return OpenSeesRunResult(True, True, success, message, case_df, disp_all, force_all, reaction_all, log_path)
    except Exception as exc:
        logs.append(f"OpenSeesPy backend failed: {exc}")
        logs.append(traceback.format_exc())
        _write_log(log_path, logs)
        case_df, disp_df, force_df, reaction_df = _write_empty_result_files(output_dir)
        return OpenSeesRunResult(True, True, False, f"OpenSeesPy backend failed: {exc}", case_df, disp_df, force_df, reaction_df, log_path)
