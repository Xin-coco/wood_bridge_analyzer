from __future__ import annotations

from pathlib import Path
from typing import Any

import math
import pandas as pd


EXCLUDED_MEMBER_TYPES = {"rope", "metal_node", "support", "load_marker", "non_structural", "ignored"}


def _bool_value(value: Any, default: bool = True) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def _text_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return "" if text.strip().lower() in {"nan", "none"} else text


def material_stock_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("material_stock_counting", {}) or {}
    stock = cfg.get("stock", {}) or {}
    rounding = cfg.get("length_rounding", {}) or {}
    pairing = cfg.get("pairing", {}) or {}
    cutting_loss = cfg.get("cutting_loss", {}) or {}
    manual = cfg.get("manual_compare", {}) or {}
    scoring = cfg.get("scoring", {}) or {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "stock_length_mm": float(stock.get("stock_length_mm", 1300.0)),
        "stock_width_mm": float(stock.get("stock_width_mm", 30.0)),
        "stock_height_mm": float(stock.get("stock_height_mm", 80.0)),
        "rounding_enabled": bool(rounding.get("enabled", True)),
        "rounding_method": str(rounding.get("method", "nearest")),
        "step_mm": float(rounding.get("step_mm", 50.0)),
        "rounding_tolerance_mm": float(rounding.get("tolerance_mm", 25.0)),
        "keep_original_length": bool(rounding.get("keep_original_length", True)),
        "pairing_enabled": bool(pairing.get("enabled", True)),
        "target_length_mm": float(pairing.get("target_length_mm", stock.get("stock_length_mm", 1300.0))),
        "pair_tolerance_mm": float(pairing.get("pair_tolerance_mm", 25.0)),
        "allow_under_target": bool(pairing.get("allow_under_target", True)),
        "allow_over_target": bool(pairing.get("allow_over_target", False)),
        "max_pieces_per_stock": int(pairing.get("max_pieces_per_stock", 2)),
        "prefer_exact_pair": bool(pairing.get("prefer_exact_pair", True)),
        "prefer_min_waste": bool(pairing.get("prefer_min_waste", True)),
        "consider_saw_kerf": bool(cutting_loss.get("consider_saw_kerf", False)),
        "saw_kerf_mm": float(cutting_loss.get("saw_kerf_mm", 3.0)),
        "trim_allowance_mm": float(cutting_loss.get("trim_allowance_mm", 0.0)),
        "manual_stock_count": int(manual.get("manual_stock_count", 46)),
        "prefer_manual_stock_count": bool(manual.get("prefer_manual_stock_count", True)),
        "base_stock_count": int(scoring.get("base_stock_count", 60)),
        "base_score": float(scoring.get("base_score", 20.0)),
        "cap_score_at_20": bool(scoring.get("cap_score_at_20", False)),
    }


def load_material_members(output_dir: Path) -> pd.DataFrame:
    corrected_path = output_dir / "corrected_rod_inventory.csv"
    clean_path = output_dir / "clean_members.csv"
    if corrected_path.exists():
        df = pd.read_csv(corrected_path)
        length_col = "corrected_length_mm" if "corrected_length_mm" in df.columns else "original_length_mm"
        rows = []
        for row in df.to_dict("records"):
            action = str(row.get("action", "keep") or "keep").strip()
            member_type = str(row.get("member_type", "wood") or "wood").strip()
            include = _bool_value(row.get("include_in_material_count", row.get("included_in_stock_count", True)))
            if action in {"ignore", "duplicate_of", "non_structural"}:
                include = False
            if member_type in EXCLUDED_MEMBER_TYPES:
                include = False
            rows.append(
                {
                    "member_id": int(row["member_id"]),
                    "member_type": member_type,
                    "node_i": row.get("node_i", ""),
                    "node_j": row.get("node_j", ""),
                    "real_length_mm": float(row.get(length_col, row.get("original_length_mm", 0.0))),
                    "is_structural": _bool_value(row.get("is_structural", True)),
                    "include_in_material_count": include,
                    "source": "corrected_rod_inventory.csv",
                    "note": _text_value(row.get("note", "")),
                }
            )
        return pd.DataFrame(rows)
    if clean_path.exists():
        df = pd.read_csv(clean_path)
        length_col = "length_mm" if "length_mm" in df.columns else "real_length_mm"
        rows = []
        for row in df.to_dict("records"):
            member_type = str(row.get("member_type", "wood") or "wood").strip()
            include = _bool_value(row.get("include_in_material_count", True)) and member_type not in EXCLUDED_MEMBER_TYPES
            rows.append(
                {
                    "member_id": int(row["member_id"]),
                    "member_type": member_type,
                    "node_i": int(row.get("node_i", -1)),
                    "node_j": int(row.get("node_j", -1)),
                    "real_length_mm": float(row.get(length_col, 0.0)),
                    "is_structural": _bool_value(row.get("is_structural", True)),
                    "include_in_material_count": include,
                    "source": "clean_members.csv",
                    "note": "",
                }
            )
        return pd.DataFrame(rows)
    raise FileNotFoundError("V2.1 material stock counting requires corrected_rod_inventory.csv or clean_members.csv.")


def round_length(length_mm: float, step_mm: float, method: str) -> float:
    if step_mm <= 0:
        return float(length_mm)
    if method == "floor":
        return math.floor(length_mm / step_mm) * step_mm
    if method == "ceil":
        return math.ceil(length_mm / step_mm) * step_mm
    return round(length_mm / step_mm) * step_mm


def rounded_member_lengths(members_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = material_stock_config(config)
    rows = []
    for row in members_df.itertuples():
        include = bool(row.include_in_material_count) and bool(row.is_structural) and str(row.member_type) == "wood"
        original = float(row.real_length_mm)
        rounded = round_length(original, cfg["step_mm"], cfg["rounding_method"]) if cfg["rounding_enabled"] else original
        rows.append(
            {
                "member_id": int(row.member_id),
                "member_type": str(row.member_type),
                "node_i": getattr(row, "node_i", ""),
                "node_j": getattr(row, "node_j", ""),
                "original_length_mm": original,
                "rounded_length_mm": rounded,
                "rounding_error_mm": rounded - original,
                "rounding_method": cfg["rounding_method"],
                "include_in_stock_count": include,
                "used_in_stock_id": "",
                "note": getattr(row, "note", ""),
            }
        )
    return pd.DataFrame(rows)
