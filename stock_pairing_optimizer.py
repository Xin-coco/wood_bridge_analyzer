from __future__ import annotations

from typing import Any
import math

import pandas as pd

from material_length_rounding import material_stock_config


def _effective_sum(lengths: list[float], cfg: dict[str, Any]) -> float:
    kerf = cfg["saw_kerf_mm"] if cfg["consider_saw_kerf"] and len(lengths) > 1 else 0.0
    trim = cfg["trim_allowance_mm"] * len(lengths)
    return sum(lengths) + kerf + trim


def _can_fit(lengths: list[float], cfg: dict[str, Any]) -> bool:
    total = _effective_sum(lengths, cfg)
    if cfg["allow_over_target"]:
        return total <= cfg["target_length_mm"] + cfg["pair_tolerance_mm"]
    return total <= cfg["target_length_mm"] + 1e-9


def optimize_pairing(rounded_df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cfg = material_stock_config(config)
    target = cfg["target_length_mm"]
    pair_tol = cfg["pair_tolerance_mm"]
    stock_rows: list[dict[str, Any]] = []
    oversized_rows: list[dict[str, Any]] = []
    working = rounded_df.copy()
    working["used_in_stock_id"] = ""

    usable = working[working["include_in_stock_count"] == True].copy()  # noqa: E712
    ignored = working[working["include_in_stock_count"] != True].copy()  # noqa: E712

    stock_id = 1
    used: set[int] = set()

    for row in ignored.itertuples():
        stock_rows.append(
            {
                "stock_id": "",
                "member_ids": str(int(row.member_id)),
                "member_lengths_mm": f"{float(row.rounded_length_mm):.1f}",
                "total_used_mm": 0.0,
                "waste_mm": "",
                "pairing_type": "ignored",
                "note": "not included in material stock count",
            }
        )

    for row in usable.sort_values("rounded_length_mm", ascending=False).itertuples():
        member_id = int(row.member_id)
        length = float(row.rounded_length_mm)
        if length > target + 1e-9:
            split_count = max(1, int(math.ceil(length / target)))
            oversized_rows.append(
                {
                    "member_id": member_id,
                    "original_length_mm": float(row.original_length_mm),
                    "rounded_length_mm": length,
                    "status": "oversized_need_split",
                    "suggested_split_count": split_count,
                    "note": "需要拆分或重新设计；未进入普通两两配对。",
                }
            )
            first_stock_id = stock_id
            remaining_length = length
            for split_idx in range(split_count):
                segment_length = min(target, remaining_length)
                remaining_length -= segment_length
                stock_rows.append(
                    {
                        "stock_id": stock_id,
                        "member_ids": str(member_id),
                        "member_lengths_mm": f"{segment_length:.1f}",
                        "total_used_mm": segment_length,
                        "waste_mm": max(0.0, target - min(segment_length, target)),
                        "pairing_type": "oversized_need_split",
                        "note": f"rounded length exceeds stock length; manual split required ({split_idx + 1}/{split_count})",
                    }
                )
                stock_id += 1
            working.loc[working["member_id"] == member_id, "used_in_stock_id"] = first_stock_id
            used.add(member_id)

    remaining = usable[~usable["member_id"].isin(used)].copy()
    near_full = remaining[remaining["rounded_length_mm"] >= target - pair_tol].sort_values("rounded_length_mm", ascending=False)
    for row in near_full.itertuples():
        member_id = int(row.member_id)
        length = float(row.rounded_length_mm)
        if member_id in used:
            continue
        if length <= target or cfg["allow_over_target"]:
            total = _effective_sum([length], cfg)
            stock_rows.append(
                {
                    "stock_id": stock_id,
                    "member_ids": str(member_id),
                    "member_lengths_mm": f"{length:.1f}",
                    "total_used_mm": total,
                    "waste_mm": target - total,
                    "pairing_type": "single_full_length",
                    "note": "near full stock length",
                }
            )
            working.loc[working["member_id"] == member_id, "used_in_stock_id"] = stock_id
            used.add(member_id)
            stock_id += 1

    remaining = usable[~usable["member_id"].isin(used)].copy().sort_values("rounded_length_mm", ascending=False)
    by_id = {int(row.member_id): row for row in remaining.itertuples()}

    for row in remaining.itertuples():
        member_id = int(row.member_id)
        if member_id in used:
            continue
        length = float(row.rounded_length_mm)
        best = None
        for other_id, other in by_id.items():
            if other_id == member_id or other_id in used:
                continue
            other_length = float(other.rounded_length_mm)
            total = _effective_sum([length, other_length], cfg)
            if not _can_fit([length, other_length], cfg):
                continue
            waste = target - total
            exact_rank = 0 if abs(total - target) <= pair_tol else 1
            if cfg["prefer_exact_pair"] and exact_rank == 0:
                priority = (0, abs(total - target), -other_length)
            else:
                priority = (exact_rank, waste if cfg["prefer_min_waste"] else abs(total - target), -other_length)
            if best is None or priority < best["priority"]:
                best = {"other_id": other_id, "other_length": other_length, "total": total, "waste": waste, "priority": priority}

        if best is not None:
            other_id = int(best["other_id"])
            pairing_type = "paired_exact" if abs(float(best["total"]) - target) <= pair_tol else "paired_under_target"
            stock_rows.append(
                {
                    "stock_id": stock_id,
                    "member_ids": f"{member_id};{other_id}",
                    "member_lengths_mm": f"{length:.1f};{best['other_length']:.1f}",
                    "total_used_mm": float(best["total"]),
                    "waste_mm": float(best["waste"]),
                    "pairing_type": pairing_type,
                    "note": "two-piece stock pairing",
                }
            )
            working.loc[working["member_id"].isin([member_id, other_id]), "used_in_stock_id"] = stock_id
            used.add(member_id)
            used.add(other_id)
            stock_id += 1
        else:
            total = _effective_sum([length], cfg)
            stock_rows.append(
                {
                    "stock_id": stock_id,
                    "member_ids": str(member_id),
                    "member_lengths_mm": f"{length:.1f}",
                    "total_used_mm": total,
                    "waste_mm": target - total,
                    "pairing_type": "single_unpaired",
                    "note": "no valid pair found",
                }
            )
            working.loc[working["member_id"] == member_id, "used_in_stock_id"] = stock_id
            used.add(member_id)
            stock_id += 1

    plan = pd.DataFrame(stock_rows)
    counted_plan = plan[plan["pairing_type"] != "ignored"].copy()
    program_stock_count = int(counted_plan["stock_id"].replace("", pd.NA).dropna().astype(int).nunique()) if not counted_plan.empty else 0
    manual_stock_count = cfg["manual_stock_count"]
    use_manual_stock_count = bool(cfg["prefer_manual_stock_count"] and manual_stock_count > 0)
    stock_count = int(manual_stock_count if use_manual_stock_count else program_stock_count)
    paired_count = int(counted_plan["pairing_type"].isin(["paired_exact", "paired_under_target"]).sum())
    single_count = int(counted_plan["pairing_type"].isin(["single_full_length", "single_unpaired"]).sum())
    waste_values = pd.to_numeric(counted_plan["waste_mm"], errors="coerce").dropna()
    raw_score = cfg["base_score"] + (cfg["base_stock_count"] - stock_count)
    capped_score = min(raw_score, cfg["base_score"]) if cfg["cap_score_at_20"] else raw_score
    unpaired_members = []
    if not counted_plan.empty:
        for item in counted_plan[counted_plan["pairing_type"] == "single_unpaired"]["member_ids"].tolist():
            unpaired_members.extend([int(x) for x in str(item).split(";") if str(x).strip()])
    summary = {
        "model_member_count": int(len(rounded_df)),
        "structural_member_count": int(len(usable)),
        "effective_wood_segment_count": int(len(usable)),
        "included_member_count": int(usable["include_in_stock_count"].sum()) if not usable.empty else 0,
        "rounded_length_type_count": int(usable["rounded_length_mm"].nunique()) if not usable.empty else 0,
        "paired_stock_count": paired_count,
        "single_stock_count": single_count,
        "oversized_member_count": int(len(oversized_rows)),
        "program_stock_wood_count": program_stock_count,
        "stock_wood_count": stock_count,
        "manual_stock_count": manual_stock_count,
        "stock_count_source": "manual_review" if use_manual_stock_count else "program_pairing",
        "stock_count_difference_vs_manual": program_stock_count - manual_stock_count,
        "model_vs_stock_difference": int(len(rounded_df)) - stock_count,
        "raw_material_score": raw_score,
        "capped_material_score": capped_score,
        "score_policy_note": "capped at base score" if cfg["cap_score_at_20"] else "raw score not capped",
        "total_waste_mm": float(waste_values.sum()) if not waste_values.empty else 0.0,
        "average_waste_mm": float(waste_values.mean()) if not waste_values.empty else 0.0,
        "pair_success_rate": float(paired_count / max(program_stock_count, 1)),
        "unpaired_member_ids": unpaired_members,
        "oversized_member_ids": [int(row["member_id"]) for row in oversized_rows],
        "recommended_to_use": bool(stock_count > 0),
        "stock_length_mm": target,
        "round_step_mm": cfg["step_mm"],
        "pair_tolerance_mm": pair_tol,
        "consider_saw_kerf": cfg["consider_saw_kerf"],
        "saw_kerf_mm": cfg["saw_kerf_mm"],
        "trim_allowance_mm": cfg["trim_allowance_mm"],
        "max_pieces_per_stock": cfg["max_pieces_per_stock"],
        "base_stock_count": cfg["base_stock_count"],
        "rounding_enabled": cfg["rounding_enabled"],
    }
    return working, plan, pd.DataFrame(oversized_rows), summary


def split_oversized_count(length_mm: float, stock_length_mm: float) -> int:
    return int(math.ceil(length_mm / stock_length_mm))
