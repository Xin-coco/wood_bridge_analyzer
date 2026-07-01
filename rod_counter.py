from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from bridge_parser import Rod


def inventory_dataframe(rods: list[Rod], standard_length_mm: float) -> pd.DataFrame:
    rows = []
    for rod in rods:
        row = rod.to_record()
        row["is_overlength"] = rod.length_mm > standard_length_mm + 0.5
        row["is_short"] = rod.length_mm < 0.25 * standard_length_mm
        rows.append(row)
    return pd.DataFrame(rows)


def find_duplicate_rods(rods: list[Rod], tolerance_mm: float = 30.0) -> list[tuple[int, int]]:
    duplicates: list[tuple[int, int]] = []
    for i, a in enumerate(rods):
        for b in rods[i + 1 :]:
            same = np.linalg.norm(a.start - b.start) <= tolerance_mm and np.linalg.norm(a.end - b.end) <= tolerance_mm
            reverse = np.linalg.norm(a.start - b.end) <= tolerance_mm and np.linalg.norm(a.end - b.start) <= tolerance_mm
            if same or reverse:
                duplicates.append((a.id, b.id))
    return duplicates


def summarize_rods(rods: list[Rod], config: dict[str, Any]) -> dict[str, Any]:
    standard_length = float(config["section"]["standard_length_mm"])
    base_count = int(config["bridge"]["base_rod_count"])
    physical_count = len(rods)
    equivalent_count = sum(r.equivalent_standard_rods for r in rods)
    overlength = [r.id for r in rods if r.length_mm > standard_length + 0.5]
    short = [r.id for r in rods if r.length_mm < 0.25 * standard_length]
    duplicates = find_duplicate_rods(rods)
    material_score = 20 + (base_count - equivalent_count)
    by_layer = defaultdict(int)
    for r in rods:
        by_layer[r.layer] += 1
    return {
        "model_rod_count": physical_count,
        "physical_count": physical_count,
        "equivalent_standard_count": equivalent_count,
        "base_count": base_count,
        "exceeds_base": equivalent_count > base_count,
        "material_score": material_score,
        "overlength_ids": overlength,
        "short_ids": short,
        "duplicate_pairs": duplicates,
        "count_by_layer": dict(by_layer),
    }


def write_inventory_summary(path: str, summary: dict[str, Any]) -> None:
    lines = [
        "# Rod Inventory Summary",
        "",
        f"- 模型杆件数量: {summary['model_rod_count']}",
        f"- 等效标准木杆数量: {summary['equivalent_standard_count']}",
        f"- 基准数量: {summary['base_count']}",
        f"- 是否超过 60 根: {'是' if summary['exceeds_base'] else '否'}",
        f"- 材料成本分: {summary['material_score']}",
        f"- 超长杆件 ID: {summary['overlength_ids'] or '无'}",
        f"- 短杆件 ID: {summary['short_ids'] or '无'}",
        f"- 疑似重复杆件: {summary['duplicate_pairs'] or '无'}",
        "",
        "## 按图层统计",
    ]
    for layer, count in sorted(summary["count_by_layer"].items()):
        lines.append(f"- {layer}: {count}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
