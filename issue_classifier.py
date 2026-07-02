from __future__ import annotations

from typing import Any


SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def severity_for_issue(issue_type: str, evidence: dict[str, Any]) -> str:
    if issue_type in {"fem_not_reliable", "singular_stiffness", "support_definition_missing"}:
        return "critical"
    if issue_type == "excessive_displacement" and float(evidence.get("max_displacement_mm", 0.0)) > float(evidence.get("limit_mm", 500.0)):
        return "critical"
    if issue_type in {"reaction_unbalanced", "high_buckling_risk", "deck_bracing_missing", "torsion_risk", "support_instability"}:
        return "high"
    if issue_type in {"single_member_nodes", "dangling_members", "disconnected_components", "oversized_members", "material_count_difference"}:
        return "medium"
    if issue_type in {"large_waste", "duplicate_members", "complex_nodes"}:
        return "low"
    return "medium"


def construction_difficulty(issue_type: str) -> str:
    if issue_type in {"support_definition_missing", "single_member_nodes", "dangling_members"}:
        return "中"
    if issue_type in {"deck_bracing_missing", "top_lateral_bracing_missing", "torsion_risk"}:
        return "低"
    if issue_type in {"high_buckling_risk", "oversized_members", "excessive_displacement"}:
        return "中-高"
    return "中"


def sort_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        issues,
        key=lambda item: (
            SEVERITY_RANK.get(str(item.get("severity", "medium")), 2),
            int(item.get("priority_rank", 999)),
            str(item.get("issue_id", "")),
        ),
    )
