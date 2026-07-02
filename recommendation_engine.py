from __future__ import annotations

from typing import Any

from issue_classifier import construction_difficulty


def recommendation_for(issue_type: str, evidence: dict[str, Any]) -> dict[str, str]:
    if issue_type == "fem_not_reliable":
        return {
            "recommendation": "先人工确认 fixed_nodes、roller_nodes 和 deck_nodes，并修复未汇交节点；确认后再运行 FEM/OpenSeesPy。不要用当前位移和杆力作为承重结论。",
            "expected_effect": "使结构模型具备可靠受力分析前提，减少由支座、荷载和节点识别错误造成的假结果。",
            "material_impact": "通常不增加木杆，主要增加节点确认和模型整理工作。",
        }
    if issue_type == "single_member_nodes":
        return {
            "recommendation": "逐一检查单杆节点，把真实连接的杆端强制汇交到同一节点；实体制作时在这些节点使用金属节点板或双侧夹板，避免只靠搭接接触。",
            "expected_effect": "提高节点传力连续性，减少机构和局部滑移风险。",
            "material_impact": "基本不增加标准木杆，可能增加金属节点板或夹板数量。",
        }
    if issue_type == "dangling_members":
        return {
            "recommendation": "检查悬空杆件端点是否未与主桁架相交；将端点移动到真实节点，或删除辅助线/非结构杆件。",
            "expected_effect": "减少无效杆件和错误刚度，提高中心线模型可信度。",
            "material_impact": "可能减少被误计入的木杆段，降低材料统计偏差。",
        }
    if issue_type == "disconnected_components":
        return {
            "recommendation": "把分离的结构分量通过桥面横梁、顶部横向联系或端部门架连接成连续空间体系；模型中应保证对应杆端共享节点。",
            "expected_effect": "让左右桁架和桥面共同工作，降低整体扭转和局部失稳风险。",
            "material_impact": "可能增加 1-4 根标准木杆，若使用绳索 X 形拉结则木杆增加较少。",
        }
    if issue_type in {"deck_bracing_missing", "torsion_risk"}:
        return {
            "recommendation": "在桥面平面增加 X 形绳索拉结，并增加横向联系，使左右桁架通过桥面横梁形成稳定空间盒子。",
            "expected_effect": "提高桥面抗扭刚度，减少偏心行走荷载下的扭转变形。",
            "material_impact": "优先使用绳索，通常不增加标准木杆；若改用木横杆，可能增加 1-2 根。",
        }
    if issue_type == "top_lateral_bracing_missing":
        return {
            "recommendation": "在左右桁架上弦之间增加顶部横杆，或用绳索形成顶部 X 形约束，缩短上弦压杆无支撑长度。",
            "expected_effect": "提升上弦压杆侧向稳定性，降低屈曲风险。",
            "material_impact": "使用绳索时不增加木杆；使用木横杆时约增加 1-2 根标准木杆。",
        }
    if issue_type == "high_buckling_risk":
        return {
            "recommendation": "对高风险压杆增加横向约束，优先处理上弦长压杆、长斜杆和端部斜撑；必要时改为并杆或缩短无支撑长度。",
            "expected_effect": "提高欧拉屈曲安全储备，降低加载测试中突然侧弯失稳的风险。",
            "material_impact": "并杆会增加木杆用量；横向绳索约束对木杆数量影响较小。",
        }
    if issue_type == "excessive_displacement":
        return {
            "recommendation": "加强中跨下弦两侧夹板或并杆，增加桥面纵向承重杆，减小横梁间距，并在桥面下方增加 X 形绳索拉结。",
            "expected_effect": "降低中跨挠度，改善桥面竖向传力连续性。",
            "material_impact": "可能增加 2-5 根标准木杆，绳索拉结不计入木杆数量。",
        }
    if issue_type == "support_instability":
        return {
            "recommendation": "明确支座节点和地面接触点；端部增加底部拉结和闭合三角形斜撑，加强端部门架，防止支座外扩。",
            "expected_effect": "提高端部约束可靠性，减少支座反力异常和水平外扩。",
            "material_impact": "可能增加 1-3 根标准木杆；底部绳索拉结可减少木杆增量。",
        }
    if issue_type == "reaction_unbalanced":
        return {
            "recommendation": "复核荷载是否只施加到桥面节点，检查支座自由度设置和节点编号，确认总竖向荷载与支座竖向反力方向一致。",
            "expected_effect": "提高计算平衡性，避免因荷载/支座输入错误导致危险杆排序失真。",
            "material_impact": "不直接增加木杆，主要是计算模型修正。",
        }
    if issue_type == "oversized_members":
        return {
            "recommendation": "将超过 1300mm 的杆件拆分为可施工长度，并在拼接处设置金属节点板或双侧夹板；若位置为主受力杆，优先重新设计杆长。",
            "expected_effect": "使构件符合标准木杆长度限制，减少施工拼接不确定性。",
            "material_impact": "标准木杆领取数量不一定增加，但会增加拼接节点和连接件。",
        }
    if issue_type == "material_count_difference":
        return {
            "recommendation": "继续使用 manual_stock_count 作为最终材料分口径，同时复核非结构杆件、重复杆件、超长杆件拆分和是否允许三段以上排料。",
            "expected_effect": "让材料统计与人工领取数量一致，避免把构件段数误当作原材数量。",
            "material_impact": "可能减少统计口径上的木杆数量，不一定改变实际结构。",
        }
    if issue_type == "large_waste":
        return {
            "recommendation": "优化短杆排料，把长度相加接近 1300mm 的短杆配对；必要时调整局部杆长到 50mm 模数。",
            "expected_effect": "减少余料，提高材料利用率。",
            "material_impact": "可能减少 1-3 根标准木杆领取数量。",
        }
    return {
        "recommendation": "复核该问题对应的节点、杆件和施工做法，优先使用标准木杆、金属节点和绳索进行修正，不使用胶粘剂。",
        "expected_effect": "提高模型可靠性和施工可检查性。",
        "material_impact": "视具体修改而定。",
    }


def enrich_issue(issue: dict[str, Any]) -> dict[str, Any]:
    rec = recommendation_for(str(issue["issue_type"]), dict(issue.get("evidence", {})))
    issue.update(rec)
    issue["construction_difficulty"] = construction_difficulty(str(issue["issue_type"]))
    return issue
