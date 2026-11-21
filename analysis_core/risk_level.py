"""analysis_core 风险等级判定模块

依据 config_setting 中的风险阈值对单指标做触发判定，累加风险分，
结合权重输出综合风险等级（正常 / 关注 / 高危）与风险分。
"""

from __future__ import annotations

from config_setting import risk_threshold


def compute_risk_score(row: dict) -> tuple[float, str, list[str]]:
    """计算单个学生的风险分、风险等级与触发的指标列表。

    返回 (risk_score, risk_level, triggered)。
    """
    rt = risk_threshold
    score = 0.0
    triggered: list[str] = []

    # 成绩维度
    if int(row.get("fail_count", 0) or 0) >= rt.fail_count:
        score += 1.0
        triggered.append("挂科偏多")
    if float(row.get("grade_drop", 0) or 0) >= rt.gpa_drop_ratio:
        score += 1.0
        triggered.append("成绩骤降")

    # 考勤维度
    if int(row.get("continuous_absence", 0) or 0) >= rt.continuous_absence:
        score += 1.0
        triggered.append("连续缺勤")

    # 生活 / 消费维度
    if int(row.get("night_not_home", 0) or 0) >= rt.night_not_home:
        score += 1.0
        triggered.append("夜间未归")
    if float(row.get("consumption_drop", 0) or 0) >= rt.consumption_drop_ratio:
        score += 1.0
        triggered.append("消费骤减")

    if score >= rt.high_risk_score_threshold:
        level = "高危"
    elif score >= rt.risk_score_threshold:
        level = "关注"
    else:
        level = "正常"
    return score, level, triggered
