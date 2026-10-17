"""analysis_core 六维评分模型。

将学生原始指标按「二级指标 -> 维度 -> 综合」三级加权口径计算成长画像得分：
1. 每个二级指标按归一化区间映射为 0~100 分（方向已对齐为越高越好）。
2. 维度得分 = 该维度各二级指标得分按其维度内权重加权。
3. 综合得分 = 六个维度得分按全局权重（config_setting.DIMENSION_WEIGHTS）加权。

所有阈值与权重集中在 config_setting，便于运营调优。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from config_setting import (
    DIMENSION_ORDER,
    DIMENSION_WEIGHTS,
    DIMENSION_SUBSCORES,
    TAG_DICT,
)


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def compute_sub_scores(record: dict) -> Dict[str, Dict[str, float]]:
    """计算每个学生各维度下二级指标得分（0~100）。

    返回 {维度键: {二级标签: 得分}}。
    """
    result: Dict[str, Dict[str, float]] = {}
    for dim in DIMENSION_ORDER:
        subs = DIMENSION_SUBSCORES.get(dim, [])
        sub_scores: Dict[str, float] = {}
        for (label, _w, field, lo, hi, higher) in subs:
            val = record.get(field)
            if val is None:
                val = lo
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = float(lo)
            if hi - lo < 1e-9:
                norm = 0.5
            else:
                norm = (val - lo) / (hi - lo)
            norm = _clip01(norm)
            if not higher:
                norm = 1.0 - norm
            sub_scores[label] = round(norm * 100, 1)
        result[dim] = sub_scores
    return result


def compute_dimension_scores(record: dict) -> Dict[str, Tuple[float, Dict[str, float]]]:
    """计算六个维度得分与其二级指标明细。

    返回 {维度键: (维度得分, {二级标签: 得分})}。
    """
    sub_all = compute_sub_scores(record)
    out: Dict[str, Tuple[float, Dict[str, float]]] = {}
    for dim in DIMENSION_ORDER:
        subs = DIMENSION_SUBSCORES.get(dim, [])
        sub_scores = sub_all[dim]
        total_w = 0.0
        acc = 0.0
        for (label, w, _field, _lo, _hi, _higher) in subs:
            s = sub_scores.get(label, 0.0)
            acc += s * w
            total_w += w
        dim_score = round(acc / total_w, 1) if total_w > 0 else 0.0
        out[dim] = (dim_score, sub_scores)
    return out


def compute_composite(dim_scores: Dict[str, Tuple[float, dict]]) -> float:
    """由六维得分按全局权重计算综合评分（0~100）。"""
    acc = 0.0
    for dim in DIMENSION_ORDER:
        ds = dim_scores[dim][0]
        acc += ds * DIMENSION_WEIGHTS.get(dim, 0.0)
    return round(acc, 2)


def build_wordcloud_tags(dim_scores: Dict[str, Tuple[float, Dict[str, float]]]) -> List[Tuple[str, float]]:
    """根据维度/二级得分生成词云标签（标签, 权重）。

    每个二级指标按其得分落入高/中/低区间，映射为正向/中性/负向标签，
    权重反映突出程度（正向标签权重=得分，负向标签权重=100-得分+20）。
    """
    tags: List[Tuple[str, float]] = []
    for dim in DIMENSION_ORDER:
        sub_scores = dim_scores[dim][1]
        for label, score in sub_scores.items():
            key = (dim, label)
            if key not in TAG_DICT:
                continue
            good, bad, mid = TAG_DICT[key]
            if score >= 75:
                tags.append((good, float(score)))
            elif score <= 50:
                tags.append((bad, float(120 - score)))
            else:
                tags.append((mid, 60.0))
    return tags


def get_personalized_advice(dim_scores: Dict[str, Tuple[float, dict]], risk_level: str) -> List[str]:
    """基于六维得分生成个性化提升建议（规则兜底）。"""
    from config_setting import DIMENSION_LABELS

    advice: List[str] = []
    for dim in DIMENSION_ORDER:
        ds = dim_scores[dim][0]
        if ds < 60:
            advice.append(f"{DIMENSION_LABELS[dim]}维度偏弱（{ds:.1f} 分）："
                          f"建议针对性加强该维度的相关活动与习惯养成。")
    if not advice:
        advice.append("整体表现良好，保持当前节奏，可尝试在竞赛、科研或社会服务上进一步突破。")

    if risk_level == "高危":
        advice.append("风险等级为高危：请尽快与辅导员联系，排查连续缺勤 / 消费异常等预警信号。")
    elif risk_level == "关注":
        advice.append("风险等级为关注：建议关注近期成绩与出勤变化，必要时寻求学业帮扶。")
    return advice
