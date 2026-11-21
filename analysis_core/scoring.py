"""analysis_core 六维评分模型

将学生原始指标按「二级指标 -> 维度 -> 综合」三级加权口径计算成长画像得分：
1. 每个二级指标按归一化区间映射为 0~100 分（方向已对齐为越高越好）。
2. 维度得分 = 该维度各二级指标得分按其维度内权重加权。
3. 综合得分 = 六个维度得分按全局权重（config_setting.DIMENSION_WEIGHTS）加权。

所有阈值与权重集中在 config_setting，便于运营调优。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import streamlit as st

from config_setting import (
    DIMENSION_LABELS,
    DIMENSION_ORDER,
    DIMENSION_WEIGHTS,
    DIMENSION_SUBSCORES,
    HANDBOOK_RULES,
    TAG_DICT,
)


def _hash_obj(obj) -> int:
    """把 dict/list 等结构稳定序列化后取哈希，供 @st.cache_data 使用。"""
    import json
    return hash(json.dumps(obj, sort_keys=True, default=str))


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


@st.cache_data(ttl=300, show_spinner=False, hash_funcs={dict: _hash_obj})
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


# 数值型词云标签的字段格式化器：原始字段 -> 「标签 + 数值」文本。
# 只收录对用户有可读意义的字段；比率类字段统一按百分比呈现。
def _fmt_pct(v: float) -> str:
    return f"{v * 100:.0f}%"


_WORDCLOUD_NUMERIC_FORMATTERS = {
    "gpa": lambda v: f"GPA {'%g' % round(float(v), 2)}",
    "attendance_rate": lambda v: f"出勤 {_fmt_pct(v)}",
    "homework_timely_rate": lambda v: f"作业及时 {_fmt_pct(v)}",
    "credit_progress": lambda v: f"学分 {_fmt_pct(v)}",
    "study_hours": lambda v: f"自习 {float(v):.0f}h",
    "library_visits": lambda v: f"图书馆 {float(v):.0f} 次",
    "competition_awards": lambda v: f"竞赛 {float(v):.0f} 项",
    "psy_score": lambda v: f"心理测评 {float(v):.0f}",
    "physical_test_score": lambda v: f"体测 {float(v):.0f}",
    "second_class_count": lambda v: f"二课活动 {float(v):.0f} 次",
    "volunteer_hours": lambda v: f"志愿 {float(v):.0f}h",
    "consumption_daily": lambda v: f"消费 {float(v):.0f}元/天",
    "sleep_regularity": lambda v: f"睡眠 {_fmt_pct(v)}",
    "dorm_regularity": lambda v: f"宿舍 {_fmt_pct(v)}",
    "breakfast_rate": lambda v: f"早餐 {_fmt_pct(v)}",
    "social_connection": lambda v: f"社交 {float(v):.0f}",
}


def _normalize_weight(w: float) -> float:
    """把各类标签权重压缩到 60~100，降低词云字号极差，同时保留相对大小。"""
    return max(60.0, min(100.0, 60.0 + float(w) * 0.4))


def _add_tag(pool: Dict[str, float], text: str, weight: float) -> None:
    """同文本去重，权重取最大并做归一化压缩。"""
    if not text:
        return
    w = _normalize_weight(max(1.0, float(weight)))
    if w > pool.get(text, 0.0):
        pool[text] = w


def build_wordcloud_tags(
    dim_scores: Dict[str, Tuple[float, Dict[str, float]]],
    record: dict | None = None,
) -> List[Tuple[str, float]]:
    """根据维度/二级得分生成词云标签（标签, 权重），总词量 40~60。

    标签来源四类：
    1. 二级指标按得分区间映射 TAG_DICT 正/中/负标签（原口径保留）；
    2. 数值型标签：有原始数值的关键字段生成「标签 + 数值」词（权重=该指标得分）；
    3. 维度级标签：每维 1 个「维度名·等级」词（权重=维度得分）；
    4. 整体标签：成长类型名、综合等级词（权重 90~100）。
    同一文本去重，权重取最大。
    """
    pool: Dict[str, float] = {}

    # 1) 二级指标正/中/负标签（原口径）
    for dim in DIMENSION_ORDER:
        sub_scores = dim_scores[dim][1]
        for label, score in sub_scores.items():
            key = (dim, label)
            if key not in TAG_DICT:
                continue
            good, bad, mid = TAG_DICT[key]
            if score >= 75:
                _add_tag(pool, good, float(score))
            elif score <= 50:
                _add_tag(pool, bad, float(120 - score))
            else:
                _add_tag(pool, mid, 60.0)

    # 2) 数值型标签：字段 -> 该字段在各二级指标中的最高得分作为权重
    if record:
        field_best: Dict[str, float] = {}
        for dim in DIMENSION_ORDER:
            for (label, _w, field, _lo, _hi, _higher) in DIMENSION_SUBSCORES.get(dim, []):
                s = dim_scores[dim][1].get(label)
                if s is None:
                    continue
                if s > field_best.get(field, -1.0):
                    field_best[field] = s
        for field, fmt in _WORDCLOUD_NUMERIC_FORMATTERS.items():
            val = record.get(field)
            if val is None or field not in field_best:
                continue
            try:
                _add_tag(pool, fmt(float(val)), field_best[field])
            except (TypeError, ValueError):
                continue
        # 校园跑：完成数/学期规定次数（权重=「校园跑完成率」二级得分）
        run_done = record.get("campus_run_valid_count")
        if run_done is not None:
            target = int(HANDBOOK_RULES["P2"]["semester_target"])
            w = dim_scores.get("health", (0, {}))[1].get("校园跑完成率", 60.0)
            _add_tag(pool, f"校园跑 {int(run_done)}/{target}", w)

        # 3) 维度级标签：「维度名·等级」（权重=维度得分）
        for dim in DIMENSION_ORDER:
            dim_score = dim_scores[dim][0]
            lv = record.get(f"level_{dim}")
            text = f"{DIMENSION_LABELS[dim]}·{lv}" if lv else DIMENSION_LABELS[dim]
            _add_tag(pool, text, dim_score)

        # 4) 整体标签：成长类型名 + 综合等级词（权重 90~100，保证视觉主词地位）
        cluster_type = record.get("cluster_type")
        if cluster_type:
            _add_tag(pool, str(cluster_type), 96.0)
        overall_level = record.get("level_overall")
        if overall_level:
            _add_tag(pool, str(overall_level), 92.0)

    return [(t, w) for t, w in pool.items()]


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
