"""web_main 学生端页面。

包含：
- 顶栏切换：综合总览 + 六个维度分页面（学业/身心/社会参与/生活适应/品德/非认知）+ AI 助手。
- 综合总览：卡片式布局，参考示例图的干净商务风格（浅灰绿主题、白卡片、圆角、浅阴影）。
- 维度页：维度得分卡、二级指标柱状图、维度分析树（Treemap）、维度建议。
- AI 成长诊断、图表 AI 解读、AI 对话助手（接口保留，LLM 调用待后续启用）。
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import tempfile

import streamlit as st

from local_database import orm_ops
from report_module import charts as chart_builder
from report_module import report_gen
from analysis_core import pipeline as analysis_pipeline
from analysis_core import scoring
from analysis_core import kmeans_model
from analysis_core import handbook_rules as hb_rules
from config_setting import (
    DIMENSION_ORDER, DIMENSION_LABELS, DIMENSION_COLORS, DIMENSION_WEIGHTS,
    DIMENSION_SUBSCORES,
    SECOND_CLASSROOM_PART_DIM,
    THEME_PRIMARY, THEME_PRIMARY_DARK, THEME_PRIMARY_LIGHT,
    THEME_BG, THEME_CARD_BG, THEME_CARD_BORDER, THEME_TEXT, THEME_TEXT_SUB,
    RISK_SOFT_BG, RISK_SOFT_BORDER, RISK_SOFT_TEXT,
    LEVEL_SOFT_BG, LEVEL_SOFT_BORDER, LEVEL_SOFT_TEXT,
    NEUTRAL_BG, NEUTRAL_BORDER,
    HANDBOOK_RULES,
    SPINNER_AI,
    build_theme_css,
)
from ai_insight import services as ai, llm_client


TAB_OVERVIEW = "综合总览"
TAB_AI = "AI 助手"
DIM_TABS = [DIMENSION_LABELS[d] for d in DIMENSION_ORDER]
ALL_TABS = [TAB_OVERVIEW] + DIM_TABS + [TAB_AI]


# ---------------------------------------------------------------------------
# 样式辅助
# ---------------------------------------------------------------------------
def _inject_theme_css() -> None:
    """注入全局 CSS：拿破仑黄暖金主题（样式统一由 config_setting 提供）。"""
    st.markdown(build_theme_css(), unsafe_allow_html=True)


def _card(title: str | None = None, accent: str = THEME_PRIMARY) -> st.delta_generator.DeltaGenerator:
    """返回一个带边框圆角的容器（Streamlit container border）。"""
    c = st.container(border=True)
    if title:
        with c:
            st.markdown(
                f'<div style="font-size:15px;font-weight:600;color:{THEME_TEXT};'
                f'margin-bottom:14px;border-left:4px solid {accent};padding-left:10px;">'
                f'{title}</div>',
                unsafe_allow_html=True,
            )
    return c


def _metric_card(title: str, value: str, subtitle: str | None = None,
                 accent: str = THEME_PRIMARY) -> str:
    """生成自定义指标卡 HTML。"""
    sub_html = (
        f'<div style="font-size:12px;color:{THEME_TEXT_SUB};margin-top:4px;">{subtitle}</div>'
        if subtitle else ""
    )
    return (
        f'<div style="background:{THEME_CARD_BG};border:1px solid {THEME_CARD_BORDER};'
        f'border-radius:12px;padding:18px;text-align:center;">'
        f'<div style="font-size:13px;color:{THEME_TEXT_SUB};margin-bottom:6px;">{title}</div>'
        f'<div style="font-size:32px;font-weight:700;color:{accent};">{value}</div>'
        f'{sub_html}</div>'
    )


def _progress_bar(label: str, value: float, color: str) -> str:
    """生成示例图风格的横向进度条 HTML（数值统一带「分」口径）。"""
    pct = max(0.0, min(100.0, float(value)))
    return (
        f'<div style="margin-bottom:14px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">'
        f'<span style="font-size:14px;color:{THEME_TEXT};font-weight:500;">{label}</span>'
        f'<span style="font-size:14px;color:{THEME_TEXT_SUB};font-weight:600;">{pct:.1f} 分</span>'
        f'</div>'
        f'<div style="width:100%;height:10px;background:{THEME_PRIMARY_LIGHT};border-radius:5px;overflow:hidden;">'
        f'<div style="width:{pct}%;height:100%;background:{color};border-radius:5px;"></div>'
        f'</div></div>'
    )


def _status_pill(label: str, level: str) -> str:
    """生成柔和色块的维度等级标签（统一使用等级口径：待改进/合格/良好/优秀）。"""
    shown = level if level in LEVEL_SOFT_TEXT else "未定级"
    bg = LEVEL_SOFT_BG.get(level, NEUTRAL_BG)
    border = LEVEL_SOFT_BORDER.get(level, NEUTRAL_BORDER)
    text = LEVEL_SOFT_TEXT.get(level, THEME_TEXT_SUB)
    return (
        f'<div style="display:inline-block;background:{bg};border:1px solid {border};'
        f'border-radius:20px;padding:5px 14px;color:{text};font-size:13px;font-weight:500;">'
        f'{label}：{shown}</div>'
    )


# 引用标签：兼容 [F01] 与 [F01, F10]（模型常把多条证据合并进一个方括号）
_CITATION_TAG_RE = re.compile(
    r"[\[【]((?:F\d{2,3})(?:\s*[,，、;；/]\s*F\d{2,3})*)[\]】]"
)


def _render_ai_text(txt: str) -> None:
    """渲染 AI 输出：[Fxx] 证据标签做高亮样式。"""
    html = _CITATION_TAG_RE.sub(
        rf'<span style="background:{THEME_PRIMARY_LIGHT};color:{THEME_PRIMARY_DARK};'
        rf'border:1px solid {THEME_CARD_BORDER};border-radius:4px;padding:0 5px;'
        rf'font-size:12px;font-weight:600;">[\1]</span>',
        txt,
    )
    st.markdown(html, unsafe_allow_html=True)


def _ai_status_banner() -> None:
    """AI 功能可用性提示（未配置密钥时温和提示）。"""
    if not llm_client.is_enabled():
        st.caption(
            "🤖 AI 功能未启用：请在 `config_setting/llm_config.yaml` 中填入云端 API Key "
            "后即可使用（数据经脱敏网关处理后发送）。"
        )


@st.cache_data(ttl=300, show_spinner=False)
def _wordcloud_png(student_id: str, tags_tuple: tuple, facecolor: str) -> str:
    """按学号缓存词云 PNG 的 base64 字符串，同一学生第二次进入直接复用。"""
    fig = chart_builder.build_wordcloud(list(tags_tuple))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=facecolor)
    try:
        from matplotlib import pyplot as _plt
        _plt.close(fig)
    except Exception:
        pass
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_safe_context(student: dict) -> dict:
    return llm_client.safe_student_profile(student)


def _get_meta(student: dict, scope: str) -> dict | None:
    """读取学生所在年级某作用域的聚类元信息。"""
    try:
        return orm_ops.get_cluster_meta(int(student.get("grade")), scope)
    except (TypeError, ValueError):
        return None


def _gap_to_next(student: dict, scope: str, score: float, level_text) -> float | None:
    """按年级聚类边界计算「距上一等级还差多少分」（作品书 3.4.2）。"""
    if not level_text:
        return None
    meta = _get_meta(student, scope)
    if not meta or not meta.get("labels") or level_text not in meta["labels"]:
        return None
    rank = meta["labels"].index(level_text)
    return kmeans_model.level_gap(score, meta.get("boundaries") or [], rank)


def _next_boundary(student: dict, scope: str, level_text) -> float | None:
    """返回当前等级的下一级边界分（最高等级返回 None）。"""
    if not level_text:
        return None
    meta = _get_meta(student, scope)
    if not meta or not meta.get("labels") or level_text not in meta["labels"]:
        return None
    rank = meta["labels"].index(level_text)
    boundaries = meta.get("boundaries") or []
    if rank >= len(boundaries):
        return None
    return float(boundaries[rank])


def _level_of_score(score: float, boundaries: list, labels: list) -> str:
    """按聚类边界把得分映射为等级文本（沿用本维度「分布驱动定级」口径）。"""
    if not labels:
        return "—"
    idx = 0
    for b in boundaries or []:
        if float(score) >= float(b):
            idx += 1
    idx = min(idx, len(labels) - 1)
    return labels[idx]


# 二级指标「原始值」展示格式化（label -> 原始值文本）
_RAW_FMT = {
    "学业成绩": lambda v: f"GPA {v:.2f}",
    "学习投入": lambda v: f"出勤率 {v * 100:.0f}%",
    "学业过程": lambda v: f"作业及时率 {v * 100:.0f}%",
    "学习能力": lambda v: f"学分完成率 {v * 100:.0f}%",
    "学术创新": lambda v: f"获奖 {v:.0f} 次",
    "心理健康": lambda v: f"测评 {v:.0f} 分",
    "压力水平": lambda v: f"压力指数 {v:.0f}",
    "身体素质": lambda v: f"体测 {v:.0f} 分",
    "体测等级": lambda v: f"折算 {v:.0f} 分",
    "校园跑完成率": lambda v: f"完成率 {v * 100:.0f}%",
    "睡眠质量": lambda v: f"规律度 {v * 100:.0f}%",
    "情绪状态": lambda v: f"测评 {v:.0f} 分",
    "第二课堂": lambda v: f"活动 {v:.0f} 次",
    "志愿服务": lambda v: f"志愿 {v:.0f} h",
    "社团任职": lambda v: ("任职 是" if v >= 0.5 else "任职 否"),
    "人际交往": lambda v: f"指数 {v:.0f}",
    "竞赛获奖": lambda v: f"获奖 {v:.0f} 次",
    "饮食规律": lambda v: f"规律度 {v * 100:.0f}%",
    "作息规律": lambda v: f"规律度 {v * 100:.0f}%",
    "消费健康": lambda v: f"日均 ¥{v:.0f}",
    "离校异常": lambda v: f"异常 {v:.0f} 次",
    "思想政治": lambda v: f"评分 {v:.0f}",
    "遵纪守法": lambda v: f"违纪 {v:.0f} 次",
    "纪律处分": lambda v: f"折算 {v:.0f} 分",
    "诚信守信": lambda v: f"记录 {v:.2f}",
    "集体意识": lambda v: f"参与度 {v * 100:.0f}%",
}


def _fmt_raw(label: str, value) -> str:
    """按指标语义格式化原始值；缺数据返回 —。负向指标在外层注明「反向计分」。"""
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    fmt = _RAW_FMT.get(label)
    return fmt(v) if fmt else f"量表 {v:.0f} 分"


@st.cache_data(ttl=300, show_spinner=False)
def _raw_field_means(class_name: str, grade: int) -> tuple:
    """计算班级 / 年级全体学生各二级指标原始字段的均值（enrich 后口径）。

    返回 (班级均值 dict, 年级均值 dict)，字段 -> 均值或 None。
    """
    fields = sorted({
        f for spec in DIMENSION_SUBSCORES.values() for (_l, _w, f, _lo, _hi, _h) in spec
    })

    def _means(records: list) -> dict:
        out: dict = {}
        if not records:
            return out
        enriched = [hb_rules.enrich_record(dict(r)) for r in records]
        for f in fields:
            vals = []
            for r in enriched:
                v = r.get(f)
                try:
                    if v is not None:
                        vals.append(float(v))
                except (TypeError, ValueError):
                    continue
            out[f] = (sum(vals) / len(vals)) if vals else None
        return out

    cm = _means(orm_ops.get_raw_students_by_class(class_name)) if class_name else {}
    gm = _means(orm_ops.get_raw_students_by_grade(int(grade))) if grade else {}
    return cm, gm


# ---------------------------------------------------------------------------
# 综合总览
# ---------------------------------------------------------------------------
def _load_json(value, default):
    """读取库中的 JSON 文本字段。"""
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


# 六维权重条展示顺序（权重从高到低：学业25/身心20/非认知18/品德15/社会12/生活10）
_WEIGHT_BAR_ORDER = ["study", "health", "non_cog", "morality", "social", "life"]


def _render_weight_bar() -> None:
    """六维权重条：宽度按综合评分权重占比，各段用维度主题色。

    悬停任一段显示该维度的二级指标数（title 原生提示）。
    """
    segments = []
    for d in _WEIGHT_BAR_ORDER:
        weight = DIMENSION_WEIGHTS[d]
        n_sub = len(DIMENSION_SUBSCORES.get(d, []))
        pct = weight * 100
        title = f"{DIMENSION_LABELS[d]} · 权重 {pct:.0f}% · {n_sub} 项二级指标"
        segments.append(
            f'<div title="{title}" style="width:{pct}%;background:{DIMENSION_COLORS[d]};'
            f'height:34px;display:flex;align-items:center;justify-content:center;'
            f'color:{THEME_CARD_BG};font-size:11px;font-weight:600;cursor:default;'
            f'overflow:hidden;white-space:nowrap;">'
            f'{DIMENSION_LABELS[d][:2]} {pct:.0f}%</div>'
        )
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin:2px 0 4px;">'
        f'<span style="font-size:13px;color:{THEME_TEXT_SUB};">六维权重构成（悬停查看各维度二级指标数）</span>'
        f'</div>'
        f'<div style="display:flex;width:100%;border-radius:8px;overflow:hidden;'
        f'border:1px solid {THEME_CARD_BORDER};">{"".join(segments)}</div>',
        unsafe_allow_html=True,
    )


def _render_level_explain(student: dict, dim_scores: dict, composite: float) -> None:
    """等级可解释下钻：二级指标 → 维度加权 → 六维权重复合，三级公式代入本人数值。"""
    with st.expander("❓ 等级怎么算的（三级公式代入本人实际数值）"):
        st.markdown(
            "**第 1 级 · 二级指标得分**　"
            "二级指标得分 = clip((原始值 − 下界) ÷ (上界 − 下界), 0, 1) × 100"
        )
        lines = ["**第 2 级 · 维度加权**　维度得分 = Σ（二级指标得分 × 维度内权重）"]
        for d in DIMENSION_ORDER:
            dim_score, sub_scores = dim_scores[d]
            terms = []
            for (sub_label, w, _f, _lo, _hi, _hb) in DIMENSION_SUBSCORES[d]:
                s = sub_scores.get(sub_label)
                if s is not None:
                    terms.append(f"{s:.1f}×{w * 100:.0f}%")
            lines.append(f"- {DIMENSION_LABELS[d]} = {' + '.join(terms)} = **{dim_score:.1f} 分**")
        st.markdown("\n".join(lines))
        comp_terms = " + ".join(
            f"{dim_scores[d][0]:.1f}×{DIMENSION_WEIGHTS[d] * 100:.0f}%"
            for d in DIMENSION_ORDER
        )
        st.markdown(
            f"**第 3 级 · 六维权重复合**　综合评分 = {comp_terms} = **{composite:.1f} 分**"
        )
        meta = _get_meta(student, "composite")
        if meta and meta.get("boundaries"):
            st.markdown(
                f"**等级划分**　综合评分在本年级内做一维 K-means 聚类（K={meta.get('k')}），"
                f"质心升序映射为「{' / '.join(meta.get('labels') or [])}」，"
                f"等级边界取相邻质心中点：{meta['boundaries']} 分；"
                f"本人 {composite:.1f} 分 → 当前等级「{student.get('level_overall') or '—'}」。"
            )


def _render_reminders(student: dict) -> None:
    """智能提醒：依据本科生手册硬性规定，只列出需要提醒 / 督促的事项。

    已达标项不逐条罗列（避免页面变成手册条文清单）。
    """
    items = _load_json(student.get("handbook_reminders"), [])
    card = _card("智能提醒 · 依据本科生手册硬性规定", accent=THEME_PRIMARY)
    with card:
        st.caption("依据《2024 级本科生手册》12 条可编程条款自动生成")
        if not items:
            st.markdown(
                f'<div style="background:{RISK_SOFT_BG["正常"]};border:1px solid {RISK_SOFT_BORDER["正常"]};'
                f'border-radius:10px;padding:12px 16px;color:{RISK_SOFT_TEXT["正常"]};font-size:14px;">'
                f'✅ 本学期暂无需要提醒的事项，保持当前节奏即可。</div>',
                unsafe_allow_html=True,
            )
            return
        st.caption(f"共 {len(items)} 条需要跟进的提示（已达标项不再罗列；点击规则编号标签可展开条款说明）")
        for idx, it in enumerate(items):
            rule = it.get("rule", "")
            rule_cfg = HANDBOOK_RULES.get(rule, {})
            rule_name = rule_cfg.get("name", "")
            if it.get("type") == "鼓励":
                bg, bd, tx, icon = (RISK_SOFT_BG["正常"], RISK_SOFT_BORDER["正常"],
                                    RISK_SOFT_TEXT["正常"], "👍")
            else:
                bg, bd, tx, icon = (RISK_SOFT_BG["关注"], RISK_SOFT_BORDER["关注"],
                                    RISK_SOFT_TEXT["关注"], "🔔")
            st.markdown(
                f'<div style="background:{bg};border-left:3px solid {bd};'
                f'border-radius:6px;padding:10px 12px;margin-bottom:4px;">'
                f'<div style="margin-bottom:4px;">'
                f'<span style="display:inline-block;background:{THEME_PRIMARY_LIGHT};'
                f'border:1px solid {THEME_PRIMARY};border-radius:4px;padding:0 8px;'
                f'color:{THEME_PRIMARY_DARK};font-size:12px;font-weight:700;">{rule}</span>'
                f'<span style="font-size:12px;color:{THEME_TEXT_SUB};margin-left:6px;">'
                f'{icon} {rule_name}</span></div>'
                f'<div style="font-size:14px;color:{tx};">{it.get("text", "")}</div></div>',
                unsafe_allow_html=True,
            )
            with st.expander(f"📖 条款说明 · {rule} {rule_name}", expanded=False):
                # 面向用户的手册条款可读文案；规则未配置 user_clause 时回退 desc
                clause = rule_cfg.get("user_clause") or rule_cfg.get("desc", "")
                st.markdown(
                    f'<div style="font-size:14px;color:{THEME_TEXT};line-height:1.7;">'
                    f'{clause}</div>',
                    unsafe_allow_html=True,
                )
                st.caption("依据：《2024 级本科生手册》，具体认定与处理以学院规定为准。")
        st.caption("以上为客观提示，具体认定与处理以学院规定为准。")


def _render_second_classroom(student: dict) -> None:
    """第二课堂成绩单：五部分计分（满分 600），donut 环形图呈现。"""
    res = hb_rules.score_second_classroom(student)
    card = _card("第二课堂成绩单 · 五部分计分", accent=THEME_PRIMARY)
    with card:
        st.caption("计分依据：手册第二部分「第二课堂成绩单」明文规则")
        d1, d2 = st.columns([3, 2])
        with d1:
            # 五段 = 五部分，各段用对应维度主题色（德/智/体/美/劳），中心显示总分/600
            part_colors = {
                name: DIMENSION_COLORS.get(SECOND_CLASSROOM_PART_DIM.get(name, ""),
                                           THEME_PRIMARY)
                for name in res["五部分"]
            }
            st.plotly_chart(
                chart_builder.build_second_classroom_donut(
                    res["五部分"], res["上限"], part_colors,
                    float(res["总分"]), float(res["满分"]),
                ),
                use_container_width=True,
            )
        with d2:
            rows = []
            for name, val in res["五部分"].items():
                cap = res["上限"][name]
                dot_color = DIMENSION_COLORS.get(
                    SECOND_CLASSROOM_PART_DIM.get(name, ""), THEME_PRIMARY)
                rows.append(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:7px 2px;border-bottom:1px solid {THEME_PRIMARY_LIGHT};font-size:13px;">'
                    f'<span style="color:{THEME_TEXT};">'
                    f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
                    f'background:{dot_color};margin-right:8px;"></span>{name}</span>'
                    f'<span style="color:{THEME_TEXT_SUB};font-weight:600;">'
                    f'{val:.0f} / {cap:.0f} 分</span></div>'
                )
            st.markdown("".join(rows), unsafe_allow_html=True)
        st.caption(
            "志愿服务每学期 ≥10h 计 20 分（超额 0.5 分/h，拓展上限 5 分）；"
            "劳动实践 ≥10h 计 20 分；寒暑假社会实践并交报告计 15 分；"
            "美育平均每学期 ≥60 分达毕业要求。"
        )


def _render_overview(student: dict) -> None:
    """综合总览页：卡片式布局。"""
    composite = float(student.get("composite_score") or 0)
    level = student.get("risk_level") or "未分析"
    cluster = student.get("cluster_label")
    # enrich 注入手册派生字段（校园跑完成率 / 体测等级分 / 处分折算分），
    # 保证页面重算的二级得分与流水线落库口径一致
    enriched = hb_rules.enrich_record(dict(student))
    dim_scores = scoring.compute_dimension_scores(enriched)
    tags = scoring.build_wordcloud_tags(dim_scores, record=enriched)

    # 顶部标题与个人信息卡片
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(
            f'<div style="font-size:22px;font-weight:700;color:{THEME_TEXT};">'
            f'学生个人成长画像</div>',
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            f'<div style="background:{THEME_CARD_BG};border:1px solid {THEME_CARD_BORDER};'
            f'border-radius:10px;padding:10px 14px;text-align:right;">'
            f'<div style="font-size:12px;color:{THEME_TEXT_SUB};">'
            f'{student.get("class_name")}</div>'
            f'<div style="font-size:13px;color:{THEME_TEXT};font-weight:500;">'
            f'{student.get("name")} · {student.get("student_id")}</div></div>',
            unsafe_allow_html=True,
        )

    _ai_status_banner()

    # 第一行：四个指标卡
    overall_level = student.get("level_overall") or "—"
    cluster_type = student.get("cluster_type") or "—"
    gap = _gap_to_next(student, "composite", composite, student.get("level_overall"))
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(_metric_card("综合评分", f"{composite:.1f}", "满分 100"), unsafe_allow_html=True)
    with m2:
        # 综合等级卡按等级着色（待改进/合格/良好/优秀 四色，色板见 config_setting）
        level_accent = LEVEL_SOFT_TEXT.get(overall_level, THEME_PRIMARY_DARK)
        st.markdown(
            _metric_card("综合等级", overall_level,
                         "年级内分布驱动的客观定级", accent=level_accent),
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            _metric_card("成长类型", cluster_type,
                         "由六维画像聚类自动命名", accent=THEME_PRIMARY),
            unsafe_allow_html=True,
        )
    with m4:
        level_color = RISK_SOFT_TEXT.get(level, THEME_TEXT_SUB)
        st.markdown(
            _metric_card("风险等级", level, "基于多源风险指标综合判定", accent=level_color),
            unsafe_allow_html=True,
        )

    # 六维权重条（悬停查看各维度二级指标数）
    _render_weight_bar()

    # 等级可解释下钻：三级公式代入本人实际数值
    _render_level_explain(student, dim_scores, composite)

    if gap is not None and gap > 0:
        boundary = _next_boundary(student, "composite", student.get("level_overall"))
        bar_html = ""
        if boundary and boundary > 0:
            pct = max(0.0, min(100.0, composite / boundary * 100.0))
            bar_html = (
                f'<div style="display:flex;align-items:center;gap:8px;margin-top:6px;">'
                f'<div style="flex:1;height:8px;background:{THEME_PRIMARY_LIGHT};'
                f'border-radius:4px;overflow:hidden;">'
                f'<div style="width:{pct:.1f}%;height:100%;background:{THEME_PRIMARY};'
                f'border-radius:4px;"></div></div>'
                f'<span style="font-size:12px;color:{THEME_TEXT_SUB};white-space:nowrap;">'
                f'{composite:.1f} → {boundary:.1f} 分</span></div>'
            )
        st.markdown(
            f'<div style="margin:6px 0 10px;font-size:13px;color:{THEME_TEXT_SUB};">'
            f'📈 距上一综合等级还差 <b style="color:{THEME_PRIMARY_DARK};">{gap:.1f}</b> 分，'
            f'可参考下方个性化建议针对性提升。{bar_html}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    # 智能提醒：只列出需要提醒 / 督促的事项
    _render_reminders(student)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # 第二行：雷达图 + 成长趋势
    c1, c2 = st.columns([7, 5])
    with c1:
        card = _card("六维能力雷达", accent=THEME_PRIMARY)
        with card:
            ref = {}
            cm = orm_ops.get_dimension_means(class_name=student.get("class_name"))
            gm = orm_ops.get_dimension_means(grade=student.get("grade"))
            if sum(cm.values()) > 0:
                ref["班级均值"] = cm
            if sum(gm.values()) > 0:
                ref["年级均值"] = gm
            st.plotly_chart(chart_builder.build_radar(student, ref_means=ref or None),
                            use_container_width=True)
            if st.button("AI 解读雷达图", key="ai_radar"):
                with st.spinner(SPINNER_AI):
                    txt, ok = ai.generate_chart_interpret(student, "多维能力雷达图")
                if ok:
                    _render_ai_text(txt)
                else:
                    st.warning(txt)

    with c2:
        card = _card("成长趋势", accent=THEME_PRIMARY)
        with card:
            st.plotly_chart(
                chart_builder.build_trend(student.get("student_id", "x"), composite),
                use_container_width=True)
            if st.button("AI 解读趋势图", key="ai_trend"):
                with st.spinner(SPINNER_AI):
                    txt, ok = ai.generate_chart_interpret(student, "成长趋势折线图")
                if ok:
                    _render_ai_text(txt)
                else:
                    st.warning(txt)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # 第三行：六维进度条（含等级与升级分差）+ 词云
    c3, c4 = st.columns([7, 5])
    with c3:
        card = _card("六维能力进度（含等级）", accent=THEME_PRIMARY)
        with card:
            bars = []
            for d in DIMENSION_ORDER:
                score, _ = dim_scores[d]
                lv = student.get(f"level_{d}")
                label = f"{DIMENSION_LABELS[d]}（{lv}）" if lv else DIMENSION_LABELS[d]
                bars.append(_progress_bar(label, score, DIMENSION_COLORS[d]))
            st.markdown("".join(bars), unsafe_allow_html=True)
            gap_lines = []
            for d in DIMENSION_ORDER:
                score, _ = dim_scores[d]
                g = _gap_to_next(student, f"dim:{d}", score, student.get(f"level_{d}"))
                if g is not None and g > 0:
                    gap_lines.append(
                        f"{DIMENSION_LABELS[d]}距上一等级还差 <b>{g:.1f}</b> 分"
                    )
            if gap_lines:
                st.caption("　｜　".join(gap_lines))

    with c4:
        card = _card("学生画像词云", accent=THEME_PRIMARY)
        with card:
            if tags:
                # 半身人形剪影词云：mask 轮廓内密集填词，外沿细轮廓线勾勒
                # 按学号 + 标签内容缓存 PNG，避免每次切换页签都重新渲染 matplotlib
                tags_tuple = tuple(tuple(t) for t in tags)
                img64 = _wordcloud_png(
                    str(student.get("student_id") or "x"), tags_tuple, THEME_CARD_BG
                )
                st.markdown(
                    f'<div style="display:flex;justify-content:center;'
                    f'background:{THEME_CARD_BG};">'
                    f'<img src="data:image/png;base64,{img64}" '
                    f'style="max-width:100%;"/></div>',
                    unsafe_allow_html=True,
                )
                st.caption("图例：字号越大表示该标签越突出。")
            else:
                st.info("暂无足够标签数据：完成分析流水线后将自动生成画像词云。")

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # 第二课堂成绩单（五部分计分）
    _render_second_classroom(student)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # 第四行：个性化建议 + 数据来源
    c5, c6 = st.columns([7, 5])
    with c5:
        card = _card("个性化提升建议", accent=THEME_PRIMARY)
        with card:
            advice = analysis_pipeline.get_personalized_advice(student)
            for i, a in enumerate(advice):
                st.markdown(
                    f'<div style="background:{THEME_PRIMARY_LIGHT};border-left:3px solid {THEME_PRIMARY};'
                    f'border-radius:6px;padding:10px 12px;margin-bottom:8px;color:{THEME_TEXT};'
                    f'font-size:14px;">{a}</div>',
                    unsafe_allow_html=True,
                )

    with c6:
        with st.expander("📂 数据来源说明"):
            st.markdown(
                "**当前演示数据**：团队自主生成的模拟学情数据集（240 条、68 字段、"
                "2023 / 2024 级 4 个教学班、固定随机种子可复现），不含任何校内真实学生信息；"
                "字段口径对标《2024 级本科生手册》六维 32 项二级指标。\n\n"
                "**各维度目标数据源（设计预留，尚未实现自动同步）**：\n"
                "- **学业发展**：教务 / 图书馆 — GPA、出勤、作业、学分、竞赛\n"
                "- **身心健康**：体测 / 心理量表 — 体测、校园跑、情绪、压力、睡眠\n"
                "- **社会参与**：第二课堂 / 志愿 / 社团 — 活动、志愿、任职、竞赛\n"
                "- **生活适应**：一卡通 / 门禁 — 饮食、作息、消费、离校\n"
                "- **品德素养**：学工 / 组织 — 思想政治、违纪、诚信、集体\n"
                "- **非认知能力**：手册无统一测量指标，设计取自作业提交时间戳、"
                "打卡连续性与问卷测评；当前为模拟数据，仅作趋势提示，不构成心理评估结论\n\n"
                "平台当前采用手动导入数据集方式；教务 / 一卡通 / 第二课堂接口"
                "仅作适配器预留，需在合规授权后启用（《作品书》4.5.3、5.3）。"
            )

    # 底部：AI 诊断 + PDF 导出
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    card = _card("AI 成长诊断与个人报告导出", accent=THEME_PRIMARY)
    with card:
        c7, c8 = st.columns([3, 2])
        with c7:
            if st.button("生成 AI 成长诊断报告", key="ai_diag"):
                with st.spinner(SPINNER_AI):
                    txt, ok = ai.generate_student_diagnosis(student)
                if ok:
                    st.success("AI 诊断已生成（基于脱敏数据）。")
                else:
                    _err = llm_client.get_last_error()
                    st.warning(
                        "AI 未启用或调用失败，已回退为规则式建议。"
                        + (f"\n\n**失败原因**：{_err}" if _err else "")
                    )
                st.session_state["ai_diag_text"] = txt
            if st.session_state.get("ai_diag_text"):
                _render_ai_text(st.session_state["ai_diag_text"])
        with c8:
            if st.button("生成 PDF 报告", key="pdf_btn"):
                ai_text = st.session_state.get("ai_diag_text")
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                    path = report_gen.export_student_pdf(student, advice, tf.name, ai_report=ai_text)
                with open(path, "rb") as f:
                    st.download_button("📄 下载 PDF", f.read(),
                                       file_name=f"成长画像_{student.get('student_id')}.pdf",
                                       mime="application/pdf")
                os.remove(path)


# ---------------------------------------------------------------------------
# 维度分页面
# ---------------------------------------------------------------------------
# 手册规则 -> 维度映射（A*→学业发展、P*→身心健康、S*→品德素养、EC→社会参与；
# 生活适应 / 非认知能力无手册条款）
_DIM_RULE_PREFIX = {
    "study": ("A",),
    "health": ("P",),
    "morality": ("S",),
    "social": ("EC",),
}


def _render_dim_raw_table(dim_key: str, sub_scores: dict, enriched: dict,
                          class_means: dict, grade_means: dict) -> None:
    """原始指标卡：二级指标原始值 / 得分 / 权重 / 归一化区间 / 班级与年级均值对比。"""
    card = _card("原始指标明细", accent=DIMENSION_COLORS[dim_key])
    with card:
        head = (
            f'<tr style="color:{THEME_TEXT_SUB};font-size:12px;text-align:left;">'
            f'<th style="padding:6px 8px;">二级指标</th><th style="padding:6px 8px;">原始值</th>'
            f'<th style="padding:6px 8px;">得分</th><th style="padding:6px 8px;">权重</th>'
            f'<th style="padding:6px 8px;">归一化区间</th><th style="padding:6px 8px;">班级均值</th>'
            f'<th style="padding:6px 8px;">年级均值</th><th style="padding:6px 8px;">与年级差值</th></tr>'
        )
        rows = []
        for (sub_label, w, field, lo, hi, higher) in DIMENSION_SUBSCORES[dim_key]:
            raw_v = enriched.get(field)
            cm, gm = class_means.get(field), grade_means.get(field)
            diff_html = "—"
            try:
                if raw_v is not None and gm is not None:
                    diff = float(raw_v) - float(gm)
                    arrow = "↑" if diff > 1e-9 else ("↓" if diff < -1e-9 else "＝")
                    good = (diff > 1e-9) if higher else (diff < -1e-9)
                    dcolor = RISK_SOFT_TEXT["正常"] if good else (
                        THEME_TEXT_SUB if arrow == "＝" else RISK_SOFT_TEXT["关注"])
                    diff_html = (f'<span style="color:{dcolor};font-weight:600;">'
                                 f'{arrow} {abs(diff):.2f}</span>')
            except (TypeError, ValueError):
                pass
            reverse_mark = ('<span style="font-size:11px;color:' + THEME_TEXT_SUB +
                            ';">（反向计分）</span>') if not higher else ""
            rows.append(
                f'<tr style="border-top:1px solid {THEME_PRIMARY_LIGHT};font-size:13px;">'
                f'<td style="padding:6px 8px;color:{THEME_TEXT};">{sub_label}{reverse_mark}</td>'
                f'<td style="padding:6px 8px;color:{THEME_TEXT};">{_fmt_raw(sub_label, raw_v)}</td>'
                f'<td style="padding:6px 8px;color:{DIMENSION_COLORS[dim_key]};font-weight:600;">'
                f'{sub_scores.get(sub_label, 0.0):.1f} 分</td>'
                f'<td style="padding:6px 8px;color:{THEME_TEXT_SUB};">{w * 100:.0f}%</td>'
                f'<td style="padding:6px 8px;color:{THEME_TEXT_SUB};">[{lo:g}, {hi:g}]</td>'
                f'<td style="padding:6px 8px;color:{THEME_TEXT_SUB};">{_fmt_raw(sub_label, cm)}</td>'
                f'<td style="padding:6px 8px;color:{THEME_TEXT_SUB};">{_fmt_raw(sub_label, gm)}</td>'
                f'<td style="padding:6px 8px;">{diff_html}</td></tr>'
            )
        st.markdown(
            f'<table style="width:100%;border-collapse:collapse;">{head}{"".join(rows)}</table>',
            unsafe_allow_html=True,
        )


def _render_dim_compare(student: dict, dim_key: str, dim_score: float) -> None:
    """同类型对比条：本人 vs 同成长类型均值 vs 年级均值。"""
    card = _card("同类型对比", accent=DIMENSION_COLORS[dim_key])
    with card:
        ctype = student.get("cluster_type")
        grade = student.get("grade")
        refs = []
        if ctype:
            tmeans = orm_ops.get_dimension_means_by_cluster_type(ctype, int(grade or 0))
            if tmeans.get(dim_key):
                refs.append((f"同类型均值（{ctype}）", tmeans[dim_key]))
        try:
            gmeans = orm_ops.get_dimension_means(grade=int(grade))
            if gmeans.get(dim_key):
                refs.append(("年级均值", gmeans[dim_key]))
        except (TypeError, ValueError):
            pass
        if not refs:
            st.info("暂无对比数据：完成分析流水线后自动生成。")
            return
        st.plotly_chart(
            chart_builder.build_compare_bar("本人", dim_score, refs,
                                            DIMENSION_COLORS[dim_key]),
            use_container_width=True,
        )


def _render_dim_tags(dim_key: str, sub_scores: dict) -> None:
    """行为标签：本维度正向（≥75）/ 负向（≤50）标签胶囊。"""
    from config_setting import TAG_DICT

    card = _card("行为标签", accent=DIMENSION_COLORS[dim_key])
    with card:
        good_tags, bad_tags = [], []
        for (sub_label, _w, _f, _lo, _hi, _h) in DIMENSION_SUBSCORES[dim_key]:
            key = (dim_key, sub_label)
            if key not in TAG_DICT:
                continue
            good, bad, _mid = TAG_DICT[key]
            s = sub_scores.get(sub_label, 0.0)
            if s >= 75:
                good_tags.append(good)
            elif s <= 50:
                bad_tags.append(bad)
        pills = []
        for t in good_tags[:2]:
            pills.append(
                f'<span style="display:inline-block;background:{RISK_SOFT_BG["正常"]};'
                f'border:1px solid {RISK_SOFT_BORDER["正常"]};border-radius:14px;'
                f'padding:3px 14px;margin:0 6px 6px 0;color:{RISK_SOFT_TEXT["正常"]};'
                f'font-size:13px;font-weight:500;">{t}</span>')
        for t in bad_tags[:2]:
            pills.append(
                f'<span style="display:inline-block;background:{RISK_SOFT_BG["关注"]};'
                f'border:1px solid {RISK_SOFT_BORDER["关注"]};border-radius:14px;'
                f'padding:3px 14px;margin:0 6px 6px 0;color:{RISK_SOFT_TEXT["关注"]};'
                f'font-size:13px;font-weight:500;">{t}</span>')
        if pills:
            st.markdown("".join(pills), unsafe_allow_html=True)
        else:
            st.caption("本维度各二级指标均处于中间区间，暂无突出标签。")


def _render_dim_handbook(student: dict, dim_key: str) -> None:
    """本维度手册提醒：筛出属于本维度的规则提醒。"""
    card = _card("本维度手册条款提醒", accent=DIMENSION_COLORS[dim_key])
    with card:
        prefixes = _DIM_RULE_PREFIX.get(dim_key)
        if not prefixes:
            st.caption("本维度暂无手册条款提醒。")
            return
        items = [it for it in _load_json(student.get("handbook_reminders"), [])
                 if any(str(it.get("rule", "")).startswith(p) for p in prefixes)]
        if not items:
            st.caption("本维度暂无手册条款提醒。")
            return
        for it in items:
            rule = it.get("rule", "")
            rule_name = HANDBOOK_RULES.get(rule, {}).get("name", "")
            if it.get("type") == "鼓励":
                bg, bd, tx, icon = (RISK_SOFT_BG["正常"], RISK_SOFT_BORDER["正常"],
                                    RISK_SOFT_TEXT["正常"], "👍")
            else:
                bg, bd, tx, icon = (RISK_SOFT_BG["关注"], RISK_SOFT_BORDER["关注"],
                                    RISK_SOFT_TEXT["关注"], "🔔")
            st.markdown(
                f'<div style="background:{bg};border-left:3px solid {bd};'
                f'border-radius:6px;padding:10px 12px;margin-bottom:8px;">'
                f'<span style="display:inline-block;background:{THEME_PRIMARY_LIGHT};'
                f'border:1px solid {THEME_PRIMARY};border-radius:4px;padding:0 8px;'
                f'color:{THEME_PRIMARY_DARK};font-size:12px;font-weight:700;">{rule}</span>'
                f'<span style="font-size:12px;color:{THEME_TEXT_SUB};margin-left:6px;">'
                f'{icon} {rule_name}</span>'
                f'<div style="font-size:14px;color:{tx};margin-top:4px;">{it.get("text", "")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_dim_tree(student: dict, dim_key: str, dim_score: float, dim_level: str,
                     sub_scores: dict, enriched: dict,
                     class_means: dict, grade_means: dict) -> None:
    """维度分析树（层级展开）：节点连线树，横向 / 纵向可切换，分支可折叠。"""
    label = DIMENSION_LABELS[dim_key]
    dim_color = DIMENSION_COLORS[dim_key]
    card = _card(f"{label} · 维度分析树（层级展开）", accent=dim_color)
    with card:
        t1, t2 = st.columns([1, 2])
        with t1:
            orientation = st.radio(
                "布局方向", ["横向", "纵向"], horizontal=True,
                key=f"tree_ori_{dim_key}",
                help="横向：根在左、叶在右；纵向：根在上、叶在下",
            )
        sub_labels = [s[0] for s in DIMENSION_SUBSCORES[dim_key]]
        with t2:
            collapsed = st.multiselect(
                "折叠分支（选中即收起该指标明细，移除即展开）",
                sub_labels, default=[], key=f"tree_collapse_{dim_key}",
            )

        meta = _get_meta(student, f"dim:{dim_key}")
        boundaries = (meta or {}).get("boundaries") or []
        lv_labels = (meta or {}).get("labels") or []

        children = []
        for (sub_label, w, field, lo, hi, higher) in DIMENSION_SUBSCORES[dim_key]:
            s = float(sub_scores.get(sub_label, 0.0))
            s_level = _level_of_score(s, boundaries, lv_labels)
            raw_v = enriched.get(field)
            cm, gm = class_means.get(field), grade_means.get(field)
            reverse = "（反向计分）" if not higher else ""
            hover = (
                f"{sub_label}　{s:.1f} 分 · {s_level}<br>"
                f"权重 {w * 100:.0f}%<br>"
                f"归一化区间 [{lo:g}, {hi:g}]{reverse}<br>"
                f"原始值 {_fmt_raw(sub_label, raw_v)}<br>"
                f"年级均值 {_fmt_raw(sub_label, gm)}"
            )
            if sub_label in collapsed:
                leaf = None
                line2 = f"{s:.1f} 分 · {s_level}（明细已折叠）"
            else:
                leaf = (f"{_fmt_raw(sub_label, raw_v)}{reverse}"
                        f"｜班级均值 {_fmt_raw(sub_label, cm)}｜权重 {w * 100:.0f}%")
                line2 = f"{s:.1f} 分 · {s_level}"
            children.append({
                "title": sub_label, "line2": line2, "score": s,
                "leaf": leaf, "hover": hover, "leaf_hover": hover,
            })
        root = {
            "title": f"{label}水平",
            "line2": f"{dim_score:.1f} 分 · {dim_level}",
            "hover": f"{label}维度得分 {dim_score:.1f} 分，等级 {dim_level}"
                     f"<br>综合权重 {DIMENSION_WEIGHTS[dim_key] * 100:.0f}%"
                     f"<br>{len(children)} 项二级指标",
        }
        fig = chart_builder.build_dimension_tree(
            root, children, dim_color,
            orientation="v" if orientation == "纵向" else "h",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("色阶：深色 = 得分高（本维度同色系）；悬停节点查看权重 / 归一化区间 / "
                   "原始值 / 年级均值；图可拖拽平移与缩放。")


def _render_dim_ai(student: dict, dim_key: str, dim_score: float, dim_level: str,
                   dim_scores: dict) -> None:
    """AI 维度诊断按钮：复用 F3 图表解读链路（事实卡片 + 引用 + 后校验）。

    AI 未启用 / 调用失败时回退规则式建议，不报错。
    """
    label = DIMENSION_LABELS[dim_key]
    card = _card(f"AI {label}维度诊断", accent=DIMENSION_COLORS[dim_key])
    with card:
        st.caption("基于脱敏事实卡片生成，结论附 [Fxx] 引用与证据清单。")
        if st.button("生成 AI 维度诊断", key=f"ai_dim_{dim_key}"):
            with st.spinner(SPINNER_AI):
                txt, ok = ai.generate_chart_interpret(
                    student, f"{label}维度画像（得分 {dim_score:.1f}，等级 {dim_level}）")
            st.session_state[f"ai_dim_text_{dim_key}"] = (txt, ok)
        saved = st.session_state.get(f"ai_dim_text_{dim_key}")
        if saved:
            txt, ok = saved
            if ok:
                _render_ai_text(txt)
            else:
                # 规则式回退：给出该维度的规则建议
                advice = scoring.get_personalized_advice(
                    dim_scores, student.get("risk_level", "正常"))
                dim_advice = [a for a in advice if a.startswith(label)]
                fallback = "\n\n".join(dim_advice) if dim_advice else (
                    f"{label}维度当前 {dim_score:.1f} 分 · {dim_level}，"
                    f"暂无专项短板建议，保持当前节奏即可。")
                st.info("AI 未启用，已回退为规则式解读：\n\n" + fallback)


def _render_dimension(student: dict, dim_key: str) -> None:
    """单个维度分页面。"""
    label = DIMENSION_LABELS[dim_key]
    dim_color = DIMENSION_COLORS[dim_key]
    # enrich 注入手册派生字段，保证二级得分与落库口径一致
    enriched = hb_rules.enrich_record(dict(student))
    dim_scores = scoring.compute_dimension_scores(enriched)
    dim_score, sub_scores = dim_scores[dim_key]
    try:
        class_means_raw, grade_means_raw = _raw_field_means(
            str(student.get("class_name") or ""), int(student.get("grade")))
    except (TypeError, ValueError):
        class_means_raw, grade_means_raw = {}, {}

    # 标题
    st.markdown(
        f'<div style="font-size:20px;font-weight:700;color:{THEME_TEXT};margin-bottom:4px;">'
        f'{label} 维度画像</div>'
        f'<div style="width:40px;height:4px;background:{dim_color};border-radius:2px;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # 第一行：得分卡 + 二级指标柱状图
    c1, c2 = st.columns([3, 7])
    with c1:
        card = _card("维度得分", accent=dim_color)
        with card:
            dim_level = student.get(f"level_{dim_key}") or "—"
            st.markdown(
                f'<div style="text-align:center;margin:20px 0;">'
                f'<div style="font-size:42px;font-weight:700;color:{dim_color};">{dim_score:.1f}</div>'
                f'<div style="font-size:11px;color:{THEME_TEXT_SUB};opacity:0.75;">满分 100</div>'
                f'<div style="font-size:13px;color:{THEME_TEXT_SUB};margin-top:2px;">等级：{dim_level}</div></div>',
                unsafe_allow_html=True,
            )
            # 状态标签统一为等级口径（待改进/合格/良好/优秀），不再使用分数段措辞
            st.markdown(_status_pill(label, dim_level if dim_level != "—" else ""),
                        unsafe_allow_html=True)
            gap = _gap_to_next(student, f"dim:{dim_key}", dim_score,
                               student.get(f"level_{dim_key}"))
            if gap is not None and gap > 0:
                st.caption(f"📈 距上一等级还差 **{gap:.1f}** 分（等级边界由本届同年级分布决定）")
            weak = min(sub_scores.items(), key=lambda x: x[1])
            st.markdown(
                f'<div style="margin-top:14px;font-size:13px;color:{THEME_TEXT_SUB};">'
                f'相对最弱项：<span style="color:{THEME_TEXT};font-weight:500;">{weak[0]}</span> '
                f'（{weak[1]:.1f} 分）</div>',
                unsafe_allow_html=True,
            )

    with c2:
        card = _card(f"{label} · 二级指标得分", accent=dim_color)
        with card:
            st.plotly_chart(chart_builder.build_sub_bar(sub_scores, label, dim_color=dim_color),
                            use_container_width=True)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # 第二行：原始指标明细 + 同类型对比 / 行为标签
    c3, c4 = st.columns([7, 5])
    with c3:
        _render_dim_raw_table(dim_key, sub_scores, enriched,
                              class_means_raw, grade_means_raw)
    with c4:
        _render_dim_compare(student, dim_key, dim_score)
        _render_dim_tags(dim_key, sub_scores)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # 第三行：本维度手册条款提醒
    _render_dim_handbook(student, dim_key)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # 第四行：维度分析树（层级展开，节点连线树）
    _render_dim_tree(student, dim_key, dim_score,
                     student.get(f"level_{dim_key}") or "—",
                     sub_scores, enriched, class_means_raw, grade_means_raw)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # 第五行：AI 维度诊断
    _render_dim_ai(student, dim_key, dim_score,
                   student.get(f"level_{dim_key}") or "—", dim_scores)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # 第六行：维度建议
    card = _card("维度提升建议", accent=dim_color)
    with card:
        advice = scoring.get_personalized_advice(dim_scores, student.get("risk_level", "正常"))
        dim_advice = [a for a in advice if a.startswith(label)]
        if dim_advice:
            for a in dim_advice:
                st.markdown(
                    f'<div style="background:{THEME_PRIMARY_LIGHT};border-left:3px solid {dim_color};'
                    f'border-radius:6px;padding:10px 12px;margin-bottom:8px;color:{THEME_TEXT};'
                    f'font-size:14px;">{a}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info(f"{label}维度暂无专项短板建议，保持当前状态即可。")


# ---------------------------------------------------------------------------
# AI 助手
# ---------------------------------------------------------------------------
def _render_ai(student: dict) -> None:
    """AI 对话助手。"""
    _ai_status_banner()
    st.markdown(
        f'<div style="font-size:18px;font-weight:700;color:{THEME_TEXT};margin-bottom:4px;">'
        f'AI 对话助手</div>',
        unsafe_allow_html=True,
    )
    # 隐私徽章（常驻）：脱敏网关说明
    st.markdown(
        f'<div style="background:{THEME_PRIMARY_LIGHT};border:1px solid {THEME_CARD_BORDER};'
        f'border-left:4px solid {THEME_PRIMARY};border-radius:8px;padding:9px 14px;'
        f'font-size:13px;color:{THEME_TEXT};margin-bottom:8px;">'
        f'🔒 数据经脱敏网关处理后发送，原始数据不出本机。</div>',
        unsafe_allow_html=True,
    )
    st.caption("仅基于你的脱敏数据作答，不泄露个人信息。")
    if "stu_chat" not in st.session_state:
        st.session_state.stu_chat = []
    for msg in st.session_state.stu_chat:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_ai_text(msg["content"])
            else:
                st.write(msg["content"])
    q = st.chat_input("向 AI 提问，例如：我哪方面最该加强？")
    if q:
        st.session_state.stu_chat.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)
        ctx = _build_safe_context(student)
        with st.chat_message("assistant"):
            with st.spinner(SPINNER_AI):
                txt, ok = ai.chat_answer(q, ctx)
            _render_ai_text(txt)
        st.session_state.stu_chat.append({"role": "assistant", "content": txt})


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def render_student_page(student: dict) -> None:
    """渲染学生端个人成长画像（顶部页签切换）。"""
    _inject_theme_css()

    # 顶部页签导航：综合总览 + 六个维度 + AI 助手（恢复 tabs 形态）
    tabs = st.tabs(ALL_TABS)

    with tabs[0]:
        _render_overview(student)

    for i, dim in enumerate(DIMENSION_ORDER):
        with tabs[i + 1]:
            _render_dimension(student, dim)

    with tabs[len(ALL_TABS) - 1]:
        _render_ai(student)
