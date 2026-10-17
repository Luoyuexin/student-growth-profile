"""web_main 教师 / 辅导员端页面。

包含：班级整体画像（含 F2 AI 研判 + 偏弱模块饼图）、学生列表、风险预警
（含 D6 实时弹窗 + F4 AI 帮扶话术）、学院宏观研判（D3：年级 + 专业两级聚合）、
AI 对话助手（F5）、配置管理。

界面风格：灰绿商务风，白卡片 + 圆角 + 浅阴影，与示例图一致。
"""

from __future__ import annotations

import json
import os
import re
import tempfile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from local_database import orm_ops
from report_module import charts as chart_builder
from report_module import report_gen
from analysis_core.risk_level import compute_risk_score
from analysis_core import scoring
from analysis_core import pipeline as analysis_pipeline
from analysis_core import handbook_rules as hb_rules
from ai_insight import services as ai, llm_client
from web_main.student_app import _progress_bar, _render_ai_text

from config_setting import (
    DIMENSION_ORDER, DIMENSION_LABELS, DIMENSION_SUBSCORES, DIMENSION_COLORS,
    DIMENSION_WEIGHTS,
    THEME_PRIMARY, THEME_PRIMARY_DARK, THEME_PRIMARY_LIGHT,
    THEME_BG, THEME_CARD_BG, THEME_CARD_BORDER, THEME_TEXT, THEME_TEXT_SUB,
    RISK_SOFT_BG, RISK_SOFT_BORDER, RISK_SOFT_TEXT,
    NEUTRAL_BG, NEUTRAL_BORDER,
    HANDBOOK_RULES,
    SPINNER_AI, SPINNER_DATA,
)


DIM_LABELS = [DIMENSION_LABELS[d] for d in DIMENSION_ORDER]
DIM_COLS = [f"score_{d}" for d in DIMENSION_ORDER]
DIM_COLOR_LIST = [DIMENSION_COLORS[d] for d in DIMENSION_ORDER]


def _inject_theme_css() -> None:
    """注入全局 CSS：灰绿商务风主题。"""
    st.markdown(f"""
    <style>
    .stApp {{
        background-color: {THEME_BG} !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        border-bottom: 1px solid {THEME_CARD_BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {THEME_CARD_BG};
        border: 1px solid {THEME_CARD_BORDER};
        border-radius: 8px 8px 0 0;
        color: {THEME_TEXT_SUB};
        padding: 10px 18px;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {THEME_PRIMARY_LIGHT};
        color: {THEME_TEXT};
        border-bottom: 2px solid {THEME_PRIMARY};
    }}
    .stButton>button {{
        border: 1px solid {THEME_PRIMARY} !important;
        color: {THEME_PRIMARY_DARK} !important;
        background-color: {THEME_CARD_BG} !important;
        border-radius: 8px !important;
        font-weight: 500;
    }}
    .stButton>button:hover {{
        background-color: {THEME_PRIMARY_LIGHT} !important;
        color: {THEME_TEXT} !important;
    }}
    .stDownloadButton>button {{
        border: 1px solid {THEME_PRIMARY} !important;
        color: {THEME_CARD_BG} !important;
        background-color: {THEME_PRIMARY} !important;
        border-radius: 8px !important;
        font-weight: 500;
    }}
    .stDownloadButton>button:hover {{
        background-color: {THEME_PRIMARY_DARK} !important;
        border-color: {THEME_PRIMARY_DARK} !important;
    }}
    .stExpander {{
        border: 1px solid {THEME_CARD_BORDER};
        border-radius: 12px;
        background-color: {THEME_CARD_BG};
    }}
    </style>
    """, unsafe_allow_html=True)


def _card(title: str | None = None, accent: str = THEME_PRIMARY) -> st.delta_generator.DeltaGenerator:
    """返回一个带边框圆角的容器。"""
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
    sub_html = (
        f'<div style="font-size:12px;color:{THEME_TEXT_SUB};margin-top:4px;">{subtitle}</div>'
        if subtitle else ""
    )
    return (
        f'<div style="background:{THEME_CARD_BG};border:1px solid {THEME_CARD_BORDER};'
        f'border-radius:12px;padding:16px;text-align:center;">'
        f'<div style="font-size:12px;color:{THEME_TEXT_SUB};margin-bottom:4px;">{title}</div>'
        f'<div style="font-size:26px;font-weight:700;color:{accent};">{value}</div>'
        f'{sub_html}</div>'
    )


def _risk_pill(level: str) -> str:
    """生成柔和风险标签。"""
    bg = RISK_SOFT_BG.get(level, NEUTRAL_BG)
    border = RISK_SOFT_BORDER.get(level, NEUTRAL_BORDER)
    text = RISK_SOFT_TEXT.get(level, THEME_TEXT_SUB)
    return (
        f'<span style="display:inline-block;background:{bg};border:1px solid {border};'
        f'border-radius:12px;padding:2px 10px;color:{text};font-size:12px;font-weight:500;">'
        f'{level}</span>'
    )


def _recompute_risk(classes: list[str]) -> None:
    """按当前风险阈值重算管辖班级学生的风险等级。"""
    students = orm_ops.list_students_by_classes(classes)
    for s in students:
        # 注意：列表记录的学号已脱敏，必须用内部主键 _raw_student_id 查原始记录，
        # 否则按 ****0001 之类的脱敏学号永远查不到人（旧版此处静默失效）。
        sid = s.get("_raw_student_id") or s["student_id"]
        raw = orm_ops.get_raw_student(sid)
        if not raw:
            continue
        score, level, _ = compute_risk_score(raw)
        orm_ops.upsert_student({
            "student_id": raw["student_id"],
            "risk_level": level,
            "risk_score": score,
        })


def _parse_classes(scope: str) -> list[str]:
    return [c.strip() for c in (scope or "").split(",") if c.strip()]


def _ai_status_banner() -> None:
    if not llm_client.is_enabled():
        st.caption(
            "🤖 AI 功能未启用：请在 `config_setting/llm_config.yaml` 中填入云端 API Key "
            "后即可使用（数据经脱敏网关处理后发送）。"
        )


# ---------------------------------------------------------------------------
# 主渲染
# ---------------------------------------------------------------------------
def render_teacher_page(teacher: dict) -> None:
    """渲染教师端班级管理与预警页面。"""
    _inject_theme_css()
    classes = _parse_classes(teacher.get("managed_classes", ""))

    # 顶部标题与管辖班级
    h1, h2 = st.columns([3, 2])
    with h1:
        st.markdown(
            f'<div style="font-size:22px;font-weight:700;color:{THEME_TEXT};">'
            f'教师端 · 班级管理与预警</div>',
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            f'<div style="background:{THEME_CARD_BG};border:1px solid {THEME_CARD_BORDER};'
            f'border-radius:10px;padding:10px 14px;text-align:right;">'
            f'<div style="font-size:12px;color:{THEME_TEXT_SUB};">管辖班级</div>'
            f'<div style="font-size:13px;color:{THEME_TEXT};font-weight:500;">'
            f'{"、".join(classes)}</div></div>',
            unsafe_allow_html=True,
        )

    # ---- D6 高危学生实时弹窗预警（4px 左侧深红条，支持手动关闭）----
    high_students = orm_ops.get_risk_students(classes, ["高危"])
    if high_students and not st.session_state.get("risk_banner_dismissed"):
        names = "、".join(f"{s['name']}({s['student_id']})" for s in high_students[:5])
        more = f" 等共 {len(high_students)} 人" if len(high_students) > 5 else ""
        b1, b2 = st.columns([11, 1])
        with b1:
            st.markdown(
                f'<div style="background:{RISK_SOFT_BG["高危"]};'
                f'border:1px solid {RISK_SOFT_BORDER["高危"]};'
                f'border-left:4px solid {RISK_SOFT_TEXT["高危"]};'
                f'border-radius:10px;padding:12px 16px;color:{RISK_SOFT_TEXT["高危"]};'
                f'font-size:14px;font-weight:500;margin-bottom:12px;">'
                f'⚠️ 实时预警：当前有 {len(high_students)} 名高危学生，请重点关注：{names}{more}</div>',
                unsafe_allow_html=True,
            )
        with b2:
            if st.button("✕ 关闭", key="dismiss_risk_banner"):
                st.session_state.risk_banner_dismissed = True
                st.rerun()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "班级整体画像", "学生列表", "风险预警", "学院宏观研判", "年级分析树", "AI 对话助手", "配置管理"])

    # ===================== Tab1 班级整体画像 =====================
    with tab1:
        _ai_status_banner()
        distribution = orm_ops.get_class_distribution(classes)
        total = sum(distribution.values())

        # 顶部统计卡
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(_metric_card("管辖学生", str(total)), unsafe_allow_html=True)
        with c2:
            st.markdown(_metric_card("高危学生", str(distribution.get("高危", 0)),
                                     accent=RISK_SOFT_TEXT["高危"]), unsafe_allow_html=True)
        with c3:
            st.markdown(_metric_card("关注学生", str(distribution.get("关注", 0)),
                                     accent=RISK_SOFT_TEXT["关注"]), unsafe_allow_html=True)
        with c4:
            avg_score = 0.0
            recs = orm_ops.get_records_for_analysis(classes)
            if recs:
                avg_score = pd.DataFrame(recs)["composite_score"].mean()
            st.markdown(_metric_card("班级均分", f"{avg_score:.1f}"), unsafe_allow_html=True)

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        cc1, cc2 = st.columns(2)
        with cc1:
            card = _card("班级风险等级分布")
            with card:
                chart_type = st.radio("图表类型", ["饼图", "柱状图"], horizontal=True, key="dist_type")
                fig = chart_builder.build_distribution(
                    distribution, chart_type="bar" if chart_type == "柱状图" else "pie")
                st.plotly_chart(fig, use_container_width=True)

        with cc2:
            recs = orm_ops.get_records_for_analysis(classes)
            if recs:
                rdf = pd.DataFrame(recs)
                means = rdf[DIM_COLS].mean()
                card = _card("班级各维度均值（定位偏弱模块）")
                with card:
                    pie = go.Figure(go.Pie(
                        labels=DIM_LABELS,
                        values=[float(means[c]) for c in DIM_COLS],
                        hole=0.55,
                        marker_colors=DIM_COLOR_LIST,
                        textinfo="label+percent",
                        insidetextorientation="horizontal",
                    ))
                    pie.update_layout(
                        height=340,
                        margin=dict(l=20, r=20, t=30, b=20),
                        paper_bgcolor=THEME_CARD_BG,
                        font=dict(color=THEME_TEXT),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
                    )
                    st.plotly_chart(pie, use_container_width=True)

        if recs:
            card = _card("🤖 AI 班级研判摘要")
            with card:
                if st.button("生成 AI 班级研判摘要", key="ai_class"):
                    agg = _build_class_agg(rdf, distribution)
                    with st.spinner(SPINNER_AI):
                        txt, ok = ai.generate_class_summary(agg)
                    if ok:
                        _render_ai_text(txt)
                    else:
                        st.warning(txt)

        card = _card("批量报告导出")
        with card:
            students = orm_ops.list_students_by_classes(classes)
            if st.button("导出班级批量 PDF 报告"):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                    path = report_gen.export_teacher_batch_pdf(students, distribution, tf.name)
                with open(path, "rb") as f:
                    st.download_button("📄 下载批量 PDF", f.read(),
                                       file_name="班级画像报告.pdf", mime="application/pdf")
                os.remove(path)

    # ===================== Tab2 学生列表 =====================
    with tab2:
        card = _card("学生列表")
        with card:
            # 隐私徽章（常驻）：脱敏口径说明
            st.markdown(
                f'<div style="display:inline-block;background:{THEME_PRIMARY_LIGHT};'
                f'border:1px solid {THEME_CARD_BORDER};border-radius:14px;padding:3px 14px;'
                f'color:{THEME_PRIMARY_DARK};font-size:12px;font-weight:500;margin-bottom:8px;">'
                f'🔒 已脱敏：姓名仅留首字 · 学号仅留末四位</div>',
                unsafe_allow_html=True,
            )
            keyword = st.text_input("搜索姓名 / 学号关键字", key="stu_search")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                sort_by = st.selectbox("排序字段",
                                       ["composite_score", "risk_score", "gpa", "name"],
                                       key="sort_by")
            with col_b:
                risk_filter = st.selectbox("风险等级筛选",
                                           ["全部", "正常", "关注", "高危"],
                                           key="risk_filter")
            with col_c:
                desc = st.checkbox("降序", value=True, key="sort_desc")

            students = orm_ops.list_students_by_classes(
                classes, keyword=keyword, sort_by=sort_by, desc=desc)
            if risk_filter != "全部":
                students = [s for s in students if s.get("risk_level") == risk_filter]
            st.write(f"共 {len(students)} 名学生")

            if not students:
                st.info("没有符合条件的学生：请调整搜索关键字或风险等级筛选；"
                        "若列表长期为空，请先在「配置管理」页重新执行分析流水线。")
            else:
                df = pd.DataFrame(students)[
                    ["student_id", "name", "class_name", "composite_score",
                     "level_overall", "cluster_type", "risk_level", "cluster_label"]]
                df.columns = ["学号(脱敏)", "姓名(脱敏)", "班级", "综合评分",
                              "综合等级", "成长类型", "风险等级", "簇号"]
                st.dataframe(df, use_container_width=True, height=320)

        card = _card("个体画像穿透查看")
        with card:
            if not students:
                st.info("暂无可穿透查看的学生：请先在上表调整筛选条件。")
            else:
                options = [f"{s['name']} / {s['student_id']}" for s in students]
                selected_idx = st.selectbox(
                    "选择学生（脱敏）", range(len(options)),
                    format_func=lambda i: options[i], key="drill_select")
                if selected_idx is not None:
                    # 展示用脱敏学号，查询用内部主键（脱敏字段无法直接作为主键检索）
                    sid = students[selected_idx].get("_raw_student_id") or students[selected_idx]["student_id"]
                    raw = orm_ops.get_raw_student(sid)
                    if raw:
                        disp = dict(raw)
                        disp["name"] = orm_ops.mask_name(raw["name"])
                        disp["student_id"] = orm_ops.mask_id(raw["student_id"])
                        c1, c2 = st.columns(2)
                        with c1:
                            rcard = _card("六维能力雷达")
                            with rcard:
                                st.plotly_chart(chart_builder.build_radar(disp),
                                                use_container_width=True)
                        with c2:
                            tcard = _card("成长趋势")
                            with tcard:
                                st.plotly_chart(
                                    chart_builder.build_trend(
                                        sid, float(disp.get("composite_score", 0) or 0)),
                                    use_container_width=True)
                        st.write(
                            f"综合评分：{(disp.get('composite_score') or 0):.1f}　"
                            f"综合等级：{disp.get('level_overall') or '—'}　"
                            f"成长类型：{disp.get('cluster_type') or '—'}　"
                            f"风险等级：{disp.get('risk_level') or '未分析'}"
                        )
                        # 六维迷你进度条（与学生端总览同款组件；enrich 后口径与落库一致）
                        dim_scores = scoring.compute_dimension_scores(
                            hb_rules.enrich_record(dict(raw)))
                        bars = []
                        for d in DIMENSION_ORDER:
                            score, _ = dim_scores[d]
                            lv = disp.get(f"level_{d}")
                            label = f"{DIMENSION_LABELS[d]}（{lv}）" if lv else DIMENSION_LABELS[d]
                            bars.append(_progress_bar(label, score, DIMENSION_COLORS[d]))
                        st.markdown("".join(bars), unsafe_allow_html=True)

                        # 个体画像报告：复用 F1 AI 诊断链路（事实卡片→强制引用→后校验），
                        # 按钮 / 会话 key 一律用原始学号，避免脱敏 ID 冲突；仅页面展示，不做 PDF
                        if st.button("生成该生画像分析报告", key=f"ai_stu_report_btn_{sid}"):
                            with st.spinner(SPINNER_AI):
                                txt, ok = ai.generate_student_diagnosis(raw)
                            st.session_state[f"ai_stu_report_{sid}"] = (txt, ok)
                        saved_report = st.session_state.get(f"ai_stu_report_{sid}")
                        if saved_report:
                            txt, ok = saved_report
                            if ok:
                                st.success("AI 报告已生成（基于脱敏数据）。")
                                _render_ai_text(txt)
                            else:
                                st.warning(txt)

    # ===================== Tab3 风险预警 =====================
    with tab3:
        # ---- 手册规则提醒（依据本科生手册硬性规定，只列需要跟进的项）----
        card = _card("手册规则提醒")
        with card:
            stats = orm_ops.get_handbook_reminder_stats(classes)
            reminded = orm_ops.get_reminded_students(classes)
            if not stats:
                st.caption("当前无手册规则提醒。")
            else:
                st.caption(
                    f"管辖班级共 {len(reminded)} 名学生存在需要提醒 / 督促的事项"
                    f"（已达标项不罗列）；按规则汇总如下，可筛选查看明细。"
                )
                cols = st.columns(4)
                for i, (rid, cnt) in enumerate(stats.items()):
                    with cols[i % 4]:
                        st.markdown(
                            _metric_card(
                                f"{rid} · {HANDBOOK_RULES.get(rid, {}).get('name', '')}",
                                str(cnt), "人", accent=RISK_SOFT_TEXT["关注"]),
                            unsafe_allow_html=True,
                        )
                opts = ["全部"] + list(stats.keys())
                pick = st.selectbox(
                    "按规则筛选", opts, key="hb_rule_filter",
                    format_func=lambda x: "全部规则" if x == "全部"
                    else f"{x} · {HANDBOOK_RULES.get(x, {}).get('name', '')}",
                )
                rows = orm_ops.get_reminded_students(classes, None if pick == "全部" else [pick])
                st.write(f"符合条件：{len(rows)} 人")
                for r in rows:
                    items = json.loads(r.get("handbook_reminders") or "[]")
                    hits = json.loads(r.get("handbook_hits") or "[]")
                    with st.expander(
                        f"{r['name']} ({r['student_id']}) · {r['class_name']} · "
                        f"命中 {'、'.join(hits)} · {len(items)} 条提醒",
                        expanded=False,
                    ):
                        for it in items:
                            st.markdown(f"- **[{it['rule']}]** {it['text']}")
                        st.caption("以上为客观提示，具体认定与处理以学院规定为准。")

        card = _card("高危 / 关注学生实时预警")
        with card:
            risk_students = orm_ops.get_risk_students(classes, ["高危", "关注"])
            if not risk_students:
                st.markdown(
                    f'<div style="background:{RISK_SOFT_BG["正常"]};border:1px solid {RISK_SOFT_BORDER["正常"]};'
                    f'border-radius:10px;padding:12px 16px;color:{RISK_SOFT_TEXT["正常"]};'
                    f'font-size:14px;">✅ 当前无高危 / 关注学生。</div>',
                    unsafe_allow_html=True,
                )
            for s in risk_students:
                level = s.get('risk_level', '未分析')
                raw_sid = s.get("_raw_student_id") or s["student_id"]
                with st.expander(
                        f"{s['name']} ({s['student_id']}) · {_risk_pill(level)} · 风险分 {(s.get('risk_score') or 0):.1f}",
                        expanded=False):
                    st.write(f"班级：{s['class_name']}　综合评分：{(s.get('composite_score') or 0):.1f}")
                    raw = orm_ops.get_raw_student(raw_sid)
                    # 组件 key 用内部主键，避免脱敏学号末四位相同导致键冲突
                    if raw and st.button("生成 AI 帮扶话术", key=f"ai_counsel_{raw_sid}"):
                        with st.spinner(SPINNER_AI):
                            txt, ok = ai.generate_counsel_tips(raw)
                        if ok:
                            _render_ai_text(txt)
                        else:
                            st.warning(txt)
                    notes = orm_ops.get_teacher_notes(raw_sid)
                    for n in notes:
                        st.caption(f"[{n['mark_level']}] {n['teacher_id']} @ {n['update_time']}：{n['note']}")
                    with st.form(key=f"note_{raw_sid}"):
                        mark = st.selectbox("标记等级",
                                            ["高危", "关注", "正常"], key=f"mk_{raw_sid}")
                        note = st.text_area("备注", key=f"nt_{raw_sid}")
                        if st.form_submit_button("保存备注 / 标记"):
                            orm_ops.save_teacher_note(
                                raw_sid, teacher["teacher_id"], note, mark)
                            st.success("已保存。")
                            st.rerun()

    # ===================== Tab4 学院宏观研判（D3） =====================
    with tab4:
        _ai_status_banner()
        recs = orm_ops.get_records_for_analysis(classes)
        if not recs:
            st.info("暂无画像数据：请先在「配置管理」页点击「重新执行分析流水线」生成数据。")
        else:
            rdf = pd.DataFrame(recs)
            grades = sorted([int(g) for g in rdf["grade"].dropna().unique()])
            sel_grade = st.selectbox("选择年级", grades, key="macro_grade")
            gdf = rdf[rdf["grade"] == sel_grade]
            majors = sorted([str(m) for m in gdf["major"].dropna().unique()])

            # 顶部统计
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(_metric_card(f"{sel_grade} 级学生数", str(len(gdf))), unsafe_allow_html=True)
            with c2:
                st.markdown(_metric_card(f"覆盖专业数", str(len(majors))), unsafe_allow_html=True)

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

            cc1, cc2 = st.columns(2)
            with cc1:
                card = _card("各专业维度均值对比")
                with card:
                    means = gdf.groupby("major")[DIM_COLS].mean().round(1)
                    bar = go.Figure()
                    for lab, col, color in zip(DIM_LABELS, DIM_COLS, DIM_COLOR_LIST):
                        bar.add_trace(go.Bar(
                            x=means.index.tolist(), y=means[col].tolist(), name=lab,
                            marker_color=color,
                        ))
                    bar.update_layout(
                        barmode="group", height=360,
                        yaxis=dict(title="维度均值", range=[0, 100], gridcolor=THEME_PRIMARY_LIGHT),
                        xaxis=dict(gridcolor=THEME_PRIMARY_LIGHT),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
                        paper_bgcolor=THEME_CARD_BG, plot_bgcolor=THEME_CARD_BG,
                        font=dict(color=THEME_TEXT),
                    )
                    st.plotly_chart(bar, use_container_width=True)

            with cc2:
                card = _card("各专业风险等级分布")
                with card:
                    risk_by = gdf.groupby(["major", "risk_level"]).size().unstack(fill_value=0)
                    for lv in ["正常", "关注", "高危"]:
                        if lv not in risk_by.columns:
                            risk_by[lv] = 0
                    risk_by = risk_by[["正常", "关注", "高危"]]
                    stack = go.Figure()
                    for lv in ["正常", "关注", "高危"]:
                        stack.add_trace(go.Bar(
                            x=risk_by.index.tolist(), y=risk_by[lv].tolist(),
                            name=lv, marker_color=RISK_SOFT_TEXT[lv],
                            text=risk_by[lv].tolist(), textposition="inside",
                        ))
                    stack.update_layout(
                        barmode="stack", height=360,
                        yaxis=dict(title="人数", gridcolor=THEME_PRIMARY_LIGHT),
                        xaxis=dict(gridcolor=THEME_PRIMARY_LIGHT),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
                        paper_bgcolor=THEME_CARD_BG, plot_bgcolor=THEME_CARD_BG,
                        font=dict(color=THEME_TEXT),
                    )
                    st.plotly_chart(stack, use_container_width=True)

    # ===================== Tab5 年级分析树（六维 Treemap） =====================
    with tab5:
        _render_grade_trees(classes)

    # ===================== Tab6 AI 对话助手（F5） =====================
    with tab6:
        _ai_status_banner()
        card = _card("AI 对话助手（班级视角）")
        with card:
            st.caption("仅基于管辖班级的脱敏聚合数据作答，不泄露个人信息。")
            if "tea_chat" not in st.session_state:
                st.session_state.tea_chat = []
            for msg in st.session_state.tea_chat:
                with st.chat_message(msg["role"]):
                    if msg["role"] == "assistant":
                        _render_ai_text(msg["content"])
                    else:
                        st.write(msg["content"])
            q = st.chat_input("向 AI 提问，例如：本班哪个模块最需加强？")
            if q:
                st.session_state.tea_chat.append({"role": "user", "content": q})
                with st.chat_message("user"):
                    st.write(q)
                ctx = _build_class_agg(pd.DataFrame(recs), orm_ops.get_class_distribution(classes)) \
                    if recs else {}
                with st.chat_message("assistant"):
                    with st.spinner(SPINNER_AI):
                        txt, ok = ai.chat_answer(q, ctx)
                    _render_ai_text(txt)
                st.session_state.tea_chat.append({"role": "assistant", "content": txt})

    # ===================== Tab7 配置管理 =====================
    with tab7:
        card = _card("预警阈值与聚类参数配置")
        with card:
            st.caption("以下调整为当前进程内存配置，重新运行后恢复默认值；"
                       "如需持久化请修改 config_setting/__init__.py 或下发 JSON 配置。")

            from config_setting import risk_threshold, get_grade_cluster_config, GRADE_CLUSTER_CONFIG

            rt = risk_threshold
            rt.fail_count = st.number_input("挂科风险阈值（门）", 1, 10, int(rt.fail_count))
            rt.continuous_absence = st.number_input("连续缺勤风险阈值（天）", 1, 30,
                                                    int(rt.continuous_absence))
            rt.night_not_home = st.number_input("夜间未归风险阈值（次）", 1, 20,
                                                int(rt.night_not_home))
            rt.consumption_drop_ratio = st.slider("消费骤降风险阈值（比例）", 0.0, 1.0,
                                                  float(rt.consumption_drop_ratio), 0.05)
            rt.risk_score_threshold = st.number_input("关注风险分阈值", 0.0, 5.0,
                                                      float(rt.risk_score_threshold), 0.5)
            rt.high_risk_score_threshold = st.number_input("高危风险分阈值", 0.0, 5.0,
                                                           float(rt.high_risk_score_threshold), 0.5)

            st.divider()
            st.write("各年级聚类模型（最近一次分析流水线实参）：")
            st.caption("多维分型 K 由肘部法则(0.4)+轮廓系数(0.6)联合自动确定；"
                       "分维度等级优先 K=4，边界为相邻质心中点。")
            meta_rows = orm_ops.list_cluster_meta()
            if not meta_rows:
                st.caption("尚未执行分析流水线。")
            else:
                grades = sorted({m["grade"] for m in meta_rows})
                for grade in grades:
                    with st.expander(f"{grade} 级聚类模型"):
                        for m in [x for x in meta_rows if x["grade"] == grade]:
                            if m["scope"] == "multi":
                                st.write(f"- 多维分型：K={m['k']}，成长类型：{'、'.join(m.get('labels') or [])}")
                                # 定 K 过程曲线图：X=K(2~8)，左轴 SSE、右轴轮廓系数、
                                # 合成得分曲线高亮最优 K（肘部 0.4 + 轮廓 0.6 加权自动确定）
                                if m.get("details"):
                                    kfig = chart_builder.build_k_selection_curve(
                                        m["details"], grade=grade)
                                    if kfig is not None:
                                        st.plotly_chart(kfig, use_container_width=True)
                                else:
                                    st.caption("定 K 过程明细尚未落库：重新执行一次分析流水线后，"
                                               "此处将展示 SSE / 轮廓系数 / 合成得分曲线。")
                            elif m["scope"] == "composite":
                                st.write(f"- 综合分级：K={m['k']}，等级 {'/'.join(m.get('labels') or [])}"
                                         f"，升级边界 {m.get('boundaries')}")
                            else:
                                dim = m["scope"].split(":", 1)[-1]
                                st.write(f"- {DIMENSION_LABELS.get(dim, dim)}："
                                         f"K={m['k']}，升级边界 {m.get('boundaries')}")

            with st.expander("年级聚类静态默认配置（config_setting）"):
                for grade, cfg in GRADE_CLUSTER_CONFIG.items():
                    st.write(f"- 年级 {grade}：K={cfg.k}（自动调优={cfg.auto_tune_k}，"
                             f"搜索范围 {cfg.k_search_range}）")

            if st.button("应用阈值并重算风险等级"):
                _recompute_risk(classes)
                st.success("风险等级已按新阈值重算。")
                st.rerun()

            st.divider()
            st.write("数据更新后，可在此重新执行完整分析流水线（清洗 → 评分 → 聚类 → 风险）：")
            if st.button("🔄 重新执行分析流水线", key="rerun_pipeline"):
                with st.spinner(SPINNER_DATA):
                    analysis_pipeline.run_pipeline(save=True)
                st.success("分析完成，聚类模型与画像结果已刷新。")
                st.rerun()


# ---------------------------------------------------------------------------
# 年级分析树（节点连线层级树，替代原 Treemap）
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def _grade_tree_means(classes_key: tuple) -> dict:
    """管辖班级全体学生各维度二级指标均值（enrich 后口径，本机聚合）。"""
    raw = orm_ops.get_raw_students_by_classes(list(classes_key))
    agg: dict = {d: {} for d in DIMENSION_ORDER}
    for d in DIMENSION_ORDER:
        for (label, _w, _f, _lo, _hi, _hb) in DIMENSION_SUBSCORES[d]:
            agg[d][label] = []
    for r in raw:
        ds = scoring.compute_dimension_scores(hb_rules.enrich_record(dict(r)))
        for d in DIMENSION_ORDER:
            for label, s in ds[d][1].items():
                agg[d][label].append(s)
    return {
        d: {label: (sum(v) / len(v) if v else 0.0) for label, v in agg[d].items()}
        for d in DIMENSION_ORDER
    }


def _render_grade_tree_one(dim_key: str, means: dict, orientation: str,
                           chart_key: str) -> None:
    """渲染单个维度的聚合分析树（根=维度均值，L1=二级指标均值，L2=均值+权重）。"""
    dim_color = DIMENSION_COLORS[dim_key]
    spec = DIMENSION_SUBSCORES[dim_key]
    sub_means = means[dim_key]
    total_w = sum(s[1] for s in spec) or 1.0
    dim_mean = sum(sub_means.get(s[0], 0.0) * s[1] for s in spec) / total_w

    collapsed = st.multiselect(
        "折叠分支", [s[0] for s in spec], default=[], key=f"gtree_col_{dim_key}",
        label_visibility="collapsed",
        placeholder="折叠分支（选中即收起该指标明细）",
    )
    children = []
    for (sub_label, w, _f, lo, hi, higher) in spec:
        ms = float(sub_means.get(sub_label, 0.0))
        reverse = "（反向计分）" if not higher else ""
        children.append({
            "title": sub_label,
            "line2": f"均值 {ms:.1f}" + ("（明细已折叠）" if sub_label in collapsed else ""),
            "score": ms,
            "leaf": None if sub_label in collapsed else f"均值 {ms:.1f}｜权重 {w * 100:.0f}%",
            "hover": f"{sub_label}　班级均值 {ms:.1f} 分<br>权重 {w * 100:.0f}%<br>"
                     f"归一化区间 [{lo:g}, {hi:g}]{reverse}",
        })
    root = {
        "title": f"{DIMENSION_LABELS[dim_key]}水平",
        "line2": f"班级均值 {dim_mean:.1f} 分",
        "hover": f"{DIMENSION_LABELS[dim_key]}　班级均值 {dim_mean:.1f} 分<br>"
                 f"综合权重 {DIMENSION_WEIGHTS[dim_key] * 100:.0f}%<br>{len(children)} 项二级指标",
    }
    fig = chart_builder.build_dimension_tree(root, children, dim_color,
                                             orientation=orientation)
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def _render_grade_trees(classes: list[str]) -> None:
    """年级 / 班级整体分析树：六个维度的节点连线层级树。"""
    card = _card("年级 / 班级整体分析树（六维 · 层级展开）")
    with card:
        st.caption("节点-连线层级树：根节点为维度均值，二级节点为指标均值，叶子为均值与权重；"
                   "深色 = 得分高（各维度同色系），悬停节点查看权重与归一化区间。")
        raw = orm_ops.get_raw_students_by_classes(classes)
        if not raw:
            st.info("暂无画像数据：请先在「配置管理」页点击「重新执行分析流水线」生成数据。")
            return

        means = _grade_tree_means(tuple(sorted(classes)))

        t1, t2, t3 = st.columns([2, 1, 1])
        with t1:
            sel_dim = st.selectbox("选择维度", DIM_LABELS, key="tree_dim")
        with t2:
            orientation = st.radio("布局方向", ["横向", "纵向"], horizontal=True,
                                   key="gtree_ori")
        with t3:
            show_all = st.checkbox("同时显示全部六维", value=False, key="tree_all")
        dims_to_show = DIMENSION_ORDER if show_all else [DIMENSION_ORDER[DIM_LABELS.index(sel_dim)]]
        ori = "v" if orientation == "纵向" else "h"
        # 六棵小树纵向堆叠，各自独立图表互不挤压
        for d in dims_to_show:
            _render_grade_tree_one(d, means, ori, chart_key=f"gtree_{d}")


def _build_class_agg(rdf: pd.DataFrame, distribution: dict) -> dict:
    """构造脱敏聚合统计（供 AI 研判 / 对话使用，无 PII）。"""
    if rdf is None or rdf.empty:
        return {"student_count": 0, "dimension_means": {}, "risk_distribution": distribution}
    means = rdf[DIM_COLS].mean().round(1)
    return {
        "student_count": int(len(rdf)),
        "dimension_means": {lab: float(means[c]) for lab, c in zip(DIM_LABELS, DIM_COLS)},
        "risk_distribution": distribution,
    }
