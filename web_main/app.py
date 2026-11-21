"""web_main 应用入口。

Streamlit 双端（学生 / 教师）统一登录与路由入口。

运行方式（在项目根目录）：
    streamlit run web_main/app.py

首次运行会自动播种模拟数据并跑一遍分析流水线；
也可手动执行：python data_source/generate_mock_data.py
"""

from __future__ import annotations

import os
import sys

import streamlit as st

# 确保项目根目录在 sys.path，便于以 `streamlit run` 方式直接运行
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from local_database import db_init, orm_ops
from analysis_core import pipeline as analysis_pipeline
import config_setting as C
from config_setting import (
    SPINNER_DATA,
    build_theme_css
)

from web_main.student_app import render_student_page
from web_main.teacher_app import render_teacher_page

DEMO_STUDENT_PWD = "stu123456"
DEMO_TEACHER_PWD = "teacher123"

PLATFORM_NAME = "StuProfile 大学生成长智能画像平台"
PLATFORM_TAGLINE = "六维成长画像 · 数据分布驱动定级 · 可溯源 AI 分析"


def _ensure_data() -> None:
    """保证数据库已初始化且含有学生数据；必要时自动播种并分析。

    兼容旧版数据库：若缺少新版字段（如 score_mental / major），则强制重建。
    """
    db_init.init_database()
    # schema 兼容性：缺少手册硬规则字段或新版等级 / 类型字段时重建数据库
    if (
        not db_init.column_exists("students", "campus_run_records")
        or not db_init.column_exists("students", "cluster_type")
    ):
        with st.spinner("检测到数据库版本过旧，正在重建并重新生成数据..."):
            import data_source.generate_mock_data as gen
            gen.main(to_db=True)
        st.success("数据库已升级并生成最新模拟数据。")
        return

    ids = orm_ops.get_all_student_ids()
    if not ids:
        with st.spinner("首次运行：正在生成模拟学情数据并执行分析流水线..."):
            import data_source.generate_mock_data as gen
            gen.main(to_db=True)
        st.success("模拟数据已生成并完成分析。")
        return

    # 有学生但缺少分析结果时，自动补跑流水线
    sample = orm_ops.get_raw_student(ids[0])
    if (
        sample is None
        or sample.get("composite_score") is None
        or sample.get("handbook_hits") is None
    ):
        with st.spinner(f"检测到分析结果缺失，{SPINNER_DATA}"):
            analysis_pipeline.run_pipeline(save=True)
        st.success("分析完成。")


def _do_login(role: str, uid: str, pwd: str) -> tuple[bool, dict | None]:
    if role == "student":
        user = orm_ops.authenticate_student(uid, pwd)
        return (user is not None, user)
    else:
        user = orm_ops.authenticate_teacher(uid, pwd)
        return (user is not None, user)


def _inject_login_css() -> None:
    """登录页样式：统一主题 + 卡片式通道页签（金色渐变选中态）。"""
    st.markdown(build_theme_css(), unsafe_allow_html=True)
    st.markdown(f"""
    <style>
    /* 通道页签放大成卡片（登录页专属，覆盖统一主题的窄条样式） */
    .stTabs [role="tablist"], .stTabs [data-baseweb="tab-list"] {{
        gap: 12px;
        border-bottom: none !important;
        margin-bottom: 16px;
    }}
    .stTabs [data-testid="stTab"], .stTabs [data-baseweb="tab"] {{
        height: 58px;
        padding: 0 30px;
        border: 1px solid {C.THEME_CARD_BORDER} !important;
        border-bottom: 1px solid {C.THEME_CARD_BORDER} !important;
        border-radius: 14px !important;
        background: {C.THEME_CARD_BG} !important;
        color: {C.THEME_TEXT_SUB} !important;
        font-size: 16px !important;
        box-shadow: 0 1px 4px {C.LOGIN_TAB_SHADOW};
    }}
    .stTabs [data-testid="stTab"]:hover, .stTabs [data-baseweb="tab"]:hover {{
        border-color: {C.THEME_PRIMARY} !important;
        background: {C.THEME_PRIMARY_LIGHT} !important;
        color: {C.THEME_PRIMARY_DARK} !important;
    }}
    .stTabs [data-testid="stTab"][aria-selected="true"], .stTabs [aria-selected="true"] {{
        border: 1px solid {C.THEME_PRIMARY} !important;
        background: {C.THEME_GRADIENT} !important;
        color: {C.PRIMARY_BTN_TEXT} !important;
        box-shadow: 0 4px 14px {C.LOGIN_TAB_ACTIVE_SHADOW};
    }}
    .stTabs [data-testid="stTab"][aria-selected="true"] p {{ color: {C.PRIMARY_BTN_TEXT} !important; }}
    </style>
    """, unsafe_allow_html=True)


def _login_form(role: str, form_key: str) -> None:
    """渲染单通道登录卡片。"""
    card = st.container(border=True)
    with card:
        if role == "student":
            st.markdown(f"**🎓 学生通道**　查看个人六维成长画像、等级与 AI 诊断")
            account_label = "学号"
            err_hint = "学号或密码错误。演示账号：202310100001 / stu123456"
        else:
            st.markdown(f"**🧑‍🏫 教师 / 辅导员通道**　班级画像、风险预警与配置管理")
            account_label = "工号"
            err_hint = "工号或密码错误。演示账号：T2023001 / teacher123"
        with st.form(form_key):
            uid = st.text_input(account_label)
            pwd = st.text_input("密码", type="password")
            submitted = st.form_submit_button("登录", width="stretch")
            if submitted:
                ok, user = _do_login(role, uid, pwd)
                if ok:
                    st.session_state.role = role
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error(err_hint)


def _tech_badges() -> None:
    """登录页底部技术徽章行。"""
    badges = ["Python", "Streamlit", "SQLite", "scikit-learn", "Plotly"]
    pills = "".join(
        f'<span style="display:inline-block;background:{C.THEME_PRIMARY_LIGHT};'
        f'border:1px solid {C.THEME_CARD_BORDER};border-radius:14px;padding:3px 14px;'
        f'margin:0 4px;color:{C.THEME_PRIMARY_DARK};font-size:13px;font-weight:500;">{b}</span>'
        for b in badges
    )
    st.markdown(
        f'<div style="text-align:center;margin-top:28px;">{pills}</div>'
        f'<div style="text-align:center;margin-top:10px;font-size:13px;'
        f'color:{C.THEME_TEXT_SUB};">🖥️ 单机可运行 · 原始数据不出域</div>'
        f'<div style="text-align:center;margin-top:6px;font-size:12px;'
        f'color:{C.THEME_TEXT_SUB};">演示数据为团队自建模拟学情数据集，'
        f'不含任何真实学生信息</div>',
        unsafe_allow_html=True,
    )


def _skin_popover() -> None:
    """右上角皮肤切换器：按钮由全站主题 CSS 着色，两套皮肤都协调。"""
    with st.popover("🎨 皮肤", width="stretch"):
        st.caption("界面主题")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("雾蓝", width="stretch", key="skin_blue_btn"):
                st.session_state.skin = C.SKIN_MIST_BLUE
                C.set_skin(C.SKIN_MIST_BLUE)
                st.rerun()
        with c2:
            if st.button("拿破仑黄", width="stretch", key="skin_gold_btn"):
                st.session_state.skin = C.SKIN_GOLD
                C.set_skin(C.SKIN_GOLD)
                st.rerun()
        st.caption("当前：" + C.SKIN_LABELS[C.get_skin()])


def _logout_button() -> bool:
    """页面级退出按钮 + 皮肤切换。"""
    # 顶部留出 Streamlit 固定工具栏高度
    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
    _, c_skin, c_out = st.columns([10, 1.4, 1])
    with c_skin:
        _skin_popover()
    with c_out:
        return st.button("退出登录", key="global_logout", width="stretch")


def main() -> None:
    st.set_page_config(
        page_title=PLATFORM_NAME, layout="wide",
        initial_sidebar_state="collapsed",
    )
    # 皮肤：每次脚本执行开头同步当前会话的选择
    if "skin" not in st.session_state:
        st.session_state.skin = C.DEFAULT_SKIN
    C.set_skin(st.session_state.skin)
    _ensure_data()

    # 会话状态
    if "role" not in st.session_state:
        st.session_state.role = None
        st.session_state.user = None

    # 已登录：渲染对应端（退出入口放在主区域，侧栏保持不出现）
    role = st.session_state.role
    if role in ("student", "teacher"):
        if _logout_button():
            st.session_state.role = None
            st.session_state.user = None
            st.session_state.pop("risk_banner_dismissed", None)
            st.rerun()
        if role == "student":
            render_student_page(st.session_state.user)
        else:
            render_teacher_page(st.session_state.user)
        return

    # 未登录：品牌化登录页
    _inject_login_css()
    _, c_skin = st.columns([12, 1.4])
    with c_skin:
        _skin_popover()
    st.markdown(
        f'<div style="text-align:center;margin-top:24px;">'
        f'<div style="font-size:34px;font-weight:700;color:{C.THEME_TEXT};">'
        f'📊 {PLATFORM_NAME}</div>'
        f'<div style="font-size:15px;color:{C.THEME_TEXT_SUB};margin-top:10px;">'
        f'{PLATFORM_TAGLINE}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    # 学生 / 教师通道：卡片式切换页签
    tab_stu, tab_tea = st.tabs(["🎓 学生通道", "🧑‍🏫 教师 / 辅导员通道"])
    with tab_stu:
        _login_form("student", "login_form_student")
    with tab_tea:
        _login_form("teacher", "login_form_teacher")

    # 演示账号提示
    with st.expander("演示账号（点击查看）"):
        st.markdown(
            "- **学生端**：`202310100001` + `stu123456`\n"
            "- **教师端**：工号 `T2023001` 或 `T2023002`，+ `teacher123`"
        )
    st.caption("提示：「重新执行分析流水线」入口已移至教师端「配置管理」页。")

    # 底部技术徽章行
    _tech_badges()


if __name__ == "__main__":
    main()
