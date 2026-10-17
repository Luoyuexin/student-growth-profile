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
from config_setting import (
    THEME_PRIMARY, THEME_PRIMARY_DARK, THEME_PRIMARY_LIGHT,
    THEME_CARD_BG, THEME_CARD_BORDER, THEME_TEXT, THEME_TEXT_SUB,
    SPINNER_DATA,
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
    """登录页品牌化样式：卡片式通道页签 + 技术徽章。"""
    st.markdown(f"""
    <style>
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {THEME_CARD_BG};
        border: 1px solid {THEME_CARD_BORDER};
        border-radius: 12px;
        color: {THEME_TEXT_SUB};
        padding: 14px 28px;
        font-size: 16px;
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {THEME_PRIMARY_LIGHT};
        color: {THEME_TEXT};
        border: 1px solid {THEME_PRIMARY};
        border-bottom: 3px solid {THEME_PRIMARY};
    }}
    </style>
    """, unsafe_allow_html=True)


def _login_form(role: str, form_key: str) -> None:
    """渲染单通道登录卡片。"""
    card = st.container(border=True)
    with card:
        if role == "student":
            st.markdown(f"**🎓 学生通道**　查看个人六维成长画像、等级与 AI 诊断")
            account_label = "学号"
            err_hint = "学号或密码错误。演示账号密码：学号 + stu123456"
        else:
            st.markdown(f"**🧑‍🏫 教师 / 辅导员通道**　班级画像、风险预警与配置管理")
            account_label = "工号"
            err_hint = "工号或密码错误。演示账号：T2023001 / teacher123"
        with st.form(form_key):
            uid = st.text_input(account_label)
            pwd = st.text_input("密码", type="password")
            submitted = st.form_submit_button("登录", use_container_width=True)
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
        f'<span style="display:inline-block;background:{THEME_PRIMARY_LIGHT};'
        f'border:1px solid {THEME_CARD_BORDER};border-radius:14px;padding:3px 14px;'
        f'margin:0 4px;color:{THEME_PRIMARY_DARK};font-size:13px;font-weight:500;">{b}</span>'
        for b in badges
    )
    st.markdown(
        f'<div style="text-align:center;margin-top:28px;">{pills}</div>'
        f'<div style="text-align:center;margin-top:10px;font-size:13px;'
        f'color:{THEME_TEXT_SUB};">🖥️ 单机可运行 · 数据不出域</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=PLATFORM_NAME, layout="wide")
    _ensure_data()

    # 会话状态
    if "role" not in st.session_state:
        st.session_state.role = None
        st.session_state.user = None

    # 已登录：渲染对应端
    if st.session_state.role == "student":
        render_student_page(st.session_state.user)
        if st.sidebar.button("退出登录"):
            st.session_state.role = None
            st.session_state.user = None
            st.rerun()
        return
    if st.session_state.role == "teacher":
        render_teacher_page(st.session_state.user)
        if st.sidebar.button("退出登录"):
            st.session_state.role = None
            st.session_state.user = None
            st.rerun()
        return

    # 未登录：品牌化登录页
    _inject_login_css()
    st.markdown(
        f'<div style="text-align:center;margin-top:24px;">'
        f'<div style="font-size:34px;font-weight:700;color:{THEME_TEXT};">'
        f'📊 {PLATFORM_NAME}</div>'
        f'<div style="font-size:15px;color:{THEME_TEXT_SUB};margin-top:10px;">'
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
            "- **学生端**：学号 + 密码 `stu123456`\n"
            "  （可用学号如 `202310100001`，详见数据库）\n"
            "- **教师端**：工号 `T2023001` 或 `T2023002`，密码 `teacher123`"
        )
    st.caption("提示：「重新执行分析流水线」入口已移至教师端「配置管理」页。")

    # 底部技术徽章行
    _tech_badges()


if __name__ == "__main__":
    main()
