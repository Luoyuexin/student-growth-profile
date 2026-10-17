"""report_module 成长报告 PDF 导出模块。

基于 reportlab 生成学生个人画像报告 / 教师批量报告。
图表由 Matplotlib（Agg，无外部浏览器依赖）渲染为 PNG 后嵌入，
保证与网页端一致且动态生成，并彻底去除对 Chrome / Playwright / kaleido 的依赖。

中文字体：Matplotlib 与 ReportLab 均显式注册系统中文字体，
避免 PDF 正文中文显示为方框 / 乱码。
"""

from __future__ import annotations

import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from report_module import charts as chart_builder
from config_setting import (
    DIMENSION_ORDER, DIMENSION_LABELS,
    THEME_PRIMARY, THEME_PRIMARY_DARK, THEME_TEXT_SUB,
    RISK_SOFT_TEXT, RISK_CHART_COLORS, CHART_NEUTRAL,
    TABLE_ROW_ALT,
)

# 注册中文字体（若有），避免 PDF 中文乱码 / 方框
_CJK = chart_builder.get_cjk_font_path()
if _CJK:
    try:
        font_manager.fontManager.addfont(_CJK)
        _FONT_NAME = font_manager.FontProperties(fname=_CJK).get_name()
        plt.rcParams["font.family"] = _FONT_NAME
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# ReportLab 中文字体注册
# ---------------------------------------------------------------------------
def _register_reportlab_cjk() -> str:
    """注册可用的 CJK TrueType 字体，返回 ReportLab 字体名；失败回退 Helvetica。"""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        ("MSYH", r"C:\Windows\Fonts\msyh.ttc", 0),       # 微软雅黑（TTC 首子字体）
        ("MSYHBD", r"C:\Windows\Fonts\msyhbd.ttc", 0),
        ("SimHei", r"C:\Windows\Fonts\simhei.ttf", None),
        ("SimSun", r"C:\Windows\Fonts\simsun.ttc", 0),
        ("NotoSansCJK", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", None),
        ("PingFang", "/System/Library/Fonts/PingFang.ttc", 0),
    ]
    for name, path, subidx in candidates:
        if not os.path.exists(path):
            continue
        try:
            if path.lower().endswith(".ttc"):
                pdfmetrics.registerFont(TTFont(name, path, subfontIndex=subidx or 0))
            else:
                pdfmetrics.registerFont(TTFont(name, path))
            return name
        except Exception:
            continue
    return "Helvetica"


RL_FONT = _register_reportlab_cjk()


def _buf_from_fig(fig) -> io.BytesIO:
    """将 matplotlib Figure 导出为 PNG 字节流并关闭图形。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _pdf_radar(student: dict) -> io.BytesIO:
    """六维能力雷达图（matplotlib polar）。"""
    labels = [DIMENSION_LABELS[d] for d in DIMENSION_ORDER]
    vals = [float(student.get(f"score_{d}", 0) or 0) for d in DIMENSION_ORDER]
    angles = [n / float(len(labels)) * 2 * 3.14159265 for n in range(len(labels))]
    vals_c = vals + [vals[0]]
    angles_c = angles + [angles[0]]

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles_c, vals_c, linewidth=2, color=THEME_PRIMARY)
    ax.fill(angles_c, vals_c, alpha=0.25, color=THEME_PRIMARY)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_ylim(0, 100)
    ax.set_title("六维能力雷达", fontsize=12)
    return _buf_from_fig(fig)


def _pdf_distribution(distribution: dict) -> io.BytesIO:
    """风险等级分布柱状图（matplotlib，替代 kaleido）。"""
    labels_order = ["正常", "关注", "高危"]
    labels = [l for l in labels_order if l in distribution] + \
             [l for l in distribution if l not in labels_order]
    values = [distribution.get(l, 0) for l in labels]
    colors_map = dict(RISK_CHART_COLORS)
    colors_map["正常"] = THEME_PRIMARY

    fig = plt.figure(figsize=(6, 3.2))
    plt.bar(labels, values, color=[colors_map.get(l, CHART_NEUTRAL) for l in labels])
    plt.title("风险等级分布", fontsize=12)
    plt.ylabel("人数")
    plt.tight_layout()
    return _buf_from_fig(fig)


def _pdf_trend(student_id: str, current_score: float) -> io.BytesIO:
    """成长趋势折线图（matplotlib，走势与网页端同源同种子）。"""
    scores = chart_builder.trend_scores(student_id, current_score, semesters=6)
    labels = [f"第{i}学期" for i in range(1, 7)]

    fig = plt.figure(figsize=(6, 3.2))
    plt.plot(labels, scores, marker="o", color=THEME_PRIMARY, linewidth=2)
    plt.ylim(0, 100)
    plt.title("成长趋势", fontsize=12)
    plt.tight_layout()
    return _buf_from_fig(fig)


def export_student_pdf(student: dict, advice: list[str], save_path: str, ai_report: str = None) -> str:
    """导出单个学生画像报告 PDF。"""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    doc = SimpleDocTemplate(
        save_path, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleX", parent=styles["Title"], fontSize=18,
        fontName=RL_FONT, leading=24,
    )
    h_style = ParagraphStyle(
        "H", parent=styles["Heading2"],
        textColor=colors.HexColor(THEME_PRIMARY_DARK),
        fontName=RL_FONT,
    )
    body = ParagraphStyle("BodyCJK", parent=styles["BodyText"],
                          fontName=RL_FONT, fontSize=10.5, leading=17)

    story = []
    story.append(Paragraph("学生个人成长画像报告", title_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"学号（脱敏）：{student.get('student_id')}　|　姓名（脱敏）：{student.get('name')}　|　"
        f"班级：{student.get('class_name')}", body))
    story.append(Spacer(1, 10))

    risk = student.get("risk_level", "正常")
    level_color = RISK_SOFT_TEXT.get(risk, THEME_TEXT_SUB)
    card_data = [
        ["综合评分", f"{float(student.get('composite_score', 0) or 0):.1f}"],
        ["综合等级", student.get("level_overall") or "—"],
        ["成长类型", student.get("cluster_type") or "—"],
        ["风险等级", risk],
    ]
    card = Table(card_data, colWidths=[5 * cm, 5 * cm])
    card.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), RL_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("FONTNAME", (0, 0), (0, -1), RL_FONT),
        ("TEXTCOLOR", (1, 3), (1, 3), colors.HexColor(level_color)),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(card)
    story.append(Spacer(1, 12))

    # 六维得分概览表（含维度等级）
    story.append(Paragraph("一、六维得分与等级", h_style))
    dim_rows = [["维度", "得分", "等级"]]
    for d in DIMENSION_ORDER:
        dim_rows.append([
            DIMENSION_LABELS[d],
            f"{float(student.get(f'score_{d}', 0) or 0):.1f}",
            str(student.get(f"level_{d}") or "—"),
        ])
    dim_table = Table(dim_rows, colWidths=[7 * cm, 3 * cm, 3 * cm], repeatRows=1)
    dim_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), RL_FONT),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(THEME_PRIMARY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(TABLE_ROW_ALT)]),
    ]))
    story.append(dim_table)
    story.append(Spacer(1, 10))

    # 雷达图
    story.append(Paragraph("二、六维能力雷达", h_style))
    story.append(Image(_pdf_radar(student), width=11 * cm, height=11 * cm))
    story.append(Spacer(1, 8))

    # 趋势图
    story.append(Paragraph("三、成长趋势", h_style))
    story.append(Image(_pdf_trend(student.get("student_id", "x"),
                                  float(student.get("composite_score", 0) or 0)),
                        width=12 * cm, height=6.4 * cm))
    story.append(Spacer(1, 8))

    # 个性化建议
    story.append(Paragraph("四、个性化提升建议", h_style))
    for i, a in enumerate(advice, 1):
        story.append(Paragraph(f"{i}. {a}", body))

    # AI 成长诊断（可选，含证据引用）
    if ai_report:
        story.append(Spacer(1, 8))
        story.append(Paragraph("五、AI 成长诊断报告（结论附证据编号）", h_style))
        # **加粗** 标记 ReportLab 支持 <b>，把 markdown 残留符号做轻量清洗
        ai_text = ai_report.replace("**", "")
        story.append(Paragraph(ai_text.replace("\n", "<br/>"), body))

    doc.build(story)
    return save_path


def export_teacher_batch_pdf(
    students: list[dict], distribution: dict, save_path: str
) -> str:
    """导出教师端班级批量报告 PDF（含分布图 + 学生清单）。"""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    doc = SimpleDocTemplate(save_path, pagesize=A4)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCJK", parent=styles["Title"], fontName=RL_FONT, leading=24,
    )
    h_style = ParagraphStyle(
        "HCJK", parent=styles["Heading2"], fontName=RL_FONT,
    )
    story = [Paragraph("班级整体画像报告", title_style), Spacer(1, 10)]

    story.append(Paragraph("风险等级分布", h_style))
    story.append(Image(_pdf_distribution(distribution), width=12 * cm, height=6.4 * cm))
    story.append(Spacer(1, 10))

    story.append(Paragraph("学生清单（脱敏）", h_style))
    header = ["学号", "姓名", "综合评分", "综合等级", "成长类型", "风险等级"]
    rows = [header]
    for s in students:
        rows.append([
            str(s.get("student_id")), str(s.get("name")),
            f"{float(s.get('composite_score', 0) or 0):.1f}",
            str(s.get("level_overall") or "—"),
            str(s.get("cluster_type") or "—"),
            str(s.get("risk_level")),
        ])
    table = Table(
        rows,
        colWidths=[3.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 3.4 * cm, 2.2 * cm],
        repeatRows=1,
    )
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), RL_FONT),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(THEME_PRIMARY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(TABLE_ROW_ALT)]),
    ]))
    story.append(table)
    doc.build(story)
    return save_path
