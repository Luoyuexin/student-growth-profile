"""config_setting 配置模块。

集中管理：
1. 风险预警阈值（成绩、考勤、消费异常等指标）。
2. 各年级 K-means 聚类 K 值与特征权重（支持分年级独立调优）。
3. 分级权限（学生 / 教师）与脱敏规则。

所有可调参数集中在此处，便于非开发人员通过修改本文件完成运营配置，
也便于后续接入配置中心 / 教务系统下发。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# 1. 风险预警阈值配置
# ---------------------------------------------------------------------------
@dataclass
class RiskThreshold:
    """各风险指标的单指标触发阈值。

    当某个原始指标超过阈值时，该风险维度计入风险分。
    风险分累加后结合权重得到综合风险等级。
    """

    # 成绩相关
    fail_count: int = 2          # 学期挂科数 >= 该值 -> 成绩风险
    gpa_drop_ratio: float = 0.25  # 成绩环比骤降比例 >= 该值 -> 成绩骤降风险

    # 考勤相关
    continuous_absence: int = 5   # 连续缺勤天数 >= 该值 -> 考勤风险

    # 生活 / 消费相关
    night_not_home: int = 3       # 夜间未归次数 >= 该值 -> 生活风险
    consumption_drop_ratio: float = 0.4  # 消费骤减比例 >= 该值 -> 消费异常风险

    # 综合判定：单项风险权重之和 >= risk_score_threshold 进入"关注"，
    # 再高一级进入"高危"。
    risk_score_threshold: float = 1.0   # 关注线
    high_risk_score_threshold: float = 2.0  # 高危线


# ---------------------------------------------------------------------------
# 2. 聚类与特征权重配置（支持分年级独立调优）
# ---------------------------------------------------------------------------
@dataclass
class GradeClusterConfig:
    """单个年级的聚类与特征权重配置。"""

    grade: int
    k: int = 4                          # 默认聚类数（可由肘部法则/轮廓系数自动确定）
    auto_tune_k: bool = True            # 是否自动确定 K
    k_search_range: tuple = (2, 8)      # 自动确定 K 时的搜索范围
    random_state: int = 42

    # 特征权重（作品书 5.2 短期规划：权重调优链路预留，当前为经验初值）
    feature_weights: Dict[str, float] = field(default_factory=dict)


# 默认特征权重（若某年级未单独配置则使用此项）
# 说明：六维评分的实际权重以 DIMENSION_WEIGHTS / DIMENSION_SUBSCORES 为准；
#       本表为聚类特征权重调优机制预留（作品书 4.5.3 已知不足其二），
#       命名沿用旧版字段，接入聚类计算前不影响现有评分口径。
DEFAULT_FEATURE_WEIGHTS: Dict[str, float] = {
    # 学业发展
    "gpa": 1.0,
    "fail_count": 0.8,
    "library_visits": 0.4,
    "study_hours": 0.6,
    # 生活适应
    "canteen_regularity": 0.5,
    "dorm_regularity": 0.5,
    # 社会参与
    "second_class_count": 0.5,
    "volunteer_hours": 0.4,
    "competition_awards": 0.7,
    # 身心健康
    "psy_score": 1.0,
    "mood_check": 0.8,
    "stress_index": 0.7,        # 反向特征（压力越大越差）
    "psy_counsel_times": 0.3,
}

# ---------------------------------------------------------------------------
# 3. 分维度等级映射（作品书 3.4.2 表 3-1）
# ---------------------------------------------------------------------------
# 一维 K-means 质心升序后按簇数量单调映射为等级文本
LEVEL_LABELS_BY_K: Dict[int, List[str]] = {
    2: ["待改进", "优秀"],
    3: ["合格", "良好", "优秀"],
    4: ["待改进", "合格", "良好", "优秀"],
    5: ["待改进", "合格", "良好", "优秀", "卓越"],
}
# 分维度等级聚类的簇数量：优先 K=4，与手册体测四级划分对齐
DIMENSION_LEVEL_K = 4


# ---------------------------------------------------------------------------
# 4. 六维成长画像指标体系（业务方定稿）
# ---------------------------------------------------------------------------
# 维度注册表：维度键 -> (中文标签, 综合评分权重)
# 综合权重合计 1.0：学业25% / 身心20% / 社会参与12% / 生活10% / 品德15% / 非认知18%
DIMENSION_WEIGHTS: Dict[str, float] = {
    "study": 0.25,      # 学业发展
    "health": 0.20,     # 身心健康
    "social": 0.12,     # 社会参与
    "life": 0.10,       # 生活适应
    "morality": 0.15,   # 品德素养
    "non_cog": 0.18,    # 非认知能力
}

DIMENSION_LABELS: Dict[str, str] = {
    "study": "学业发展",
    "health": "身心健康",
    "social": "社会参与",
    "life": "生活适应",
    "morality": "品德素养",
    "non_cog": "非认知能力",
}

# ---------------------------------------------------------------------------
# UI 主题色（拿破仑黄 · 暖金商务风）
# 拿破仑黄做主强调色，配暖米白底 + 深褐文字，避免大面积高饱和铺色，
# 保持商务质感的同时有明确的品牌辨识度。
# ---------------------------------------------------------------------------
THEME_PRIMARY = "#5B82B8"          # 拿破仑黄：主强调色、按钮边框、图表主色
THEME_PRIMARY_DARK = "#1E4171"     # 深金棕：hover / 重要文字（黄底文字需压深才够对比）
THEME_PRIMARY_LIGHT = "#CCE1FF"    # 浅香槟：hover 背景、标签底、进度条轨道
THEME_BG = "#F8FBFE"               # 页面背景：暖米白（比卡片略深，衬出白卡片）
THEME_CARD_BG = "#FFFFFF"          # 卡片背景：纯白
THEME_CARD_BORDER = "#C9DCEF"      # 卡片边框：浅金边
THEME_TEXT = "#1C2C66"             # 主文字：暖炭黑
THEME_TEXT_SUB = "#5A6F8C"         # 次要文字：暖灰
THEME_GRID = "#E6EEF8"             # 图表网格：浅金
THEME_GRADIENT = f"linear-gradient(135deg, {THEME_PRIMARY} 0%, #4C79B8 55%, {THEME_PRIMARY_DARK} 100%)"
THEME_SHADOW = "0 2px 10px rgba(30,65,113,0.08)"

# 柔和风险色（用于背景色块 + 文字，避免 bright 红绿橙）
RISK_SOFT_BG = {"正常": "#E9F3F6", "关注": "#FFF4E8", "高危": "#FDEDF0"}
RISK_SOFT_BORDER = {"正常": "#A9CFD8", "关注": "#F0CDA0", "高危": "#EFB8C2"}
RISK_SOFT_TEXT = {"正常": "#2C6E7E", "关注": "#B26A22", "高危": "#B04A5A"}

# 等级色板（待改进 / 合格 / 良好 / 优秀 四色，用于等级卡着色与状态标签）
LEVEL_SOFT_BG = {"待改进": "#FDEDF0", "合格": "#FFF4E8", "良好": "#EDF5FA", "优秀": "#E9F3F6"}
LEVEL_SOFT_BORDER = {"待改进": "#EFB8C2", "合格": "#F0CDA0", "良好": "#B9D3E6", "优秀": "#A9CFD8"}
LEVEL_SOFT_TEXT = {"待改进": "#B04A5A", "合格": "#B26A22", "良好": "#3C7391", "优秀": "#2C6E7E"}

# 中性占位色（未知等级 / 兜底标签）
NEUTRAL_BG = "#F2F6FB"
NEUTRAL_BORDER = "#DCE6F2"

# 维度品牌色（用于雷达图、教师端分组柱图、饼图等，保持全站一致）
# 以拿破仑黄为轴心展开的协调配色：金黄→橄榄→赭橙→雅紫→灰蓝，
# 六色在暖调基底上互相可分辨，避免交通灯高饱和色。
DIMENSION_COLORS: Dict[str, str] = {
    "study": "#2F6FB5",      # 金（主色系）
    "health": "#4E9AA8",     # 青绿
    "social": "#6E86C9",     # 赭橙
    "life": "#7BA7D0",       # 棕金
    "morality": "#8E86BE",   # 雅紫
    "non_cog": "#4F7FA8",    # 灰蓝
}

# 维度展示顺序（用于雷达图 / 顶栏 / 分析树）
DIMENSION_ORDER: List[str] = ["study", "health", "social", "life", "morality", "non_cog"]

# ---------------------------------------------------------------------------
# 图表专用色板（Plotly / Matplotlib / PDF 统一从这里取色，页面代码不硬编码色值）
# ---------------------------------------------------------------------------
CHART_SECONDARY = "#4E88A8"      # 橄榄绿（辅助系列色）
CHART_ACCENT = "#7A7FC0"         # 赭橙（辅助系列色）
CHART_NEUTRAL = "#9BAEC4"        # 中性暖灰（兜底 / 均值参考）

# 风险等级的图表用色（低饱和，与柔和色块区分开：图表填充需要更高对比）
RISK_CHART_COLORS = {"正常": "#4E8C9E", "关注": "#D29A5B", "高危": "#C17078"}

# 连续得分色阶（浅=低分，深=高分）：浅香槟 → 拿破仑黄 → 深金棕
SCORE_COLORSCALE = [
    [0.0, "#EAF2FC"],   # 浅香槟
    [0.5, "#6E97C9"],   # 拿破仑黄
    [1.0, "#1E4171"],   # 深金棕
]

# 词云配色：暖金 / 赭橙 / 橄榄 / 雅紫 / 灰蓝，低饱和；按字号三档取色
WORDCLOUD_PALETTE = [
    "#1E4171", "#2F5FA0", "#4C79B8", "#6E97C9",
    "#7BA7D0", "#4E88A8", "#6E86C9", "#8E86BE",
    "#5A6F8C", "#9BAEC4",
]
WORDCLOUD_DEEP_POOL = ["#1E4171", "#203F77", "#2F5FA0", "#4A4E86"]   # 大字号
WORDCLOUD_MID_POOL = ["#4C79B8", "#4E88A8", "#6E86C9", "#7BA7D0"]    # 中字号

# 雷达图：个人能力填充 + 均值参考线配色（半透明填充叠加用）
RADAR_SELF_FILL = "rgba(91,130,184,0.26)"
RADAR_REF_COLORS = ["#9BAEC4", "#4E88A8", "#6E86C9"]
RADAR_REF_FILL_OPACITY = 0.15    # 参考面叠加不透明度（要求区间 0.12~0.18）

# 成长趋势图面积填充
TREND_FILL = "rgba(91,130,184,0.14)"

# PDF 表格隔行底色
TABLE_ROW_ALT = "#F4F8FD"

# ---------------------------------------------------------------------------
# 全站统一 CSS（拿破仑黄主题）
# 学生端 / 教师端 / 登录页共用同一份，避免各页面重复维护样式。
# 页面侧只需 st.markdown(build_theme_css(), unsafe_allow_html=True)。
# ---------------------------------------------------------------------------
def build_theme_css() -> str:
    """生成全站统一主题 CSS 字符串（拿破仑黄 · 暖金商务风）。

    选择器同时兼容 Streamlit 1.4x（data-baseweb=tab）与 1.6x（data-testid=stTab），
    避免升级后样式静默失效。
    """
    return f"""
    <style>
    /* ---------- 画布：暖米白底 + 左上柔光 + 顶部金线 ---------- */
    .stApp {{
        background-color: {THEME_BG} !important;
        color: {THEME_TEXT};
        font-family: "PingFang SC", "Microsoft YaHei", "Segoe UI", system-ui, sans-serif;
    }}
    .stApp::before {{
        content: "";
        position: fixed; inset: 0;
        pointer-events: none;
        background: radial-gradient(1000px 380px at 8% -14%,
                    rgba(91,130,184,0.16) 0%, rgba(91,130,184,0) 70%);
        z-index: 0;
    }}
    .stApp::after {{
        content: "";
        position: fixed; top: 0; left: 0; right: 0; height: 3px;
        background: {THEME_GRADIENT};
        z-index: 999999;
    }}
    header[data-testid="stHeader"] {{
        background: transparent !important;
        box-shadow: none !important;
    }}
    /* 演示场景下隐藏右上角 Deploy 按钮，界面更干净 */
    [data-testid="stAppDeployButton"] {{ display: none !important; }}
    /* 本系统不使用侧栏：强制隐藏侧栏容器、自动导航与折叠控件。
       Streamlit 前端会记住用户上次的展开状态，仅靠 initial_sidebar_state
       对已有会话不生效，故在全局主题 CSS 中兜底，保证主区域全宽。
       选择器用 [data-testid*="Sidebar"] 宽匹配，兼容不同 Streamlit 版本的 DOM 变更。 */
    section[data-testid="stSidebar"],
    [data-testid*="Sidebar"],
    [data-testid="collapsedControl"],
    [data-testid="stExpandSidebarButton"] {{ display: none !important; }}
    /* 顶部留白收一收，内容更紧凑（默认 96px 偏空） */
    [data-testid="stMainBlockContainer"], .block-container {{
        padding-top: 2.2rem !important;
        padding-bottom: 2.5rem !important;
    }}

    /* ---------- 页签：下划线高亮 + 香槟 hover ---------- */
    .stTabs [role="tablist"],
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
        border-bottom: 1px solid {THEME_CARD_BORDER};
        background: transparent !important;
    }}
    .stTabs [data-testid="stTab"],
    .stTabs [data-baseweb="tab"] {{
        height: 46px;
        padding: 0 18px;
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 10px 10px 0 0;
        color: {THEME_TEXT_SUB};
        font-weight: 600;
        transition: color .18s ease, background .18s ease, border-color .18s ease;
    }}
    .stTabs [data-testid="stTab"] p {{ font-weight: 600 !important; }}
    .stTabs [data-testid="stTab"]:hover,
    .stTabs [data-baseweb="tab"]:hover {{
        color: {THEME_PRIMARY_DARK} !important;
        background: {THEME_PRIMARY_LIGHT} !important;
    }}
    .stTabs [data-testid="stTab"][aria-selected="true"],
    .stTabs [aria-selected="true"] {{
        color: {THEME_PRIMARY_DARK} !important;
        border-bottom: 3px solid {THEME_PRIMARY} !important;
        background: linear-gradient(180deg, rgba(204,225,255,0) 0%,
                    {THEME_PRIMARY_LIGHT} 100%) !important;
    }}
    .stTabs [data-testid="stTab"][aria-selected="true"] p {{
        color: {THEME_PRIMARY_DARK} !important;
    }}
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {{ display: none; }}

    /* ---------- 卡片：圆角 + 浅金边（无边框容器不受影响） ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stVerticalBlock"] {{
        border-radius: 14px;
        border-color: {THEME_CARD_BORDER} !important;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {THEME_CARD_BG};
        box-shadow: {THEME_SHADOW};
    }}

    /* ---------- 按钮：描金次要按钮 + 金色渐变主按钮 ---------- */
    .stButton>button, .stFormSubmitButton>button {{
        border: 1px solid {THEME_PRIMARY} !important;
        color: {THEME_PRIMARY_DARK} !important;
        background-color: {THEME_CARD_BG} !important;
        border-radius: 10px !important;
        font-weight: 600;
        transition: all .18s ease;
    }}
    .stButton>button:hover, .stFormSubmitButton>button:hover {{
        background-color: {THEME_PRIMARY_LIGHT} !important;
        color: {THEME_PRIMARY_DARK} !important;
        transform: translateY(-1px);
        box-shadow: 0 3px 12px rgba(30,65,113,0.16);
    }}
    .stButton>button[kind="primary"], .stFormSubmitButton>button[kind="primary"],
    .stDownloadButton>button {{
        background: {THEME_GRADIENT} !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700;
        box-shadow: 0 3px 12px rgba(91,130,184,0.28);
    }}
    .stButton>button[kind="primary"]:hover, .stFormSubmitButton>button[kind="primary"]:hover,
    .stDownloadButton>button:hover {{
        filter: brightness(1.05);
        box-shadow: 0 5px 18px rgba(91,130,184,0.4);
    }}

    /* ---------- 输入控件：聚焦描金 ---------- */
    .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {{
        border-color: {THEME_PRIMARY} !important;
        box-shadow: 0 0 0 2px rgba(91,130,184,0.22) !important;
    }}
    div[data-testid="stChatInput"]:focus-within {{
        border-color: {THEME_PRIMARY} !important;
        box-shadow: 0 0 0 2px rgba(91,130,184,0.18) !important;
    }}

    /* ---------- 折叠面板 / 进度条 / 指标 ---------- */
    div[data-testid="stExpander"] details,
    div[data-testid="stExpanderDetails"] {{
        border: 1px solid {THEME_CARD_BORDER} !important;
        border-radius: 12px !important;
        background: {THEME_CARD_BG};
    }}
    div[data-testid="stExpander"] summary:hover {{ color: {THEME_PRIMARY_DARK}; }}
    div[data-testid="stProgress"] > div > div > div > div {{
        background-color: {THEME_PRIMARY} !important;
        background-image: {THEME_GRADIENT} !important;
    }}
    div[data-testid="stMetricValue"] {{ color: {THEME_PRIMARY_DARK} !important; }}
    div[data-testid="stMetricLabel"] {{ color: {THEME_TEXT_SUB} !important; }}
    </style>
    """


# 界面统一加载文案（spinner 口径一致）
SPINNER_AI = "AI 正在分析，请稍候…"
SPINNER_DATA = "正在处理数据，请稍候…"

# 第二课堂「五部分」↔ 六维主题色映射。
# 五部分即"德智体美劳"：德→品德素养、智→学业发展、体→身心健康、劳→社会参与；
# 其中「美学与人文素养」六维中无直接对应项，按业务口径取「非认知能力」。
SECOND_CLASSROOM_PART_DIM: Dict[str, str] = {
    "道德与思想素质": "morality",
    "学术与科技创新": "study",
    "体育与健康": "health",
    "美学与人文素养": "non_cog",
    "劳动与社会实践": "social",
}

# 二级指标评分规范。
# 每个维度包含若干二级指标，格式：
#   (二级指标标签, 维度内权重, 原始字段, 归一化下界, 归一化上界, 越大越好)
# 二级指标得分 = clip((原始值 - 下界) / (上界 - 下界), 0, 1) * 100
#   （若越大越好=False，则取 100 - 上述值）
# 维度得分 = Σ(二级指标得分 × 维度内权重)
DIMENSION_SUBSCORES: Dict[str, List[tuple]] = {
    "study": [
        ("学业成绩", 0.30, "gpa", 1.0, 4.0, True),
        ("学习投入", 0.20, "attendance_rate", 0.5, 1.0, True),
        ("学业过程", 0.15, "homework_timely_rate", 0.4, 1.0, True),
        ("学习能力", 0.15, "credit_progress", 0.6, 1.0, True),
        ("学术创新", 0.20, "competition_awards", 0.0, 5.0, True),
    ],
    "health": [
        ("心理健康", 0.25, "psy_score", 40.0, 100.0, True),
        ("压力水平", 0.12, "stress_index", 0.0, 100.0, False),
        ("身体素质", 0.20, "physical_test_score", 40.0, 100.0, True),
        # 手册 P1 体测四级（优秀/良好/及格/不及格）折算分
        ("体测等级", 0.10, "physical_test_grade_score", 0.0, 100.0, True),
        # 手册 P2 奔跑在北化：完成率 = 有效次数 / 学期规定次数
        ("校园跑完成率", 0.15, "campus_run_ratio", 0.0, 1.0, True),
        ("睡眠质量", 0.09, "sleep_regularity", 0.0, 1.0, True),
        ("情绪状态", 0.09, "mood_check", 40.0, 100.0, True),
    ],
    "social": [
        ("第二课堂", 0.25, "second_class_count", 0.0, 20.0, True),
        ("志愿服务", 0.20, "volunteer_hours", 0.0, 60.0, True),
        ("社团任职", 0.15, "club_role", 0.0, 1.0, True),
        ("人际交往", 0.20, "social_connection", 0.0, 100.0, True),
        ("竞赛获奖", 0.20, "competition_awards", 0.0, 5.0, True),
    ],
    "life": [
        ("饮食规律", 0.25, "canteen_regularity", 0.4, 1.0, True),
        ("作息规律", 0.25, "dorm_regularity", 0.4, 1.0, True),
        ("消费健康", 0.30, "consumption_daily", 10.0, 50.0, True),
        ("离校异常", 0.20, "leave_abnormal", 0.0, 5.0, False),
    ],
    "morality": [
        ("思想政治", 0.20, "political_status", 0.0, 100.0, True),
        ("遵纪守法", 0.25, "discipline_violation", 0.0, 3.0, False),
        # 手册 S1~S3 处分折算计分：无处分 100 / 警告 70 / 严重警告 55 / 记过 40 / 留校察看 20
        ("纪律处分", 0.15, "discipline_penalty_score", 0.0, 100.0, True),
        ("诚信守信", 0.22, "integrity_record", 0.0, 1.0, True),
        ("集体意识", 0.18, "collective_participation", 0.0, 1.0, True),
    ],
    "non_cog": [
        ("恒毅力", 1 / 6, "grit", 0.0, 100.0, True),
        ("自制力", 1 / 6, "self_control", 0.0, 100.0, True),
        ("投入度", 1 / 6, "engagement", 0.0, 100.0, True),
        ("后设认知", 1 / 6, "metacognition", 0.0, 100.0, True),
        ("自我感知", 1 / 6, "self_perception", 0.0, 100.0, True),
        ("学习动机", 1 / 6, "learning_motivation", 0.0, 100.0, True),
    ],
}

# 词云标签词表：(维度键, 二级标签) -> (正向标签, 负向标签, 中性标签)
TAG_DICT: Dict[tuple, tuple] = {
    ("study", "学业成绩"): ("成绩优异", "成绩偏弱", "成绩中等"),
    ("study", "学习投入"): ("出勤规律", "出勤不足", "出勤一般"),
    ("study", "学业过程"): ("作业及时", "作业拖延", "作业平稳"),
    ("study", "学习能力"): ("学分达标", "学分滞后", "学分平稳"),
    ("study", "学术创新"): ("竞赛突出", "创新不足", "偶有成果"),
    ("health", "心理健康"): ("心态积极", "心理预警", "心态平稳"),
    ("health", "压力水平"): ("压力适中", "高压状态", "压力略高"),
    ("health", "身体素质"): ("体格强健", "体质偏弱", "体质中等"),
    ("health", "体测等级"): ("体测优秀", "体测未达标", "体测合格"),
    ("health", "校园跑完成率"): ("跑步达标", "跑步未达标", "跑步进行中"),
    ("health", "睡眠质量"): ("睡眠良好", "睡眠欠佳", "睡眠一般"),
    ("health", "情绪状态"): ("情绪稳定", "情绪波动", "情绪平稳"),
    ("social", "第二课堂"): ("活动积极", "参与较少", "活动一般"),
    ("social", "志愿服务"): ("志愿热心", "志愿较少", "志愿一般"),
    ("social", "社团任职"): ("社团活跃", "社团静默", "社团参与"),
    ("social", "人际交往"): ("社交广泛", "社交孤立", "社交适度"),
    ("social", "竞赛获奖"): ("屡获佳绩", "少有获奖", "偶有获奖"),
    ("life", "饮食规律"): ("饮食规律", "饮食紊乱", "饮食一般"),
    ("life", "作息规律"): ("作息规律", "作息紊乱", "作息一般"),
    ("life", "消费健康"): ("消费适度", "消费异常", "消费平稳"),
    ("life", "离校异常"): ("在校安稳", "离校异常", "离校正常"),
    ("morality", "思想政治"): ("政治过硬", "有待加强", "表现良好"),
    ("morality", "遵纪守法"): ("遵纪守规", "违纪记录", "遵规守纪"),
    ("morality", "纪律处分"): ("无处分记录", "有处分记录", "素行良好"),
    ("morality", "诚信守信"): ("诚信优良", "诚信存疑", "诚信良好"),
    ("morality", "集体意识"): ("集体热心", "集体淡漠", "集体参与"),
    ("non_cog", "恒毅力"): ("坚韧不拔", "恒毅不足", "恒毅中等"),
    ("non_cog", "自制力"): ("自律自强", "自制偏弱", "自制中等"),
    ("non_cog", "投入度"): ("全情投入", "投入不足", "投入中等"),
    ("non_cog", "后设认知"): ("善于反思", "反思不足", "反思一般"),
    ("non_cog", "自我感知"): ("自信笃定", "自信不足", "自信中等"),
    ("non_cog", "学习动机"): ("内驱力强", "动机不足", "动机中等"),
}


def iter_dimensions() -> List[str]:
    """按展示顺序返回维度键列表。"""
    return list(DIMENSION_ORDER)


def get_dimension_subscore_spec(dim_key: str) -> List[tuple]:
    """返回某维度的二级指标评分规范。"""
    return DIMENSION_SUBSCORES.get(dim_key, [])

# 各年级独立聚类配置（可在运营中调整 K 与权重）
GRADE_CLUSTER_CONFIG: Dict[int, GradeClusterConfig] = {
    2022: GradeClusterConfig(grade=2022, k=4),
    2023: GradeClusterConfig(grade=2023, k=4),
    2024: GradeClusterConfig(grade=2024, k=3),
    2025: GradeClusterConfig(grade=2025, k=3),
}


def get_grade_cluster_config(grade: int) -> GradeClusterConfig:
    """获取某年级的聚类配置，缺省年级使用默认兜底配置。"""
    if grade in GRADE_CLUSTER_CONFIG:
        cfg = GRADE_CLUSTER_CONFIG[grade]
    else:
        cfg = GradeClusterConfig(grade=grade, k=4)
    # 填充默认特征权重（用户未单独指定的字段）
    merged = dict(DEFAULT_FEATURE_WEIGHTS)
    merged.update(cfg.feature_weights)
    cfg.feature_weights = merged
    return cfg


# ---------------------------------------------------------------------------
# 3. 权限 / 脱敏配置
# ---------------------------------------------------------------------------
# 角色定义
ROLE_STUDENT = "student"
ROLE_TEACHER = "teacher"

# 脱敏规则（作品书 4.4.3）：姓名仅留首字，学号仅留末四位
MASK_NAME_SHOW = 1          # 姓名可见字数：仅首字（其余用 * 替代）
MASK_ID_KEEP_TAIL = 4       # 学号保留后 N 位

# 教师可管理的班级范围（演示用，工号 -> 年级班级列表）
TEACHER_CLASS_SCOPE: Dict[str, List[str]] = {
    "T2023001": ["2023级-计算机1班", "2023级-计算机2班"],
    "T2023002": ["2024级-软件1班", "2024级-软件2班"],
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def mask_name(name: str) -> str:
    """姓名脱敏：仅保留首字，其余以 * 替代。例如 '张三' -> '张*'，'王小明' -> '王**'。"""
    if not name:
        return ""
    if len(name) <= MASK_NAME_SHOW:
        return name[0] if len(name) == 1 else name[0] + "*" * (len(name) - 1)
    return name[0] + "*" * (len(name) - MASK_NAME_SHOW)


def mask_id(student_id: str) -> str:
    """学号脱敏：仅保留后四位，其余以 * 替代。例如 '2023010101' -> '*******0101'。"""
    if not student_id:
        return ""
    sid = str(student_id)
    if len(sid) <= MASK_ID_KEEP_TAIL:
        return sid
    return "*" * (len(sid) - MASK_ID_KEEP_TAIL) + sid[-MASK_ID_KEEP_TAIL:]


def load_config_from_json(path: str) -> None:
    """从 JSON 文件覆盖加载配置（预留教务系统 / 配置中心下发扩展点）。

    仅覆盖风险阈值与年级聚类配置中出现的键，未出现的保留默认值。
    """
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "risk_threshold" in data:
        for k, v in data["risk_threshold"].items():
            if hasattr(RiskThreshold, k):
                setattr(risk_threshold, k, v)

    if "grade_cluster" in data:
        for grade_str, gcfg in data["grade_cluster"].items():
            grade = int(grade_str)
            base = GRADE_CLUSTER_CONFIG.get(grade, GradeClusterConfig(grade=grade))
            for k, v in gcfg.items():
                if k == "k_search_range" and isinstance(v, list):
                    v = tuple(v)
                if hasattr(base, k):
                    setattr(base, k, v)
            GRADE_CLUSTER_CONFIG[grade] = base


# 全局默认风险阈值实例（可直接修改其属性以在运行时调整）
risk_threshold = RiskThreshold()


# ---------------------------------------------------------------------------
# 5. 本科手册硬性规则（依据 2024 级《本科生手册》）
# ---------------------------------------------------------------------------
# 设计说明：
#   本区块只做「这个学生是否需要被提醒 / 督促」的客观判定，用于生成智能提醒话术。
#   判定结果不额外展示状态标签，也不替代六维评分（评分口径见 DIMENSION_SUBSCORES）。
# 全部阈值集中于此，便于按年级 / 学院通知调整。
HANDBOOK_RULES: Dict[str, dict] = {
    # ---------------- 学业类 A ----------------
    "A1": {
        "name": "学业警示（无学位警示）",
        "desc": "课程最高平均 GPA 低于 2.00 触发学业警示",
        "user_clause": "课程最高平均 GPA 低于 2.00 的学生将被给予学业警示。触发警示后需尽快安排课程重修与学业答疑。",
        "gpa_line": 2.00,
    },
    "A2": {
        "name": "年级晋级",
        "desc": "必修学分完成率低于本年级晋级线，需要提醒跟进",
        "user_clause": "学生须按年级完成规定学分方可晋级：大一学年必修学分完成率不低于 2/3，大二、大三不低于 3/4。未达到晋级要求者按学籍管理规定作留级或降级处理。",
        "credit_ratio": {1: 2 / 3, 2: 3 / 4, 3: 3 / 4},
        "note": "大四不再做晋级评估，转为毕业条件 / 学位条件核查",
    },
    "A3": {
        "name": "退学红线",
        "desc": "大一必修学分低于 1/3；连续 2 次未晋级；累计 3 次未晋级",
        "user_clause": "退学红线：大一学年必修学分完成率低于 1/3；或连续 2 次、累计 3 次未达到晋级要求。触及者按手册规定启动退学处理程序。",
        "freshman_credit_ratio": 1 / 3,
        "consecutive_fail": 2,
        "total_fail": 3,
    },
    "A4": {
        "name": "毕业条件",
        "desc": "德育合格、处分已解除或未受处分、体测达标、修满毕业最低学分",
        "user_clause": "毕业条件：德育合格、无处分记录或处分已解除、体质健康测试达标（大一大二两学年平均不低于 60 分）、修满培养方案规定的全部学分。",
        # 业务方口径：体测按大一大二两学年平均分判定
        "fitness_avg_years": (1, 2),
        "fitness_avg_line": 60.0,
        "note": "德育合格与专业能力等由院系审核，系统仅对可量化项作提示",
    },
    "A5": {
        "name": "学士学位",
        "desc": "课程最高平均 GPA ≥ 2.00 且 毕设 GPA ≥ 2.00",
        "user_clause": "学士学位授予条件：课程最高平均 GPA 不低于 2.00，且毕业设计（论文）GPA 不低于 2.00。",
        "gpa_line": 2.00,
        "thesis_gpa_line": 2.00,
    },
    # ---------------- 体质健康类 P ----------------
    "P1": {
        "name": "体质健康测试",
        "desc": "体测四级；毕业评定低于 50 按结业处理；不及格准予补测一次",
        "user_clause": "体质健康测试按四级评定：90 分及以上优秀、80~89.9 良好、60~79.9 及格、60 分以下不及格。不及格者本学期可补测一次；毕业评定低于 50 分按结业处理；办理免测者按 50 分计，无故缺考且未请假者各项按 0 分计。",
        "grade_lines": [(90.0, "优秀"), (80.0, "良好"), (60.0, "及格")],  # 低于 60 为不及格
        "graduation_line": 50.0,
        "exempt_score": 50.0,      # 免测按 50 分计
        "absent_zero": True,       # 无故未测且未请假，各项按 0 分计
        "makeup_once": True,       # 不及格本学期准予补测一次
    },
    "P2": {
        "name": "奔跑在北化（校园跑）",
        "user_clause": "「奔跑在北化」校园跑：每学期有效次数不少于 40 次（每周不少于 3 次；单次里程男不低于 1.5 公里、女不低于 1.2 公里；配速 4~10 分钟/公里；每日最多计 2 次、每周最多计 8 次）。有效次数不达标者，该学期体育课成绩按不及格处理。学年累计里程男不低于 130 公里、女不低于 110 公里。",
        # 学期规定次数以当学期启动通知为准，此处为演示默认值，可一处调整
        "semester_target": 40,
        "weekly_min": 3,                          # 每周不少于 3 次
        "single_km": {"男": 1.5, "女": 1.2},      # 单次最低有效里程
        "pace_range": (4.0, 10.0),                # 有效配速区间（分钟/公里）
        "daily_max": 2,                           # 每日最多计 2 次
        "weekly_max": 8,                          # 每周最多计 8 次
        "year_km": {"男": 130.0, "女": 110.0},    # 学年累计里程要求
        # 逐次记录按学年编号（1~32 周 = 上/下两个学期），本学期取后半段
        "year_weeks": (1, 32),
        "semester_weeks": (17, 32),
        "note": "学期有效次数不达标按体育课不及格（一票否决）",
    },
    "P3": {
        "name": "跑步诚信",
        "desc": "代跑、骑车打卡等违规行为按手册处理",
        "user_clause": "跑步诚信：代跑、骑车打卡等违规行为按手册相关规定处理，并记入诚信记录。",
        "violation_threshold": 1,                 # 出现即提醒
    },
    # ---------------- 学籍纪律类 S ----------------
    "S1": {
        "name": "课程考核资格",
        "desc": "缺课超过 1/3 或欠交作业超过 1/4，取消课程考核资格",
        "user_clause": "课程考核资格：缺课超过该课程总学时 1/3，或欠交作业超过 1/4，取消该课程考核资格。",
        "absence_ratio": 1 / 3,
        "homework_miss_ratio": 1 / 4,
    },
    "S2": {
        "name": "缺席（旷课）处分",
        "desc": "累计缺席学时达到对应档位给予纪律处分",
        "user_clause": "缺席处分：一学期累计缺席 20、30、40、60 学时，分别给予警告、严重警告、记过、留校察看处分。",
        "thresholds": [(20, "警告"), (30, "严重警告"), (40, "记过"), (60, "留校察看")],
    },
    "S3": {
        "name": "未经批准离校 / 学籍情形",
        "desc": "未经批准连续离校按天数处分；不参加活动 / 未注册可触及退学情形",
        "user_clause": "未经批准连续离校 4、6、8、10 天，分别给予警告、严重警告、记过、留校察看处分；连续 2 周或累计 4 周不参加学校规定的活动，或逾期 2 周未注册，按学籍规定可予退学处理。",
        "leave_thresholds": [(4, "警告"), (6, "严重警告"), (8, "记过"), (10, "留校察看")],
        "inactive_consecutive_weeks": 2,   # 连续 2 周不参加规定活动
        "inactive_total_weeks": 4,         # 累计 4 周不参加规定活动
        "unregistered_weeks": 2,           # 逾期 2 周未注册
    },
    # ---------------- 第二课堂 EC ----------------
    "EC": {
        "name": "第二课堂计分",
        "desc": "志愿服务 / 劳动实践 / 社会实践 / 美育等按手册明文规则计分",
        "user_clause": "第二课堂成绩单按五部分计分（合计 600 分）：道德与思想素质 180 分、学术与科技创新 120 分、体育与健康 100 分、美学与人文素养 100 分、劳动与社会实践 100 分。志愿服务每学期满 10 小时计 20 分（超出部分每小时 0.5 分，拓展上限 5 分）；劳动实践每学期满 10 小时计 20 分；完成寒暑假社会实践并提交报告计 15 分；美育平均每学期不低于 60 分为毕业要求。",
        "volunteer_hours": 10,             # 每学期志愿 ≥10h 计 20 分
        "volunteer_score": 20,
        "volunteer_extra_per_hour": 0.5,   # 超出部分每小时 0.5 分
        "volunteer_extra_cap": 5,          # 拓展分上限
        "labor_hours": 10,                 # 每学期劳动 ≥10h 计 20 分
        "labor_score": 20,
        "social_practice_score": 15,       # 完成寒暑假社会实践并交报告
        "aesthetic_per_semester": 60,      # 美育平均每学期 ≥60 分
        # 宿舍卫生检查分 -> 计分档（优 / 良 / 中 / 合格 / 不合格）
        "dorm_hygiene": [
            ("优", 90, 100, 20),
            ("良", 80, 90, 16),
            ("中", 70, 80, 12),
            ("合格", 60, 70, 10),
            ("不合格", 0, 60, 0),
        ],
        "parts_max": {
            "道德与思想素质": 180,
            "学术与科技创新": 120,
            "体育与健康": 100,
            "美学与人文素养": 100,
            "劳动与社会实践": 100,
        },
        "total_max": 600,
    },
}


# 智能提醒话术模板（只做客观提示，不做主观评判）
HANDBOOK_REMINDERS: Dict[str, str] = {
    "gpa_low": "当前课程最高平均 GPA 为 {gpa}，低于学位授予要求的 2.00，建议尽快安排课程重修与答疑。",
    "credit_warn": "本学期必修学分完成率为 {ratio}，低于 {grade} 年级晋级要求（{threshold}），建议规划好后续选课与重修。",
    "credit_freshman": "大一学年必修学分完成率仅 {ratio}，低于 1/3，已触及退学警示红线，建议尽快与辅导员沟通。",
    "unpass_consecutive": "已连续 {n} 次未达到晋级要求，请注意手册中关于退学处理的相关规定。",
    "unpass_total": "累计 {n} 次未达到晋级要求，请注意手册中关于退学处理的相关规定。",
    "fitness_fail": "本学期体测总分 {score} 分，未达 60 分及格线，按规定可申请补测一次，建议加强锻炼。",
    "fitness_level": "体测总分 {score} 分，达到{level}水平，继续保持。",
    "fitness_exempt": "本学期体测已办理免测，毕业评定时按 {score} 分计。",
    "fitness_absent": "本学期体测无故缺考且未请假，各项按规定按 0 分计，请尽快联系体育部门说明情况。",
    "run_progress": "当前校园跑次数 {done}/{target}，建议之后每周跑 {weekly} 次。",
    "run_up": "这周校园跑比上周多跑了 {n} 次，加油！",
    "run_done": "本学期校园跑已完成 {done}/{target}，达标完成，继续保持运动习惯。",
    "run_year_km": "学年累计跑量 {km} 公里，距 {target} 公里目标还差 {gap} 公里。",
    "absence_s1": "本学期缺课比例已达 {ratio}，超过 1/3，按规定可能被取消课程考核资格，建议尽快补齐出勤。",
    "absence_s2": "本学期累计缺席 {hours} 学时，已达手册「{level}」处分标准，请及时向辅导员说明情况。",
    "leave_s3": "未经批准连续离校 {days} 天，已达手册「{level}」处分标准，请及时返校并补办手续。",
    "inactive_s3": "已{kind} {weeks} 周未参加规定活动，请注意手册中关于退学处理的相关规定。",
    "unregistered_s3": "逾期未注册已达 {weeks} 周，请注意学籍相关规定，尽快办理注册手续。",
    "ec_volunteer": "本学期志愿服务 {hours} 小时，距 10 小时计 20 分的要求仍有差距，建议继续参与。",
    "ec_labor": "本学期劳动实践 {hours} 小时，未达 10 小时的计分要求，建议尽快补充。",
    "ec_aesthetic": "美育平均每学期 {score} 分，尚未达到 60 分的毕业要求，建议多参与美育类活动。",
    "ec_social": "本学期寒暑假社会实践尚未完成，完成后可计 15 分，建议尽快安排。",
    "homework_s1": "本学期欠交作业比例已达 {ratio}，超过 1/4，按规定可能被取消课程考核资格，建议尽快补齐作业。",
    "run_pe_fail": "本学期校园跑有效次数 {done}/{target} 未达规定次数，按规定该学期体育课成绩按不及格处理。",
    "run_violation": "本学期存在 {n} 次跑步违规记录（代跑 / 骑车打卡等），请按手册要求规范打卡。",
    "fitness_graduation": "体测毕业评定为 {score} 分，低于 50 分，按手册规定将影响毕业结论，建议尽快与体育部门沟通。",
    "graduation_fitness": "大一大二两学年体测平均为 {avg} 分，低于 60 分，按毕业条件要求需重点关注体测达标情况。",
    "graduation_discipline": "存在尚未解除的纪律处分记录，按毕业条件要求需先完成处分解除程序。",
    "thesis_gpa_low": "毕业设计（论文）GPA 为 {gpa}，低于学位授予要求的 2.00，建议尽快与指导教师沟通。",
    "praise_clean": "本学期未出现纪律处分与考试违纪记录，表现良好。",
}


def get_handbook_rule(rule_id: str) -> dict:
    """按编号获取手册规则（如 'A1' / 'P2' / 'EC'）。"""
    return HANDBOOK_RULES.get(rule_id, {})


def cohort_to_level(grade: int, today=None) -> int:
    """把入学年份（students.grade，如 2023）换算为当前年级序号（1=大一 … 4=大四）。

    学年以 9 月为界，跨学年自动推进，避免把年级写死导致逐年失效。
    例：2026 年 9 月，2023 级 = 大四，2024 级 = 大三。
    """
    from datetime import date as _date

    d = today or _date.today()
    academic_year = d.year if d.month >= 9 else d.year - 1
    return max(1, min(4, academic_year - int(grade) + 1))
