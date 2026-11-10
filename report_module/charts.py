"""report_module 可视化图表模块。

基于 Plotly / Matplotlib 生成交互图表，所有图表均由分析结果动态计算生成，
非前端写死静态数据：
1. 六维能力雷达图（学业 / 身心 / 社会参与 / 生活适应 / 品德 / 非认知）。
2. 成长趋势折线图（基于当前综合评分回溯模拟历史学期走势）。
3. 班级风险等级分布图（饼图 / 柱状图）。
4. 学生画像词云图（Matplotlib + wordcloud，人形/圆形 mask，中文渲染）。
5. 维度分析树（Plotly Treemap 矩形树图，学生端可切换 / 教师端六维）。
6. 维度内二级指标柱状图。
"""

from __future__ import annotations

import colorsys
import os
import random

import plotly.graph_objects as go
import streamlit as st

import config_setting as C
from config_setting import (
    DIMENSION_ORDER,
    DIMENSION_LABELS
)


def _hash_obj(obj) -> int:
    """把 list/dict 等结构稳定序列化后取哈希，供 @st.cache_data 使用。"""
    import json
    return hash(json.dumps(obj, sort_keys=True, default=str))


# ---------------------------------------------------------------------------
# 主题色板（全部集中自 config_setting，本文件不再硬编码色值）
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 维度同色系色阶（学生端六个维度分页面使用）
# ---------------------------------------------------------------------------
# 把得分渐变的色相由统一橄榄绿换成各维度主题色，亮度档位保持不变，
# 因此「颜色越深 = 得分越高」的读法、以及白色文字的对比度都与原来一致。
_HUE_SCALE_LEVELS = (0.92, 0.64, 0.37)   # 与 C.SCORE_COLORSCALE 三档亮度对齐
_HUE_SCALE_CACHE: dict = {}


def _hex_to_rgb(hex_color: str):
    c = hex_color.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(
        *(max(0, min(255, int(round(v)))) for v in rgb)
    )


def build_hue_colorscale(base_hex: str) -> list:
    """按维度主题色生成同色系连续色阶（浅色=低分，深色=高分）。

    保持 base_hex 的色相与饱和度，仅按固定亮度档位重新取样，
    因此换色不改变原有「深浅表示得分高低」的视觉含义。
    """
    if base_hex in _HUE_SCALE_CACHE:
        return _HUE_SCALE_CACHE[base_hex]
    r, g, b = _hex_to_rgb(base_hex)
    h, _l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    scale = [
        [pos, _rgb_to_hex(tuple(v * 255 for v in colorsys.hls_to_rgb(h, lv, s)))]
        for pos, lv in zip((0.0, 0.5, 1.0), _HUE_SCALE_LEVELS)
    ]
    _HUE_SCALE_CACHE[base_hex] = scale
    return scale

# ---------------------------------------------------------------------------
# 中文 / CJK 字体（词云与 matplotlib 渲染需要）
# ---------------------------------------------------------------------------
def get_cjk_font_path() -> str | None:
    """查找系统中可用的中文字体，供 wordcloud / matplotlib 使用。

    优先 Windows 微软雅黑；找不到则返回 None（调用方需降级）。
    """
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _dark_enough_pool(palette: list, max_lightness: float = 0.72) -> list:
    """从色板中筛出亮度足够低的颜色，保证小字号文字肉眼清晰可读。"""
    out = []
    for hex_color in palette:
        r, g, b = _hex_to_rgb(hex_color)
        _h, lightness, _s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
        if lightness <= max_lightness:
            out.append(hex_color)
    return out or list(palette)


# 词云布词随机种子序列：首个使可见词 ≥40 的种子被采用（确定性复现）
_WORDCLOUD_SEEDS = (42, 7, 13, 21, 33, 99, 5, 77)


def _wc_small_pool() -> list:
    """按「当前皮肤」实时筛出小字号可用深色池。

    不能写成模块级常量：那会在 import 时固化成默认皮肤色板，换肤后词云仍是旧配色。
    """
    return _dark_enough_pool(C.WORDCLOUD_PALETTE)


def _wc_color_func(word, font_size, position, orientation,
                   random_state=None, **kwargs):
    """词云取色：字号越大颜色越深，整体限制在主题低饱和色板内。

    小字号词只用深色系，避免浅色小字看不清；色池全部按当前皮肤实时取。
    """
    if random_state is None:
        random_state = random
    if font_size > 90:
        pool = C.WORDCLOUD_DEEP_POOL
    elif font_size > 55:
        pool = C.WORDCLOUD_MID_POOL
    else:
        pool = _wc_small_pool()
    return random_state.choice(pool)


# ---------------------------------------------------------------------------
# 六维雷达图
# ---------------------------------------------------------------------------
def _rgba(hex_color: str, opacity: float) -> str:
    """HEX -> rgba() 字符串（用于半透明填充叠加）。"""
    r, g, b = _hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{opacity})"


def build_radar(row: dict, ref_means: dict = None) -> go.Figure:
    """构建六维度能力雷达图。

    ref_means: 可选，{标签: {维度键: 分值}}，用于在雷达图上叠加班级 / 年级均值
    参考面（半透明填充叠加，图例可单独点击显隐每条参考系列）。
    """
    dims = [DIMENSION_LABELS[d] for d in DIMENSION_ORDER]
    keys = DIMENSION_ORDER
    vals = [float(row.get(f"score_{k}", 0) or 0) for k in keys]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=dims + [dims[0]],
        fill="toself",
        name="个人能力",
        line_color=C.THEME_PRIMARY,
        fillcolor=C.RADAR_SELF_FILL,
    ))

    if ref_means:
        for i, (label, rm) in enumerate(ref_means.items()):
            rref = [float(rm.get(k, 0) or 0) for k in keys]
            color = C.RADAR_REF_COLORS[i % len(C.RADAR_REF_COLORS)]
            # 均值参考由虚线升级为半透明填充叠加，强弱对比更直观；
            # 图例点击可单独显隐（Plotly 原生交互）
            fig.add_trace(go.Scatterpolar(
                r=rref + [rref[0]],
                theta=dims + [dims[0]],
                fill="toself",
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
                fillcolor=_rgba(color, C.RADAR_REF_FILL_OPACITY),
            ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=C.THEME_GRID),
            angularaxis=dict(gridcolor=C.THEME_GRID),
            bgcolor=C.THEME_CARD_BG,
        ),
        showlegend=bool(ref_means),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                    itemclick="toggle", itemdoubleclick="toggleothers"),
        height=360,
        margin=dict(l=40, r=40, t=30, b=30),
        paper_bgcolor=C.THEME_CARD_BG,
        font=dict(color=C.THEME_TEXT),
    )
    return fig


# ---------------------------------------------------------------------------
# 成长趋势
# ---------------------------------------------------------------------------
def trend_scores(seed_key: str, current_score: float, semesters: int = 6) -> list[float]:
    """基于当前综合评分确定性回溯历史学期走势（截面快照的演示走势）。

    用 md5 学号派生种子而非内置 hash()：后者每个进程随机化
    （PYTHONHASHSEED），会导致同一学生每次打开页面趋势都不一样。
    网页端与 PDF 端共用本函数，保证两处图形一致。
    """
    import hashlib
    seed = int(hashlib.md5(str(seed_key).encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(seed % (10 ** 8))
    scores = [float(current_score)]
    for _ in range(semesters - 1):
        delta = rng.gauss(-2, 4)  # 历史总体略低于当前
        prev = max(20, min(100, scores[-1] + delta))
        scores.append(prev)
    return scores[::-1]  # 时间正序


def build_trend(student_id: str, current_score: float, semesters: int = 6) -> go.Figure:
    """构建成长趋势折线图。

    由于原始数据为截面快照，这里基于当前综合评分用确定性随机游走
    回溯生成历史学期走势（仅用于演示趋势形态）。
    """
    scores = trend_scores(student_id, current_score, semesters)

    labels = [f"第{i}学期" for i in range(1, semesters + 1)]
    fig = go.Figure()
    # 面积图：走势下方铺主题色浅填充，视觉重心落在最新学期
    fig.add_trace(go.Scatter(
        x=labels, y=scores, mode="lines+markers",
        line=dict(color=C.THEME_PRIMARY, width=3),
        marker=dict(size=8, color=C.THEME_CARD_BG,
                    line=dict(color=C.THEME_PRIMARY, width=2)),
        fill="tozeroy",
        fillcolor=C.TREND_FILL,
    ))

    # 末端徽章：较上学期 +/-X.X 分
    if len(scores) >= 2:
        delta = scores[-1] - scores[-2]
        badge = f"较上学期 {'+' if delta >= 0 else ''}{delta:.1f} 分"
        fig.add_annotation(
            x=labels[-1], y=scores[-1],
            text=badge, showarrow=True, arrowhead=2, ax=0, ay=-34,
            bgcolor=C.THEME_PRIMARY, bordercolor=C.THEME_PRIMARY,
            font=dict(color=C.THEME_CARD_BG, size=12),
            borderpad=4, opacity=0.95,
        )
    # 图下注记：演示数据说明（接入真实学期数据后自动替换）
    fig.add_annotation(
        xref="paper", yref="paper", x=0, y=-0.22,
        text="注：历史走势为基于当前综合评分回溯生成的演示走势（原始数据为截面快照）。",
        showarrow=False, font=dict(size=11, color=C.THEME_TEXT_SUB),
        xanchor="left",
    )
    fig.update_layout(
        height=360,
        margin=dict(l=40, r=40, t=30, b=56),
        yaxis=dict(title="综合评分", range=[0, 100], gridcolor=C.THEME_GRID),
        xaxis=dict(title="学期", gridcolor=C.THEME_GRID),
        paper_bgcolor=C.THEME_CARD_BG,
        plot_bgcolor=C.THEME_CARD_BG,
        font=dict(color=C.THEME_TEXT),
    )
    return fig


# ---------------------------------------------------------------------------
# 风险分布
# ---------------------------------------------------------------------------
def build_distribution(distribution: dict, chart_type: str = "pie") -> go.Figure:
    """构建班级风险等级分布图。

    chart_type: 'pie' 饼图 | 'bar' 柱状图。
    """
    labels_order = ["正常", "关注", "高危"]
    labels = [l for l in labels_order if l in distribution] + \
             [l for l in distribution if l not in labels_order]
    values = [distribution.get(l, 0) for l in labels]
    colors = [C.RISK_CHART_COLORS.get(l, C.CHART_NEUTRAL) for l in labels]

    if chart_type == "bar":
        fig = go.Figure(go.Bar(
            x=labels, y=values,
            marker_color=colors,
            text=values,
            textposition="outside",
        ))
        fig.update_layout(
            height=340,
            margin=dict(l=40, r=40, t=30, b=40),
            yaxis=dict(title="人数", gridcolor=C.THEME_GRID),
            xaxis=dict(gridcolor=C.THEME_GRID),
            paper_bgcolor=C.THEME_CARD_BG,
            plot_bgcolor=C.THEME_CARD_BG,
            font=dict(color=C.THEME_TEXT),
        )
    else:
        fig = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.55,
            marker_colors=colors,
            textinfo="label+percent",
            insidetextorientation="horizontal",
        ))
        fig.update_layout(
            height=340,
            margin=dict(l=40, r=40, t=30, b=30),
            paper_bgcolor=C.THEME_CARD_BG,
            font=dict(color=C.THEME_TEXT),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        )
    return fig


# ---------------------------------------------------------------------------
# 二级指标柱状图
# ---------------------------------------------------------------------------
def build_sub_bar(sub_scores: dict, dim_label: str, dim_color: str | None = None) -> go.Figure:
    """构建某维度下二级指标得分柱状图。

    sub_scores: {二级标签: 得分(0~100)}。
    dim_color: 该维度主题色。传入时所有柱子统一用该纯色（学生端维度页，
               使页面与所选维度主题色一致）；不传时沿用按得分深浅的橄榄绿色阶。
    """
    labels = list(sub_scores.keys())
    vals = [float(sub_scores[k]) for k in labels]

    if dim_color:
        # 纯色：与本维度主题色一致
        marker = dict(color=dim_color, line=dict(width=0))
    else:
        marker = dict(
            color=vals,
            colorscale=C.SCORE_COLORSCALE,
            cmin=0,
            cmax=100,
            line=dict(width=0),
        )

    fig = go.Figure(go.Bar(
        x=labels,
        y=vals,
        marker=marker,
        text=[f"{v:.1f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=40, r=40, t=50, b=40),
        yaxis=dict(title="得分", range=[0, 100], gridcolor=C.THEME_GRID),
        xaxis=dict(gridcolor=C.THEME_GRID),
        title=dict(text=f"{dim_label} · 二级指标得分", font=dict(size=15, color=C.THEME_TEXT)),
        paper_bgcolor=C.THEME_CARD_BG,
        plot_bgcolor=C.THEME_CARD_BG,
        font=dict(color=C.THEME_TEXT),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# 维度分析树（Treemap）
# ---------------------------------------------------------------------------
def build_treemap(sub_scores: dict, dim_label: str, dim_key: str = None,
                  dim_color: str | None = None) -> go.Figure:
    """构建维度分析树（Plotly Treemap 矩形树图）。

    sub_scores: {二级标签: 得分(0~100)}；父节点为维度根，叶子面积反映得分。
    叶子颜色按得分连续色阶着色，避免红绿橙。
    dim_color: 该维度主题色。传入时色阶改用该维度的同色系（浅=低分，深=高分），
               深浅语义与色阶条保持一致；不传时沿用统一的橄榄绿色阶。
    """
    labels = [dim_label] + list(sub_scores.keys())
    parents = [""] + [dim_label] * len(sub_scores)
    values = [sum(sub_scores.values())] + [float(v) for v in sub_scores.values()]
    leaf_scores = [float(v) for v in sub_scores.values()]

    # 根节点用二级指标平均分着色，保证 marker.colors 全为数值、可统一映射连续色阶
    root_score = sum(leaf_scores) / len(leaf_scores) if leaf_scores else 0.0
    node_colors = [root_score] + leaf_scores

    colorscale = build_hue_colorscale(dim_color) if dim_color else C.SCORE_COLORSCALE

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(
            colors=node_colors,
            colorscale=colorscale,
            cmin=0,
            cmax=100,
            showscale=True,
            colorbar=dict(title="得分", ticksuffix="分"),
        ),
        textinfo="label+value",
        hovertemplate="%{label}<br>得分 %{value:.1f}<extra></extra>",
        textfont=dict(color=C.THEME_CARD_BG, size=13),
    ))
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=C.THEME_CARD_BG,
        font=dict(color=C.THEME_TEXT),
    )
    return fig


# ---------------------------------------------------------------------------
# 词云（半身人形剪影 mask + 细轮廓线）
# ---------------------------------------------------------------------------
def _bust_mask(width: int = 600, height: int = 620):
    """生成「半身人像」剪影 mask（头部 + 肩部 + 上胸），供 WordCloud 使用。

    构型：正圆头部（直径约占高度 26%）+ 颈部 + 圆角肩部（椭圆帽）+ 上胸
    （直边矩形），底部在约 62% 高度处平切收束；全身像（手臂 / 双腿）刻意不取。
    返回 numpy 数组：0 = 可布词区域，255 = 留空区域（wordcloud 约定）。
    """
    import numpy as np
    from PIL import Image, ImageDraw

    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    cx = width // 2

    # 头部：正圆，直径约 26% 高度
    r_head = int(height * 0.13)
    head_cy = int(height * 0.165)
    draw.ellipse([cx - r_head, head_cy - r_head, cx + r_head, head_cy + r_head], fill=0)

    # 颈部：竖矩形，上端没入头部、下端没入肩部
    neck_half = int(r_head * 0.55)
    draw.rectangle([cx - neck_half, head_cy + int(r_head * 0.7),
                    cx + neck_half, int(height * 0.40)], fill=0)

    # 肩部 + 上胸：椭圆帽（圆角肩线）下接直边矩形，底部在 62% 高度平切
    # 肩宽取 0.44：剪影仍为自然半身比例，同时保证轮廓内有足够布词面积
    sh_cy = int(height * 0.42)
    sh_rx = int(width * 0.44)
    sh_ry = int(height * 0.145)
    draw.ellipse([cx - sh_rx, sh_cy - sh_ry, cx + sh_rx, sh_cy + sh_ry], fill=0)
    cut_y = int(height * 0.62)
    draw.rectangle([cx - sh_rx, sh_cy, cx + sh_rx, cut_y], fill=0)

    return np.array(img)


@st.cache_data(ttl=300, show_spinner=False, hash_funcs={list: _hash_obj})
def build_wordcloud(tags: list, width: int = 660, height: int = 680, skin: str = ""):
    """构建学生画像词云图（Matplotlib Figure）。

    tags: [(标签, 权重), ...]，字号与权重正相关规则不变。
    skin: 仅用于 @st.cache_data 的缓存分区。词云配色按当前皮肤实时取，
          缓存键若不含皮肤，网页内换肤会直接复用上一套皮肤生成的图。
    使用半身人形剪影 mask（头部 + 肩部 + 上胸），人形外沿以主题边框色细线勾勒；
    词密集填充轮廓内部，配色沿用主题低饱和色板（小字只用深色系保证可读）。
    返回 matplotlib.figure.Figure；若 wordcloud 不可用或无中文字体，回退为水平条形词频图。

    布词说明：wordcloud 库在某个词放不下时会中断放置后续全部词（级联中断），
    因此按固定种子序列逐个尝试，取首个可见词 ≥40 的结果（均无则取最多者）。
    种子序列固定，同一学生的词云跨进程可复现。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font_path = get_cjk_font_path()

    try:
        from wordcloud import WordCloud

        freq = {t: max(1.0, float(w)) for t, w in tags}
        mask = _bust_mask(width, height)
        wc = None
        for seed in _WORDCLOUD_SEEDS:
            cand = WordCloud(
                width=width,
                height=height,
                mask=mask,
                contour_width=1,
                contour_color=C.THEME_CARD_BORDER,   # 主题边框色细轮廓线，勾勒人形外沿
                background_color=C.THEME_CARD_BG,
                prefer_horizontal=0.85,
                relative_scaling=0.25,
                margin=1,
                max_words=220,
                min_font_size=14,
                max_font_size=80,
                color_func=_wc_color_func,
                font_path=font_path,
                random_state=seed,
            ).generate_from_frequencies(freq)
            if wc is None or len(cand.layout_) > len(wc.layout_):
                wc = cand
            if len(cand.layout_) >= 40:
                wc = cand
                break

        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        return fig
    except Exception:
        # 回退：水平条形图
        fig = plt.figure(figsize=(7, 4.5), dpi=100)
        ax = fig.add_subplot(111)
        if tags:
            tags_sorted = sorted(tags, key=lambda x: x[1], reverse=True)[:15]
            words = [t for t, _ in tags_sorted][::-1]
            weights = [w for _, w in tags_sorted][::-1]
            colors_bar = [C.SCORE_COLORSCALE[int(min(1.0, w / 100) * 2)][1] for w in weights]
            ax.barh(words, weights, color=colors_bar)
            ax.set_xlabel("权重", color=C.THEME_TEXT)
            ax.tick_params(colors=C.THEME_TEXT)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        ax.set_title("学生特征标签", fontsize=13, color=C.THEME_TEXT)
        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# 定 K 过程曲线（肘部法则 SSE + 轮廓系数 + 合成得分，作品书 3.4.1）
# ---------------------------------------------------------------------------
def build_k_selection_curve(details: dict, grade: int | None = None) -> go.Figure | None:
    """把自动定 K 过程画成双轴曲线图。

    details: choose_best_k 返回的 {tested_k, inertias, silhouettes, scores, best_k}。
    X = 候选 K（2~8）；左轴 SSE（越小越好），右轴轮廓系数（越大越好）；
    合成得分（肘部 0.4 + 轮廓 0.6 加权）单独一条高亮曲线，并标注选中的最优 K。
    """
    tested = list(details.get("tested_k") or [])
    if not tested:
        return None
    inertias = list(details.get("inertias") or [])
    silhouettes = list(details.get("silhouettes") or [])
    scores = [s for s in (details.get("scores") or [])]
    best_k = details.get("best_k")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tested, y=inertias, mode="lines+markers", name="SSE（左轴，越小越好）",
        line=dict(color=C.CHART_SECONDARY, width=2), marker=dict(size=7),
    ))
    fig.add_trace(go.Scatter(
        x=tested, y=silhouettes, mode="lines+markers", name="轮廓系数（右轴，越大越好）",
        line=dict(color=C.CHART_ACCENT, width=2, dash="dot"), marker=dict(size=7),
        yaxis="y2",
    ))
    if any(s is not None for s in scores):
        fig.add_trace(go.Scatter(
            x=tested, y=scores, mode="lines+markers",
            name="合成得分（肘部0.4+轮廓0.6）",
            line=dict(color=C.THEME_PRIMARY, width=3), marker=dict(size=8),
            yaxis="y2",
        ))
    # 高亮最优 K 点并标注
    if best_k in tested:
        bi = tested.index(best_k)
        ann_y = None
        if bi < len(scores) and scores[bi] is not None:
            ann_y = scores[bi]
        elif bi < len(silhouettes):
            ann_y = silhouettes[bi]
        if ann_y is not None:
            grade_txt = f"{grade} 级" if grade else "本年级"
            fig.add_annotation(
                x=best_k, y=ann_y, yref="y2",
                text=f"{grade_txt}最优 K={best_k}<br>（肘部 0.4 + 轮廓 0.6 加权自动确定）",
                showarrow=True, arrowhead=2, ax=0, ay=-44,
                bgcolor=C.THEME_PRIMARY, bordercolor=C.THEME_PRIMARY,
                font=dict(color=C.THEME_CARD_BG, size=12), borderpad=4,
            )
            fig.add_trace(go.Scatter(
                x=[best_k], y=[ann_y], mode="markers", yaxis="y2",
                marker=dict(size=15, color=C.THEME_PRIMARY, symbol="circle-open",
                            line=dict(width=3, color=C.THEME_PRIMARY_DARK)),
                showlegend=False, hoverinfo="skip",
            ))
    fig.update_layout(
        height=340,
        margin=dict(l=50, r=50, t=30, b=40),
        xaxis=dict(title="候选 K", dtick=1, gridcolor=C.THEME_GRID),
        yaxis=dict(title="SSE（肘部法则）", gridcolor=C.THEME_GRID),
        yaxis2=dict(title="轮廓系数 / 合成得分", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28),
        paper_bgcolor=C.THEME_CARD_BG,
        plot_bgcolor=C.THEME_CARD_BG,
        font=dict(color=C.THEME_TEXT),
    )
    return fig


# ---------------------------------------------------------------------------
# 第二课堂成绩单 donut（五段 = 五维主题色，中心总分 / 600）
# ---------------------------------------------------------------------------
def build_second_classroom_donut(parts: dict, caps: dict, part_colors: dict,
                                 total: float, total_max: float) -> go.Figure:
    """构建第二课堂成绩单环形图。

    parts: {部分名: 得分}；caps: {部分名: 上限}；
    part_colors: {部分名: 主题色}（五段对应德/智/体/美/劳五个维度主题色）。
    中心显示「总分 / 满分」。
    """
    names = list(parts.keys())
    vals = [float(parts[n]) for n in names]
    colors = [part_colors.get(n, C.CHART_NEUTRAL) for n in names]
    hover = [
        f"{n}<br>{float(parts[n]):.0f} / {float(caps.get(n, 0)):.0f} 分<extra></extra>"
        for n in names
    ]
    fig = go.Figure(go.Pie(
        labels=names, values=vals, hole=0.62,
        marker_colors=colors,
        textinfo="label+value",
        hovertemplate=hover,
        insidetextorientation="horizontal",
        sort=False,
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor=C.THEME_CARD_BG,
        font=dict(color=C.THEME_TEXT),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18),
        annotations=[dict(
            text=f"<b>{total:.0f}</b><br><span style='font-size:12px'>/ {total_max:.0f} 分</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=26, color=C.THEME_PRIMARY_DARK),
        )],
    )
    return fig


# ---------------------------------------------------------------------------
# 维度分析树（节点连线层级树，纯 Plotly 自绘：圆角节点 + 肘形连线）
# ---------------------------------------------------------------------------
def _scale_hex(colorscale: list, t: float) -> str:
    """在 [[pos, hex], ...] 连续色阶上取 t∈[0,1] 处的颜色。"""
    t = max(0.0, min(1.0, float(t)))
    pts = sorted(colorscale, key=lambda p: p[0])
    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        p0, c0 = pts[i]
        p1, c1 = pts[i + 1]
        if p0 <= t <= p1:
            r0, g0, b0 = _hex_to_rgb(c0)
            r1, g1, b1 = _hex_to_rgb(c1)
            k = 0.0 if p1 - p0 < 1e-9 else (t - p0) / (p1 - p0)
            return _rgb_to_hex((r0 + (r1 - r0) * k,
                                g0 + (g1 - g0) * k,
                                b0 + (b1 - b0) * k))
    return pts[-1][1]


def _text_px(text: str, font_size: float = 12.0) -> float:
    """估算文本像素宽度（中文按全宽、ASCII 按半宽多一点），用于节点宽度。"""
    w = 0.0
    for ch in str(text):
        w += font_size + 1.5 if ord(ch) > 0x2E7F else font_size * 0.58
    return w


def _shorten_mean(mean_seg: str, own_seg: str) -> str:
    """纵向树叶子去重：均值段与本人值段重复的指标名只保留一次。

    如 own="量表 81 分"、mean="班级均值 量表 66 分" -> "班级均值 66 分"。
    均值段前缀（"班级均值" / "年级均值" / "均值"）始终保留；无法识别时原样返回。
    """
    m_tokens = mean_seg.split()
    if not m_tokens:
        return mean_seg
    head = m_tokens[0] if m_tokens[0].endswith("均值") else ""
    rest = m_tokens[1:] if head else list(m_tokens)
    own_tokens = own_seg.split()
    i = 0
    while i < len(rest) - 1 and i < len(own_tokens) and rest[i] == own_tokens[i]:
        i += 1
    body = " ".join(rest[i:]) if rest[i:] else (rest[-1] if rest else "")
    if not head:
        return body or mean_seg
    return f"{head} {body}".strip()


def _rounded_rect_path(x0: float, y0: float, x1: float, y1: float, r: float) -> str:
    """圆角矩形 SVG path（plotly shape type='path'，数据坐标）。"""
    r = min(r, (x1 - x0) / 2.0, (y1 - y0) / 2.0)
    return (
        f"M {x0 + r},{y0} L {x1 - r},{y0} Q {x1},{y0} {x1},{y0 + r} "
        f"L {x1},{y1 - r} Q {x1},{y1} {x1 - r},{y1} L {x0 + r},{y1} "
        f"Q {x0},{y1} {x0},{y1 - r} L {x0},{y0 + r} Q {x0},{y0} {x0 + r},{y0} Z"
    )


@st.cache_data(ttl=300, show_spinner=False, hash_funcs={dict: _hash_obj, list: _hash_obj})
def build_dimension_tree(root: dict, children: list, dim_color: str,
                         orientation: str = "h") -> go.Figure:
    """构建「维度分析树」节点连线层级图（3 层：根 -> 二级指标 -> 明细叶子）。

    root: {"title": 维度名, "line2": "55.0 分 · 合格", "hover": ...}
    children: [{
        "title": 二级指标名, "line2": "76.3 分 · 良好", "score": 76.3,
        "leaf": "GPA 3.42｜班级均值 3.28｜权重 30%"（None 表示该分支已折叠）,
        "hover": 节点悬停文本, "leaf_hover": 叶子悬停文本,
    }, ...]
    dim_color: 维度主题色（根节点深底白字；二级节点按得分取同色系色阶，深=高）。
    orientation: "h" 横向（根在左，默认）| "v" 纵向（根在上）。

    布局算法（tidy）：叶子等距排列，父节点位于子节点中点；连线四点肘形。
    """
    scale = build_hue_colorscale(dim_color)
    FONT = 12          # 正文字号（>=11）
    FONT_SMALL = 11
    PAD_X = 14         # 节点水平内边距
    NODE_H2 = 46       # 双行节点高（根 / 二级）
    NODE_H1 = 30       # 单行叶子高
    VGAP = 28          # 横向布局时子节点垂直间距（原 16，调大后更舒展）
    HGAP = 64          # 层级间距

    def _node_w(lines, font=FONT):
        return max(_text_px(t, font) for t in lines) + 2 * PAD_X

    # ---- 计算节点宽度 ----
    root_lines = [root["title"]] + ([root["line2"]] if root.get("line2") else [])
    root_w = _node_w(root_lines)
    child_ws = []
    for ch in children:
        c_lines = [ch["title"]] + ([ch["line2"]] if ch.get("line2") else [])
        child_ws.append(_node_w(c_lines))
    max_child_w = max(child_ws) if child_ws else 0

    # 纵向布局专用：叶子长标签拆行显示（第一行个人值、第二行对比值，行内「｜」
    # 分隔改为两段；≥6 个分支时进一步拆三行：个人值 / 均值 / 权重），均值段与
    # 本人值段重复的指标名只保留一次；相邻叶子间距不足时优先缩字号
    #（最小 10px），不允许重叠。
    leaf_font = FONT_SMALL
    leaf_lines_v: list = [None] * len(children)
    three_line = False
    if orientation == "v":
        V_BUDGET = 950.0    # 展开向像素预算（数据坐标 1:1 ≈ 像素）
        V_GAP = 10.0        # 相邻叶子最小水平间距
        three_line = len(children) >= 6
        while True:
            leaf_lines_v = []
            for ch in children:
                if not ch.get("leaf"):
                    leaf_lines_v.append(None)
                    continue
                parts = [p.strip() for p in ch["leaf"].split("｜") if p.strip()]
                if len(parts) >= 3:
                    own = parts[0]
                    mean = _shorten_mean(parts[1], own)
                    weight = parts[2]
                    if three_line:
                        # _add_node 多行布局中靠后的行在视觉上靠上
                        leaf_lines_v.append([weight, mean, own])
                    else:
                        leaf_lines_v.append([f"{mean} · {weight}", own])
                elif len(parts) == 2:
                    leaf_lines_v.append([parts[1], parts[0]])
                else:
                    leaf_lines_v.append(parts[:1])
            leaf_ws_v = [
                (_node_w(ll, leaf_font) if ll else 0) for ll in leaf_lines_v
            ]
            span_need = (
                sum(max(leaf_ws_v[i], child_ws[i]) for i in range(len(children)))
                + V_GAP * max(0, len(children) - 1) + 2 * 24.0
            )
            if span_need <= V_BUDGET or leaf_font <= 10:
                break
            leaf_font = max(10.0, leaf_font - 0.5)
    else:
        leaf_ws_v = [
            (_node_w([ch["leaf"]], FONT_SMALL) if ch.get("leaf") else 0)
            for ch in children
        ]
    leaf_ws = leaf_ws_v
    max_leaf_w = max(leaf_ws) if leaf_ws else 0

    # ---- tidy 布局：叶子等距（按节点实际厚度变距），父节点取子节点中点 ----
    # 展开向（span）节点厚度：横向=节点高度，纵向=max(叶子宽, 二级节点宽)（防重叠）
    child_pos = []      # 每个二级节点的 (厚度, 宽, has_leaf)
    for i, ch in enumerate(children):
        has_leaf = bool(ch.get("leaf"))
        if orientation == "v":
            thick = max(leaf_ws[i], child_ws[i])
        else:
            thick = (NODE_H1 if has_leaf else NODE_H2)
        child_pos.append((thick, child_ws[i], has_leaf))

    PAD_S = 24.0
    slot_gap = VGAP if orientation == "h" else 10.0   # 纵向用更小但足够的叶间距
    centers: list[float] = []   # 每个 slot 的展开向中心坐标（变距：相邻不重叠）
    acc = PAD_S
    for thick, _w, _hl in child_pos:
        acc += thick / 2.0
        centers.append(acc)
        acc += thick / 2.0 + slot_gap
    span_max = (centers[-1] + child_pos[-1][0] / 2.0 + PAD_S) if centers else 120.0
    root_sc = (centers[0] + centers[-1]) / 2.0 if centers else 60.0

    # ---- 抽象坐标：depth（层级向）与 slot（展开向），最后按方向映射到 (x, y) ----
    # depth 坐标（横向时向右；纵向时向下）。
    # 注意：节点在 depth 向的厚度 = 横向为节点宽，纵向为节点高（NODE_H2）。
    if orientation == "v":
        LEAF_H_V = 58.0 if three_line else float(NODE_H2)
        d0 = 20.0
        d1 = d0 + NODE_H2 + HGAP
        d2 = d1 + NODE_H2 + HGAP
        depth_max = d2 + LEAF_H_V + 20.0
    else:
        d0, d1 = 20.0, 20.0 + root_w + HGAP
        d2 = d1 + max_child_w + HGAP
        depth_max = d2 + max_leaf_w + 30.0
    # 根 / 二级节点在 depth 向的实际厚度（连线端点用，避免线停在文字边缘）
    root_dthick = root_w if orientation == "h" else float(NODE_H2)

    def _sc(slot_idx):
        """slot 序号 -> 展开向中心坐标。"""
        return centers[slot_idx]

    def _map(d, s):
        """(depth, span) -> (x, y)；纵向时交换并翻转 y（根在上）。"""
        if orientation == "v":
            return s, d
        return d, s

    shapes, annotations = [], []
    hx, hy, htext = [], [], []   # 悬停热点
    edge_x, edge_y = [], []

    def _add_node(d_left, sc, w, h, lines, fill, font_color, hover,
                  fonts=(FONT, FONT_SMALL), bold_line=0):
        # 节点矩形在 depth 向占 w，在 span 向占 h（纵向时反过来）
        if orientation == "v":
            x0r, y0r = _map(d_left, sc - w / 2)
            x1r, y1r = _map(d_left + h, sc + w / 2)
        else:
            x0r, y0r = _map(d_left, sc - h / 2)
            x1r, y1r = _map(d_left + w, sc + h / 2)
        x0r, x1r = min(x0r, x1r), max(x0r, x1r)
        y0r, y1r = min(y0r, y1r), max(y0r, y1r)
        shapes.append(dict(
            type="path",
            path=_rounded_rect_path(x0r, y0r, x1r, y1r, 9),
            fillcolor=fill, line=dict(color=C.THEME_CARD_BORDER, width=1),
        ))
        cx, cy = (x0r + x1r) / 2.0, (y0r + y1r) / 2.0
        if len(lines) == 1:
            annotations.append(dict(
                x=cx, y=cy, text=lines[0], showarrow=False,
                font=dict(size=fonts[1], color=font_color),
                xanchor="center", yanchor="middle",
            ))
        elif len(lines) == 2:
            # 注意 y 轴反向：cy-10 为视觉上行，cy+8 为视觉下行
            text0 = f"<b>{lines[0]}</b>" if bold_line == 0 else lines[0]
            text1 = f"<b>{lines[1]}</b>" if bold_line == 1 else lines[1]
            annotations.append(dict(
                x=cx, y=cy + 8, text=text0, showarrow=False,
                font=dict(size=fonts[0], color=font_color),
                xanchor="center", yanchor="middle",
            ))
            annotations.append(dict(
                x=cx, y=cy - 10, text=text1, showarrow=False,
                font=dict(size=fonts[1], color=font_color),
                xanchor="center", yanchor="middle",
            ))
        else:
            # 三行叶子（纵向 ≥6 分支）：靠后的行视觉靠上（y 轴反向）
            offs = (14, -2, -18)
            for li, t in enumerate(lines[:3]):
                annotations.append(dict(
                    x=cx, y=cy + offs[li],
                    text=f"<b>{t}</b>" if li == bold_line else t,
                    showarrow=False,
                    font=dict(size=fonts[min(li, len(fonts) - 1)], color=font_color),
                    xanchor="center", yanchor="middle",
                ))
        if hover:
            # 三个隐形热点覆盖节点大部分区域，悬停显示完整数值
            for f in (0.25, 0.5, 0.75):
                if orientation == "v":
                    px, py = _map(d_left + h / 2.0, sc - w / 2.0 + w * f)
                else:
                    px, py = _map(d_left + w * f, sc)
                hx.append(px)
                hy.append(py)
                htext.append(hover)

    def _add_elbow(da, sa, db, sb):
        """四点肘形连线（depth 向先走一半，再跨 span，再走另一半）。"""
        dm = (da + db) / 2.0
        for px, py in [_map(da, sa), _map(dm, sa), _map(dm, sb), _map(db, sb)]:
            edge_x.append(px)
            edge_y.append(py)
        edge_x.append(None)
        edge_y.append(None)

    # ---- 根节点（父节点位于子节点中点）----
    _add_node(d0, root_sc, root_w, NODE_H2, root_lines,
              _scale_hex(scale, 1.0), C.THEME_CARD_BG, root.get("hover"))

    # ---- 二级节点 + 叶子 ----
    for slot, ((_thick, cw, has_leaf), ch) in enumerate(zip(child_pos, children)):
        sc = _sc(slot)
        score = ch.get("score")
        t = 0.0 if score is None else float(score) / 100.0
        fill = _scale_hex(scale, t)
        font_color = C.THEME_CARD_BG if (score is not None and float(score) >= 55) else C.THEME_TEXT
        c_lines = [ch["title"]] + ([ch["line2"]] if ch.get("line2") else [])
        _add_node(d1, sc, cw, NODE_H2, c_lines, fill, font_color, ch.get("hover"))
        _add_elbow(d0 + root_dthick, root_sc, d1, sc)
        if has_leaf:
            # 纵向：叶子为两行文本节点（个人值 / 对比值），字号可按需缩至 10px
            child_dthick = cw if orientation == "h" else float(NODE_H2)
            if orientation == "v":
                l_lines = leaf_lines_v[slot] or [ch["leaf"]]
                _add_node(d2, sc, leaf_ws[slot], LEAF_H_V,
                          l_lines, C.THEME_PRIMARY_LIGHT, C.THEME_TEXT,
                          ch.get("leaf_hover") or ch.get("hover"),
                          fonts=(leaf_font, leaf_font),
                          bold_line=len(l_lines) - 1)
            else:
                lw = _node_w([ch["leaf"]], FONT_SMALL)
                _add_node(d2, sc, lw, NODE_H1,
                          [ch["leaf"]], C.THEME_PRIMARY_LIGHT, C.THEME_TEXT,
                          ch.get("leaf_hover") or ch.get("hover"),
                          fonts=(FONT_SMALL, FONT_SMALL))
            _add_elbow(d1 + child_dthick, sc, d2, sc)

    fig = go.Figure()
    # 连线（在最底层）
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color=C.THEME_CARD_BORDER, width=2),
        hoverinfo="skip", showlegend=False,
    ))
    # 悬停热点
    fig.add_trace(go.Scatter(
        x=hx, y=hy, mode="markers",
        marker=dict(size=26, color=C.THEME_PRIMARY, opacity=0),
        text=htext, hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))
    for s in shapes:
        fig.add_shape(s)
    for ann in annotations:
        fig.add_annotation(ann)

    if orientation == "v":
        # 顶部 / 底部加大边距：根节点标签与最深叶子标签均完整可见、不被裁切
        V_PAD_T, V_PAD_B = 50.0, 50.0
        fig.update_xaxes(range=[-20, span_max + 20], visible=False)
        fig.update_yaxes(range=[depth_max + V_PAD_B, -V_PAD_T], visible=False,
                         scaleanchor="x", scaleratio=1)
        fig.update_layout(
            height=max(430, depth_max + V_PAD_T + V_PAD_B + 60),
            margin=dict(l=20, r=20, t=30, b=30),
        )
    else:
        fig.update_xaxes(range=[-10, depth_max + 10], visible=False)
        fig.update_yaxes(range=[span_max + 10, -10], visible=False,
                         scaleanchor="x", scaleratio=1)
        fig.update_layout(
            height=max(300, span_max + 40),
            margin=dict(l=20, r=20, t=20, b=20),
        )
    fig.update_layout(
        paper_bgcolor=C.THEME_CARD_BG,
        plot_bgcolor=C.THEME_CARD_BG,
        font=dict(color=C.THEME_TEXT),
        dragmode="pan",
    )
    return fig


# ---------------------------------------------------------------------------
# 同类型对比条（本人 vs 同成长类型均值 vs 年级均值）
# ---------------------------------------------------------------------------
def build_compare_bar(self_label: str, self_val: float,
                      ref_pairs: list, dim_color: str,
                      title: str = "") -> go.Figure:
    """横向对比条形图：本人（维度主题色）+ 参考系列（辅助色）。

    ref_pairs: [(标签, 数值), ...]，如 [("同成长类型均值", 70.2), ("年级均值", 68.5)]。
    """
    labels = [self_label] + [p[0] for p in ref_pairs]
    vals = [float(self_val)] + [float(p[1]) for p in ref_pairs]
    ref_colors = [C.CHART_ACCENT, C.CHART_NEUTRAL, C.CHART_SECONDARY]
    colors = [dim_color] + [ref_colors[i % len(ref_colors)] for i in range(len(ref_pairs))]
    fig = go.Figure(go.Bar(
        y=labels, x=vals, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" for v in vals],
        textposition="outside",
    ))
    layout = dict(
        height=200,
        # y 轴类目标签（本人 / 同类型均值（××型）/ 年级均值）较长，交给 automargin
        # 自适应左侧宽度，避免第一个类目标签被裁切；容器不留负边距。
        margin=dict(l=10, r=48, t=30 if title else 10, b=10),
        xaxis=dict(range=[0, 100], gridcolor=C.THEME_GRID, title="得分"),
        yaxis=dict(autorange="reversed", automargin=True),
        paper_bgcolor=C.THEME_CARD_BG,
        plot_bgcolor=C.THEME_CARD_BG,
        font=dict(color=C.THEME_TEXT),
        showlegend=False,
    )
    if title:
        # 仅在确有标题时写入 layout.title：置 None 会被 plotly.js 渲染成 "undefined"
        layout["title"] = dict(text=title, font=dict(size=14, color=C.THEME_TEXT))
    fig.update_layout(**layout)
    return fig
