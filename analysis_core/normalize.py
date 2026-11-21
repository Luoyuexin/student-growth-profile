"""analysis_core 特征无量纲归一化模块。

实现标准：
- 二级指标的 0~100 归一化（含负向指标反向）在 scoring.py 按
  config_setting.DIMENSION_SUBSCORES 的区间配置完成；
- 本模块负责聚类前的 Min-Max 归一化：把六维（或综合）得分在
  **年级内**映射到 [0,1]。各维得分本身已在固定区间、分布规整，
  故选 Min-Max 而非 Z-score。
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


def minmax_scale(values: Iterable[float], reverse: bool = False) -> pd.Series:
    """单列 Min-Max 归一化到 [0,1]；常数列给 0.5；reverse=True 时反向。"""
    s = pd.Series(list(values), dtype=float)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-9:
        norm = pd.Series(np.full(len(s), 0.5), index=s.index)
    else:
        norm = (s - lo) / (hi - lo)
    if reverse:
        norm = 1.0 - norm
    return norm.clip(0.0, 1.0)


def normalize_score_frame(
    df: pd.DataFrame,
    columns: Sequence[str],
    reverse_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """对给定得分列逐列做年级内 Min-Max，新增 ``<col>_n`` 列（不改原列）。"""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        out[f"{col}_n"] = minmax_scale(out[col], reverse=col in reverse_cols).to_numpy()
    return out


# 兼容性别名：部分旧脚本/文档按 normalize_features 称呼本函数。
def normalize_features(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    reverse_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """旧接口名兼容包装。

    旧实现按「学习/生活/活动/心理」四组原始字段归一化；六维体系落地后
    统一改为对传入的得分列（默认六维 score_*）做 Min-Max。
    """
    if columns is None:
        from config_setting import DIMENSION_ORDER
        columns = [f"score_{d}" for d in DIMENSION_ORDER]
    return normalize_score_frame(df, columns, reverse_cols)
