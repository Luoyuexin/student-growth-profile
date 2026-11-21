"""analysis_core K-means 聚类建模模块

两级聚类：
- 分型（多维）：对六维得分向量做多维 K-means，K 在 [2,8] 内由
  肘部法则（SSE，权重 0.4）与轮廓系数（权重 0.6）加权联合确定，
  分年级独立建模；簇再由规则赋予可读的「成长类型」名称。
- 分级（一维）：对每个维度（及综合分）的复合得分做一维 K-means，
  质心天然有序，单调映射为「待改进 / 合格 / 良好 / 优秀」，
  相邻质心中点作为等级边界，支撑「距上一等级还差多少分」。

聚类参数（搜索范围、随机种子、n_init）集中在 config_setting。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config_setting import (
    DIMENSION_LABELS,
    DIMENSION_ORDER,
    LEVEL_LABELS_BY_K,
    get_grade_cluster_config,
)

ELBOW_WEIGHT = 0.4      # 肘部法则权重（作品书 3.4.1）
SILHOUETTE_WEIGHT = 0.6  # 轮廓系数权重（高于肘部）


def choose_best_k(X: np.ndarray, k_range=(2, 8), random_state: int = 42):
    """在 k_range 内搜索最优 K（多维分型）。

    综合肘部法则（簇内平方和 SSE，越小越好）与轮廓系数（越大越好），
    两者在「有效候选 K」内归一化后按 0.4 / 0.6 加权，取总分最大的 K。

    轮廓系数要求 1 < K < n-1；不满足前提的 K 置为 -1 并直接淘汰
    返回 (best_k, details)。
    """
    if X.shape[0] < 2:
        return 1, {}
    k_min, k_max = k_range
    k_max = min(k_max, X.shape[0] - 1)
    if k_max < k_min:
        k_min = k_max = max(1, X.shape[0] - 1)

    tested: List[int] = []
    inertias: List[float] = []
    silhouettes: List[float] = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        tested.append(k)
        inertias.append(float(km.inertia_))
        # 轮廓系数要求 1 < k < n_samples - 1
        if 1 < k < X.shape[0] - 1 and len(set(labels)) > 1:
            sil = float(silhouette_score(X, labels))
        else:
            sil = -1.0  # 不满足计算前提 -> 自动淘汰
        silhouettes.append(sil)

    inertias_arr = np.array(inertias, dtype=float)
    sil_arr = np.array(silhouettes, dtype=float)
    valid = sil_arr >= 0.0  # 有效候选

    # 归一化只在有效候选之间进行，避免 -1 拉偏 min/max
    def _norm_up(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """越大越好归一化；无效候选为 0。"""
        out = np.zeros_like(arr)
        if mask.any():
            sub = arr[mask]
            if sub.max() - sub.min() > 1e-9:
                out[mask] = (sub - sub.min()) / (sub.max() - sub.min())
            else:
                out[mask] = 1.0
        return out

    def _norm_down(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """越小越好归一化（SSE）；无效候选为 0。"""
        out = np.zeros_like(arr)
        if mask.any():
            sub = arr[mask]
            if sub.max() - sub.min() > 1e-9:
                out[mask] = (sub.max() - sub) / (sub.max() - sub.min())
            else:
                out[mask] = 1.0
        return out

    if valid.any():
        inertia_n = _norm_down(inertias_arr, valid)
        sil_n = _norm_up(sil_arr, valid)
        score = ELBOW_WEIGHT * inertia_n + SILHOUETTE_WEIGHT * sil_n
        score = np.where(valid, score, -np.inf)  # 无效 K 淘汰
    else:
        # 样本量极小、全部 K 都不满足轮廓系数前提时，仅按肘部法则决策
        inertia_n = _norm_down(inertias_arr, np.ones_like(valid))
        score = inertia_n

    best_idx = int(np.argmax(score))
    best_k = tested[best_idx]
    details = {
        "tested_k": tested,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "scores": [None if not np.isfinite(s) else float(s) for s in score],
        "best_k": best_k,
    }
    return best_k, details


def fit_grade(
    df,
    grade: int,
    feature_cols: List[str],
    return_details: bool = False,
):
    """对单个年级的数据做多维分型聚类并返回簇标签。

    df: 含 feature_cols 的 DataFrame（已在年级内 Min-Max 归一化）。
    返回 labels（以及可选的 (best_k, details)）。
    """
    cfg = get_grade_cluster_config(grade)
    X = df[feature_cols].to_numpy(dtype=float)

    if cfg.auto_tune_k:
        best_k, details = choose_best_k(X, k_range=cfg.k_search_range,
                                        random_state=cfg.random_state)
    else:
        best_k, details = cfg.k, {"best_k": cfg.k, "tested_k": [cfg.k]}

    km = KMeans(n_clusters=best_k, random_state=cfg.random_state, n_init=10)
    labels = km.fit_predict(X)
    if return_details:
        return labels, best_k, details
    return labels


# ---------------------------------------------------------------------------
# 一维分级聚类
# ---------------------------------------------------------------------------
def fit_1d_levels(
    values: Sequence[float],
    preferred_k: int = 4,
    random_state: int = 42,
) -> Dict:
    """对一维复合得分聚类，输出有序等级与边界。

    - preferred_k 优先取 4（与手册体测「优秀 / 良好 / 及格 / 不及格」四级对齐）；
    - 簇质心升序 c1<c2<…<cK，按单调顺序映射等级（表 3-1）；
    - 边界阈值取相邻质心中点，用于输出「距上一等级还差多少分」。

    返回：
      {
        "k": int,
        "labels":  [每个学生的等级文本],
        "ranks":   [每个学生的等级序号 0..K-1],
        "centers": [升序质心],
        "boundaries": [相邻质心中点，长度 K-1],
        "level_names": [由低到高的等级名],
      }
    """
    x = np.asarray(list(values), dtype=float).reshape(-1, 1)
    n = x.shape[0]
    k = max(1, min(int(preferred_k), n))
    level_names = LEVEL_LABELS_BY_K.get(k, [f"等级{i + 1}" for i in range(k)])

    if k == 1:
        return {
            "k": 1, "labels": [level_names[0]] * n, "ranks": [0] * n,
            "centers": [float(x.mean())], "boundaries": [],
            "level_names": level_names,
        }

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    raw_labels = km.fit_predict(x)
    centers = km.cluster_centers_.reshape(-1)

    # 质心升序 -> 原始簇编号映射到等级序号
    order = np.argsort(centers)            # order[rank] = 原簇号
    rank_of_cluster = {int(cluster): int(rank) for rank, cluster in enumerate(order)}
    centers_sorted = [float(centers[c]) for c in order]
    boundaries = [
        round(float((centers_sorted[i] + centers_sorted[i + 1]) / 2.0), 2)
        for i in range(k - 1)
    ]

    ranks = [rank_of_cluster[int(lab)] for lab in raw_labels]
    labels = [level_names[r] for r in ranks]
    return {
        "k": k,
        "labels": labels,
        "ranks": ranks,
        "centers": [round(c, 2) for c in centers_sorted],
        "boundaries": boundaries,
        "level_names": level_names,
    }


def level_gap(score: float, boundaries: Sequence[float], rank: int) -> Optional[float]:
    """按等级边界计算「距上一等级还差多少分」；已在最高级返回 None。"""
    if rank is None or rank < 0 or rank >= len(boundaries):
        return None
    gap = float(boundaries[rank]) - float(score)
    return round(max(0.0, gap), 1)


# ---------------------------------------------------------------------------
# 多维簇 -> 可读成长类型（作品书 5.2：全面发展型 / 学业偏科型 / 实践活跃型 …）
# ---------------------------------------------------------------------------
# 最低维度命中的类型名（身心健康 / 品德相关措辞已做去标签化处理）
_WEAKEST_TYPE = {
    "study": "学业偏科型",
    "health": "身心待进型",
    "social": "社交待拓型",
    "life": "生活适应待进型",
    "morality": "品德修养提升型",
    "non_cog": "潜能待激发型",
}


def name_cluster_types(
    centroids: np.ndarray,
    dim_keys: Optional[List[str]] = None,
) -> List[str]:
    """根据各簇的六维质心给出可读「成长类型」名称。

    centroints: shape (K, 6)，顺序与 dim_keys 一致，使用**原始维度分**
                （0~100，不经年级内归一化）以保证语义可解释。
    规则（保守、可解释，不使用心理类负面措辞）：
      1. 六维全部 ≥ 80………………全面发展型
      2. 均分 < 60 ………………… 奋起追赶型
      3. 社会参与突出（最高且高出均值 8 分、且 ≥72）…实践活跃型
      4. 学业突出（≥72 且高出均值 8 分）且结构不均衡 …学业引领型
      5. 最弱维确实偏低（< 60 且低于均值 8 分）……按最弱维度命名
      6. 整体较好（均分 ≥ 72）…… 均衡良好型
      7. 兜底 ……………………… 平稳成长型
    注意：模拟数据中品德素养普遍偏高，单维高企会拉大标准差，
    但只要没有维度过低就不按「弱项」命名，避免给较优群体贴负向标签。
    同名簇自动追加「·2 / ·3」后缀，保证一个年级内类型名唯一。
    """
    keys = dim_keys or DIMENSION_ORDER
    mat = np.asarray(centroids, dtype=float).reshape(-1, len(keys))
    names: List[str] = []
    for row in mat:
        mean = float(row.mean())
        max_i = int(np.argmax(row))
        min_i = int(np.argmin(row))
        top_dim = keys[max_i]
        weak_dim = keys[min_i]
        weak_val = float(row[min_i])
        top_val = float(row[max_i])

        if float(row.min()) >= 80:
            name = "全面发展型"
        elif mean < 60:
            name = "奋起追赶型"
        elif top_dim == "social" and top_val >= 72 and top_val - mean >= 8:
            name = "实践活跃型"
        elif top_dim == "study" and top_val >= 72 and top_val - mean >= 8:
            name = "学业引领型"
        elif weak_val < 60 and mean - weak_val >= 8:
            name = _WEAKEST_TYPE.get(weak_dim, "平稳成长型")
        elif mean >= 72:
            name = "均衡良好型"
        else:
            name = "平稳成长型"
        names.append(name)

    # 同名去重（同年级内簇名唯一，便于展示与统计）
    used: Dict[str, int] = {}
    uniq: List[str] = []
    for name in names:
        if name not in used:
            used[name] = 1
            uniq.append(name)
        else:
            used[name] += 1
            uniq.append(f"{name}·{used[name]}")
    return uniq
