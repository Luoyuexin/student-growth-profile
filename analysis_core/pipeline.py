"""analysis_core 分析流水线编排（作品书 3.1 数据加工流水线）。

手册规则富化 -> 清洗与异常裁剪 -> 六维与综合得分 -> 年级内归一化 ->
多维 K-means 分型（+成长类型命名）-> 分维度一维 K-means 分级（+等级边界）
-> 风险判定 -> 画像落库，由统一入口一次执行完成全量刷新。
"""

from __future__ import annotations

import json

import pandas as pd

from analysis_core import data_clean, handbook_rules, kmeans_model, risk_level
from analysis_core import scoring
from analysis_core.normalize import normalize_score_frame
from local_database import orm_ops
from config_setting import (
    DIMENSION_ORDER,
    DIMENSION_LEVEL_K,
)

# 多维分型所用的特征：六个维度的得分（聚类前在年级内 Min-Max）
CLUSTER_FEATURES = [f"score_{d}" for d in DIMENSION_ORDER]


def run_pipeline(records: list[dict] | None = None, save: bool = True) -> pd.DataFrame:
    """执行完整分析流水线。

    records: 若不提供则从数据库读取全部学生。
    返回带分析结果的 DataFrame（attrs['cluster_details'] 含定 K 过程）。
    """
    if records is None:
        ids = orm_ops.get_all_student_ids()
        records = [orm_ops.get_raw_student(i) for i in ids if i]
    records = [r for r in records if r]

    # 1) 手册硬规则：先注入派生指标（校园跑完成率 / 体测等级分 / 处分折算分）
    records = [handbook_rules.enrich_record(r) for r in records]

    # 2) 清洗（越界裁剪；活动类缺失补 0，其余中位数）
    df = data_clean.clean_records(records)
    clean_rows = df.to_dict(orient="records")

    # 3) 六维评分（二级 -> 维度 -> 综合），基于清洗后的记录保证口径一致
    dim_scores_list = [scoring.compute_dimension_scores(r) for r in clean_rows]
    for d in DIMENSION_ORDER:
        df[f"score_{d}"] = [float(ds[d][0]) for ds in dim_scores_list]
    df["composite_score"] = [scoring.compute_composite(ds) for ds in dim_scores_list]

    # 4) 分年级独立建模
    df["cluster_label"] = -1
    df["cluster_type"] = None
    df["level_overall"] = None
    for d in DIMENSION_ORDER:
        df[f"level_{d}"] = None

    cluster_details = {}
    for grade, gdf in df.groupby("grade"):
        if gdf.empty:
            continue
        grade = int(grade)
        idx = gdf.index

        # 4.1 年级内 Min-Max 归一化后做多维分型
        gnorm = normalize_score_frame(gdf, CLUSTER_FEATURES)
        labels, best_k, details = kmeans_model.fit_grade(
            gnorm, grade, [f"{c}_n" for c in CLUSTER_FEATURES], return_details=True
        )
        # fit_grade 内部按 feature_cols 取列；归一化列名已是 <col>_n
        df.loc[idx, "cluster_label"] = labels

        # 用**原始维度分**质心命名成长类型（归一化质心不可解释）
        centroids = (
            gdf[CLUSTER_FEATURES]
            .assign(_lab=labels)
            .groupby("_lab")[CLUSTER_FEATURES]
            .mean()
            .sort_index()
            .to_numpy(dtype=float)
        )
        type_names = kmeans_model.name_cluster_types(centroids, DIMENSION_ORDER)
        df.loc[idx, "cluster_type"] = [type_names[int(lab)] for lab in labels]
        orm_ops.upsert_cluster_meta(
            grade, "multi", int(best_k),
            centers=[[round(float(v), 2) for v in row] for row in centroids],
            boundaries=[],
            labels=type_names,
            # 定 K 过程明细一并落库，供教师端配置页绘制「定 K 过程曲线图」
            details=details,
        )

        # 4.2 综合分一维聚类 -> 综合等级与边界
        overall = kmeans_model.fit_1d_levels(
            gdf["composite_score"], preferred_k=DIMENSION_LEVEL_K,
            random_state=42,
        )
        df.loc[idx, "level_overall"] = overall["labels"]
        orm_ops.upsert_cluster_meta(
            grade, "composite", overall["k"],
            centers=overall["centers"], boundaries=overall["boundaries"],
            labels=overall["level_names"],
        )

        # 4.3 每个维度一维聚类 -> 维度等级与边界
        for d in DIMENSION_ORDER:
            res = kmeans_model.fit_1d_levels(
                gdf[f"score_{d}"], preferred_k=DIMENSION_LEVEL_K, random_state=42
            )
            df.loc[idx, f"level_{d}"] = res["labels"]
            orm_ops.upsert_cluster_meta(
                grade, f"dim:{d}", res["k"],
                centers=res["centers"], boundaries=res["boundaries"],
                labels=res["level_names"],
            )

        cluster_details[grade] = {"k": int(best_k), "details": details}

    # 5) 风险判定（基于 7 项阈值对 5 类信号累加计分）
    risk_levels, risk_scores = [], []
    for row in clean_rows:
        sc, lv, _ = risk_level.compute_risk_score(row)
        risk_scores.append(sc)
        risk_levels.append(lv)
    df["risk_level"] = risk_levels
    df["risk_score"] = risk_scores

    # 6) 画像落库（字段脱敏在对外读取接口完成）
    if save:
        for pos, idx in enumerate(df.index):
            ds = dim_scores_list[pos]
            src = records[pos]
            # 手册规则判定：只记录「需要提醒」的条目与命中的规则编号
            hb = handbook_rules.judge(src)
            payload = {
                "student_id": df.at[idx, "student_id"],
                "cluster_label": int(df.at[idx, "cluster_label"]),
                "cluster_type": df.at[idx, "cluster_type"],
                "level_overall": df.at[idx, "level_overall"],
                "composite_score": float(df.at[idx, "composite_score"]),
                "risk_level": df.at[idx, "risk_level"],
                "risk_score": float(df.at[idx, "risk_score"]),
                "campus_run_valid_count": int(src.get("campus_run_valid_count") or 0),
                "campus_run_year_km": float(src.get("campus_run_year_km") or 0.0),
                "handbook_hits": json.dumps(hb["hits"], ensure_ascii=False),
                "handbook_reminders": json.dumps(hb["reminders"], ensure_ascii=False),
                "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            for d in DIMENSION_ORDER:
                payload[f"score_{d}"] = float(ds[d][0])
                payload[f"level_{d}"] = df.at[idx, f"level_{d}"]
            orm_ops.upsert_student(payload)

    df.attrs["cluster_details"] = cluster_details
    return df


def get_personalized_advice(row: dict) -> list[str]:
    """基于六维得分与风险等级，生成个性化提升建议（规则兜底）。"""
    dim_scores = scoring.compute_dimension_scores(row)
    return scoring.get_personalized_advice(dim_scores, row.get("risk_level", "正常"))
