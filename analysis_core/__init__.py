"""analysis_core 模块入口。"""

from analysis_core.data_clean import clean_records, clean_dataframe
from analysis_core.normalize import (
    minmax_scale,
    normalize_score_frame,
    normalize_features,
)
from analysis_core.kmeans_model import (
    choose_best_k,
    fit_grade,
    fit_1d_levels,
    level_gap,
    name_cluster_types,
)
from analysis_core.risk_level import compute_risk_score
from analysis_core.pipeline import run_pipeline, get_personalized_advice

__all__ = [
    "clean_records",
    "clean_dataframe",
    "minmax_scale",
    "normalize_score_frame",
    "normalize_features",
    "choose_best_k",
    "fit_grade",
    "fit_1d_levels",
    "level_gap",
    "name_cluster_types",
    "compute_risk_score",
    "run_pipeline",
    "get_personalized_advice",
]
