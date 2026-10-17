"""analysis_core 数据清洗模块。

职责（作品书 2.2 / 3.3 节）：
1. 读取原始学情记录（dict / DataFrame）。
2. 对数值字段设定上下界并裁剪越界值（55 个数值字段）。
3. 缺失值填充：活动 / 参与类指标缺失视为「未参与」按 0 处理；
   其余字段用列中位数填充（整列全缺时退化为下界）。

展示层脱敏在 local_database.orm_ops 完成，本模块不做脱敏。
"""

from __future__ import annotations

import pandas as pd

# 数值型字段及合理取值范围（用于异常值截断 / 缺失填充）
NUMERIC_FIELDS = {
    "gpa": (0.0, 4.0),
    "fail_count": (0, 20),
    "library_visits": (0, 500),
    "study_hours": (0, 1000),
    "attendance_rate": (0.0, 1.0),
    "homework_timely_rate": (0.0, 1.0),
    "credit_progress": (0.0, 1.0),
    "competition_awards": (0, 100),
    "psy_score": (0, 100),
    "psy_counsel_times": (0, 50),
    "mood_check": (0, 100),
    "stress_index": (0, 100),
    "physical_test_score": (0, 100),
    "sleep_regularity": (0.0, 1.0),
    "second_class_count": (0, 200),
    "volunteer_hours": (0, 1000),
    "club_role": (0, 1),
    "social_connection": (0, 100),
    "canteen_regularity": (0.0, 1.0),
    "dorm_regularity": (0.0, 1.0),
    "consumption_daily": (0.0, 100.0),
    "breakfast_rate": (0.0, 1.0),
    "leave_abnormal": (0, 50),
    "political_status": (0, 100),
    "discipline_violation": (0, 10),
    "integrity_record": (0, 1),
    "collective_participation": (0.0, 1.0),
    "grit": (0, 100),
    "self_control": (0, 100),
    "engagement": (0, 100),
    "metacognition": (0, 100),
    "self_perception": (0, 100),
    "learning_motivation": (0, 100),
    "continuous_absence": (0, 200),
    "night_not_home": (0, 200),
    "consumption_drop": (0.0, 1.0),
    "grade_drop": (0.0, 1.0),
    # 手册硬规则原始字段（config_setting.HANDBOOK_RULES）
    "thesis_gpa": (0.0, 4.0),
    "fail_grade_consecutive": (0, 20),
    "fail_grade_total": (0, 20),
    "campus_run_valid_count": (0, 300),
    "campus_run_year_km": (0, 1000.0),
    "campus_run_ratio": (0.0, 1.0),
    "pe_course_score": (0.0, 100.0),
    "run_violation_count": (0, 20),
    "course_absence_ratio": (0.0, 1.0),
    "homework_miss_ratio": (0.0, 1.0),
    "absence_hours": (0.0, 400.0),
    "leave_days": (0, 90),
    "inactive_weeks_consecutive": (0, 40),
    "inactive_weeks_total": (0, 80),
    "unregistered_weeks": (0, 40),
    "labor_hours": (0.0, 1000.0),
    "social_practice_done": (0, 1),
    "academic_tech_score": (0.0, 120.0),
    "dorm_hygiene_score": (0.0, 100.0),
    "physical_test_grade_score": (0.0, 100.0),
    "discipline_penalty_score": (0.0, 100.0),
}

# 活动 / 参与 / 奖惩类「计数」字段：缺失即「未参与 / 未发生」，按 0 填充
# （作品书 3.3：活动类指标缺失视为未参与按 0 处理）
ACTIVITY_ZERO_FIELDS = {
    "competition_awards", "second_class_count", "volunteer_hours", "club_role",
    "social_practice_done", "labor_hours", "academic_tech_score",
    "fail_count", "fail_grade_consecutive", "fail_grade_total",
    "psy_counsel_times", "run_violation_count", "absence_hours", "leave_days",
    "continuous_absence", "night_not_home", "leave_abnormal",
    "inactive_weeks_consecutive", "inactive_weeks_total", "unregistered_weeks",
    "discipline_violation",
}


def clean_records(records: list[dict]) -> pd.DataFrame:
    """将原始记录列表清洗为规整 DataFrame。

    - 超出合理范围的异常值截断到边界；
    - 活动 / 参与类缺失按 0 填充；
    - 其余数值缺失用列中位数填充（整列全缺时取下界）。
    """
    df = pd.DataFrame(records)
    for col, (lo, hi) in NUMERIC_FIELDS.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.clip(lower=lo, upper=hi)
        if col in ACTIVITY_ZERO_FIELDS:
            s = s.fillna(0)
        else:
            median = s.median()
            s = s.fillna(median if not pd.isna(median) else lo)
        df[col] = s
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """对已存在的 DataFrame 执行同样的清洗逻辑。"""
    records = df.to_dict(orient="records")
    return clean_records(records)
