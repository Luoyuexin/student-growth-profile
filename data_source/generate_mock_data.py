"""data_source 模拟学情数据生成脚本

生成贴近真实分布的模拟学情数据，写入 CSV（data_source/raw_student_data.csv）
并可选写入 SQLite 数据库，便于本地一键演示。

数据维度覆盖：六维 32 项二级指标（学业发展 / 身心健康 / 社会参与 /
生活适应 / 品德素养 / 非认知能力）+ 五类风险信号 + 12 条手册硬规则原始字段。
共 240 名学生（2023 / 2024 两级、4 个教学班，每班 60 人），固定随机种子可复现。
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys

import numpy as np

# 将项目根目录加入 sys.path，保证脚本可独立运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_database import db_init, orm_ops
from local_database.db_init import DEFAULT_STUDENT_PASSWORD, hash_password
from analysis_core import pipeline as analysis_pipeline
from config_setting import HANDBOOK_RULES, cohort_to_level

# 班级配置：年级 -> 班级列表（与 config_setting.TEACHER_CLASS_SCOPE 对应）
GRADE_CLASSES = {
    2023: ["2023级-计算机1班", "2023级-计算机2班"],
    2024: ["2024级-软件1班", "2024级-软件2班"],
}

# 班级编号映射，保证学号为纯数字（与界面提示示例 202310100001 一致）
# 格式：{年级: {班级名: 班级代码}}
CLASS_CODES = {
    2023: {"2023级-计算机1班": "101", "2023级-计算机2班": "102"},
    2024: {"2024级-软件1班": "201", "2024级-软件2班": "202"},
}

SURNAMES = ["张", "王", "李", "赵", "刘", "陈", "杨", "黄", "周", "吴"]
GIVEN = ["伟", "芳", "娜", "敏", "静", "强", "磊", "洋", "艳", "勇", "杰", "娟", "涛", "明", "霞"]

# 校园跑逐次记录的学年周数（1~32 周，上/下两个学期）
RUN_YEAR_WEEKS = int(HANDBOOK_RULES["P2"]["year_weeks"][1])

# 手册规则的确定性样本（key=seq）。用固定序号而非随机，保证每条规则在演示数据中
# 都必然有触发样本，不随随机流变化漂移。
# 班级分布：1~60 2023级-计算机1班 / 61~120 2023级-计算机2班
#           121~180 2024级-软件1班 / 181~240 2024级-软件2班
HANDBOOK_CASES = {
    # A1 学业警示：课程最高平均 GPA < 2.00
    "gpa_warn": {5: 1.86, 65: 1.72, 125: 1.90, 185: 1.95},
    # A5 学士学位：毕设 GPA < 2.00（仅大四）
    "thesis_gpa": {6: 1.80, 66: 1.95},
    # A2 年级晋级：必修学分完成率低于本年级晋级线
    "credit_low": {128: 0.68, 188: 0.71},
    # A3 退学红线：连续 2 次 / 累计 3 次未晋级（连续次数, 累计次数）
    "fail_grade": {35: (2, 2), 95: (1, 3), 155: (2, 3)},
    # A4 毕业条件：体测大一大二两学年平均 < 60
    "fitness_low": {12: {"1": 52.0, "2": 55.5}, 72: {"1": 57.0, "2": 51.0}},
    # S1 课程考核资格：缺课 > 1/3 、欠交作业 > 1/4
    "course_absence": {18: 0.42, 78: 0.38},
    "homework_miss": {24: 0.32, 84: 0.28},
    # S2 缺席学时四档：20 / 30 / 40 / 60
    "absence": {29: 22.0, 59: 34.0, 89: 46.0, 119: 62.0},
    # S3 未经批准离校四档：4 / 6 / 8 / 10 天
    "leave": {33: 5, 63: 7, 93: 9, 123: 11},
    # S3 学籍情形：连续 2 周不参加活动 / 累计 4 周不参加活动 / 逾期 2 周未注册
    "inactive_consecutive": {17: 2},
    "inactive_total": {27: 4},
    "unregistered": {20: 2},
}

FIELDS = [
    "student_id", "name", "gender", "grade", "class_name", "major", "password_hash",
    "gpa", "fail_count", "library_visits", "study_hours",
    "attendance_rate", "homework_timely_rate", "credit_progress", "competition_awards",
    "psy_score", "psy_counsel_times", "mood_check", "stress_index",
    "physical_test_score", "sleep_regularity",
    "second_class_count", "volunteer_hours", "club_role", "social_connection",
    "canteen_regularity", "dorm_regularity", "consumption_daily",
    "breakfast_rate", "leave_abnormal",
    "political_status", "discipline_violation", "integrity_record",
    "collective_participation",
    "grit", "self_control", "engagement", "metacognition",
    "self_perception", "learning_motivation",
    "continuous_absence", "night_not_home", "consumption_drop", "grade_drop",
    # 手册硬规则原始字段（对应 config_setting.HANDBOOK_RULES）
    "fitness_test_status", "fitness_history", "thesis_gpa",
    "fail_grade_consecutive", "fail_grade_total",
    "campus_run_records", "campus_run_valid_count", "campus_run_year_km",
    "pe_course_score", "run_violation_count",
    "course_absence_ratio", "homework_miss_ratio",
    "absence_hours", "leave_days",
    "inactive_weeks_consecutive", "inactive_weeks_total", "unregistered_weeks",
    "labor_hours", "social_practice_done", "aesthetic_detail",
    "academic_tech_score", "dorm_hygiene_score",
    "handbook_hits", "handbook_reminders",
]


def generate_student(seq: int, grade: int, class_name: str) -> dict:
    """生成单个学生的模拟记录。

    通过基础"综合潜力"随机变量驱动各维度，使数据具备自然相关性，
    少量学生被注入风险信号以形成高危 / 关注样本。
    """
    rng = random.Random(seq * 7919 + grade)
    potential = rng.gauss(0.6, 0.18)  # 综合潜力 0~1
    potential = min(1.0, max(0.05, potential))

    gpa = round(min(4.0, max(1.0, 2.0 + potential * 2.0 + rng.gauss(0, 0.25))), 2)
    fail_count = int(max(0, rng.gauss(1.5 - potential * 1.4, 1.0)))
    library_visits = int(max(0, rng.gauss(60 + potential * 80, 25)))
    study_hours = int(max(0, rng.gauss(120 + potential * 160, 40)))
    attendance_rate = round(min(1.0, max(0.4, potential + rng.gauss(0, 0.08))), 3)
    homework_timely_rate = round(min(1.0, max(0.3, potential + rng.gauss(0, 0.10))), 3)
    credit_progress = round(min(1.0, max(0.5, 0.7 + potential * 0.3 + rng.gauss(0, 0.05))), 3)
    canteen_regularity = round(min(1.0, max(0.2, potential + rng.gauss(0, 0.12))), 3)
    dorm_regularity = round(min(1.0, max(0.2, potential + rng.gauss(0, 0.12))), 3)
    second_class_count = int(max(0, rng.gauss(8 + potential * 14, 5)))
    volunteer_hours = int(max(0, rng.gauss(20 + potential * 40, 12)))
    club_role = 1 if rng.random() < (0.25 + potential * 0.35) else 0
    social_connection = round(min(100, max(0, 40 + potential * 50 + rng.gauss(0, 12))), 1)
    competition_awards = int(max(0, rng.gauss(1 + potential * 3, 1.2)))
    consumption_daily = round(min(60.0, max(5.0, 20 + potential * 20 + rng.gauss(0, 6))), 1)
    breakfast_rate = round(min(1.0, max(0.1, potential + rng.gauss(0, 0.15))), 3)
    physical_test_score = round(min(100, max(0, 55 + potential * 35 + rng.gauss(0, 10))), 1)
    sleep_regularity = round(min(1.0, max(0.2, potential + rng.gauss(0, 0.12))), 3)
    political_status = round(min(100, max(0, 70 + potential * 25 + rng.gauss(0, 8))), 1)
    discipline_violation = 0
    integrity_record = 1
    collective_participation = round(min(1.0, max(0, 0.4 + potential * 0.5 + rng.gauss(0, 0.12))), 3)

    # 非认知能力六维（行为代理 + 模拟，0-100 越高越好）
    base_nc = 45 + potential * 45
    grit = round(min(100, max(0, base_nc + rng.gauss(0, 10))), 1)
    self_control = round(min(100, max(0, base_nc + rng.gauss(0, 10))), 1)
    engagement = round(min(100, max(0, base_nc + rng.gauss(0, 10))), 1)
    metacognition = round(min(100, max(0, base_nc + rng.gauss(0, 10))), 1)
    self_perception = round(min(100, max(0, base_nc + rng.gauss(0, 10))), 1)
    learning_motivation = round(min(100, max(0, base_nc + rng.gauss(0, 10))), 1)

    # 心理健康维度（0-100，越高越好；压力指数越高越差）
    psy_score = round(min(100, max(0, 62 + potential * 28 + rng.gauss(0, 9))), 1)
    mood_check = round(min(100, max(0, 60 + potential * 30 + rng.gauss(0, 10))), 1)
    stress_index = round(min(100, max(0, 45 - potential * 28 + rng.gauss(0, 14))), 1)
    psy_counsel_times = int(max(0, rng.gauss(1 + (1 - potential) * 1.5, 1.2)))

    # 风险指标：多数正常，少数注入异常
    continuous_absence = 0
    night_not_home = 0
    leave_abnormal = 0
    consumption_drop = 0.0
    grade_drop = 0.0
    is_high = False
    roll = rng.random()
    if roll < 0.08:  # 高危样本
        is_high = True
        continuous_absence = rng.randint(5, 12)
        night_not_home = rng.randint(3, 9)
        leave_abnormal = rng.randint(2, 6)
        consumption_drop = round(rng.uniform(0.4, 0.8), 3)
        grade_drop = round(rng.uniform(0.25, 0.6), 3)
        fail_count += rng.randint(1, 3)
        discipline_violation = rng.randint(1, 2)
        integrity_record = 0 if rng.random() < 0.4 else 1
    elif roll < 0.22:  # 关注样本
        continuous_absence = rng.randint(2, 5)
        night_not_home = rng.randint(0, 3)
        leave_abnormal = rng.randint(0, 2)
        consumption_drop = round(rng.uniform(0.0, 0.4), 3)
        grade_drop = round(rng.uniform(0.0, 0.25), 3)

    # 风险样本心理健康与品德同步走弱
    if is_high:
        psy_score = round(max(0, psy_score - rng.uniform(15, 30)), 1)
        mood_check = round(max(0, mood_check - rng.uniform(15, 25)), 1)
        stress_index = round(min(100, stress_index + rng.uniform(15, 30)), 1)
        psy_counsel_times += rng.randint(1, 3)
        grit = round(max(0, grit - rng.uniform(10, 25)), 1)
        self_control = round(max(0, self_control - rng.uniform(10, 25)), 1)
        engagement = round(max(0, engagement - rng.uniform(10, 25)), 1)
    elif roll < 0.22:
        psy_score = round(max(0, psy_score - rng.uniform(5, 12)), 1)
        stress_index = round(min(100, stress_index + rng.uniform(5, 12)), 1)

    # ==================== 手册硬规则原始字段 ====================
    # 独立随机流（hrng）：保证上面既有字段的分布与数值完全不受影响。
    hrng = random.Random(seq * 104729 + grade)
    level = cohort_to_level(grade)          # 入学年份 -> 年级序号（1~4）

    # ---- P1 体测：状态 + 历年总分（当年总分与 physical_test_score 保持一致）----
    fitness_status = "正常"
    if seq % 37 == 0:
        fitness_status = "免测"                                   # 手册：免测按 50 分计
        physical_test_score = float(HANDBOOK_RULES["P1"]["exempt_score"])
    elif seq % 41 == 0:
        fitness_status = "无故缺考"                               # 手册：无敌未测各项按 0 分计
        physical_test_score = 0.0

    fitness_history = {}
    for y in range(1, level + 1):
        if y == level:
            fitness_history[str(y)] = round(float(physical_test_score), 1)
        else:
            fitness_history[str(y)] = round(
                min(100.0, max(0.0, physical_test_score + hrng.gauss(2.0, 3.0))), 1
            )
    if seq in HANDBOOK_CASES["fitness_low"]:                      # A4 反例
        fitness_history.update(HANDBOOK_CASES["fitness_low"][seq])
        if level >= 4:
            fitness_history[str(level)] = round(float(physical_test_score), 1)

    # ---- P2 校园跑：学年逐次记录（有效次数 / 学年里程由判定引擎核算，不预先写死）----
    run_quality = potential + hrng.gauss(0, 0.10)
    if is_high:
        run_quality -= 0.35
    run_quality = min(1.0, max(0.02, run_quality))
    campus_run_records = []
    for week in range(1, RUN_YEAR_WEEKS + 1):
        lam = 1.6 + run_quality * 3.2                             # 每周期望打卡次数
        times = min(6, max(0, int(round(hrng.gauss(lam, 0.8)))))
        for _ in range(times):
            campus_run_records.append({
                "week": week,
                "day": hrng.randint(1, 7),
                "km": round(max(0.6, hrng.gauss(2.1, 0.35)), 2),
                "pace": round(min(11.5, max(3.2, hrng.gauss(6.1, 0.9))), 2),
            })
    pe_course_score = round(min(100.0, max(0.0, 60 + run_quality * 38 + hrng.gauss(0, 5))), 1)
    run_violation_count = 1 if seq % 53 == 0 else 0               # P3 代跑 / 骑车打卡

    # ---- S1 课程考核资格（单门课口径，独立于整体出勤率）----
    course_absence_ratio = round(min(1.0, max(0.0, hrng.gauss(0.06, 0.05))), 3)
    homework_miss_ratio = round(min(1.0, max(0.0, hrng.gauss(0.05, 0.05))), 3)
    if is_high:
        course_absence_ratio = round(min(1.0, course_absence_ratio + 0.15), 3)
        homework_miss_ratio = round(min(1.0, homework_miss_ratio + 0.12), 3)

    # ---- S2 学期累计缺席学时 ----
    absence_hours = round(min(120.0, max(0.0, hrng.gauss(4, 4) + (10 if is_high else 0))), 1)

    # ---- S3 未经批准离校天数 / 学籍情形 ----
    leave_days = int(max(0, round(hrng.gauss(0, 1) + leave_abnormal * 0.8)))
    inactive_weeks_consecutive = 0
    inactive_weeks_total = 0
    unregistered_weeks = 0

    # ---- A3 未晋级次数（大一不参与学年末晋级评估，故默认 0）----
    fail_grade_consecutive = 0
    fail_grade_total = 0

    # ---- A5 毕设 GPA（仅毕业年级）----
    thesis_gpa = None
    if level >= 4:
        thesis_gpa = round(min(4.0, max(0.0, gpa + hrng.gauss(0.05, 0.3))), 2)

    # ---- EC 第二课堂：劳动 / 社会实践 / 美育 / 学术科技 / 宿舍卫生 ----
    # 参数按「未达标比例约 5%~13%」标定，避免提醒列表被稀释（详见项目自检记录）
    labor_hours = round(min(80.0, max(0.0, hrng.gauss(11 + potential * 14, 6))), 1)
    social_practice_done = 1 if hrng.random() < (0.72 + potential * 0.25) else 0
    academic_tech_score = round(min(120.0, max(0.0, hrng.gauss(18 + potential * 55, 18))), 1)
    dorm_hygiene_score = round(min(100.0, max(40.0, hrng.gauss(78 + potential * 14, 7))), 1)
    aesthetic_detail = {
        "参赛次数": int(max(0, round(hrng.gauss(0.4 + potential * 2.2, 1.0)))),
        "观众次数": int(max(0, round(hrng.gauss(1.6 + potential * 2.0, 1.2)))),
        "讲座次数": int(max(0, round(hrng.gauss(0.5 + potential * 3.0, 1.3)))),
        "网课学时": int(max(0, round(hrng.gauss(3.5 + potential * 6, 2.4)))),
        "比赛拓展分": int(max(0, round(hrng.gauss(potential * 7, 4.5)))),
        "展览参观次数": int(max(0, round(hrng.gauss(1.2 + potential * 2.4, 1.3)))),
    }

    # ---- 确定性注入演示样本（保证 12 条手册规则均有触发案例）----
    if seq in HANDBOOK_CASES["gpa_warn"]:
        gpa = HANDBOOK_CASES["gpa_warn"][seq]
    if seq in HANDBOOK_CASES["thesis_gpa"]:
        thesis_gpa = HANDBOOK_CASES["thesis_gpa"][seq]
    if seq in HANDBOOK_CASES["credit_low"]:
        credit_progress = HANDBOOK_CASES["credit_low"][seq]
    if seq in HANDBOOK_CASES["fail_grade"]:
        fail_grade_consecutive, fail_grade_total = HANDBOOK_CASES["fail_grade"][seq]
    if seq in HANDBOOK_CASES["course_absence"]:
        course_absence_ratio = HANDBOOK_CASES["course_absence"][seq]
    if seq in HANDBOOK_CASES["homework_miss"]:
        homework_miss_ratio = HANDBOOK_CASES["homework_miss"][seq]
    if seq in HANDBOOK_CASES["absence"]:
        absence_hours = float(HANDBOOK_CASES["absence"][seq])
    if seq in HANDBOOK_CASES["leave"]:
        leave_days = int(HANDBOOK_CASES["leave"][seq])
    if seq in HANDBOOK_CASES["inactive_consecutive"]:
        inactive_weeks_consecutive = HANDBOOK_CASES["inactive_consecutive"][seq]
        inactive_weeks_total = max(inactive_weeks_total, inactive_weeks_consecutive)
    if seq in HANDBOOK_CASES["inactive_total"]:
        inactive_weeks_total = HANDBOOK_CASES["inactive_total"][seq]
    if seq in HANDBOOK_CASES["unregistered"]:
        unregistered_weeks = HANDBOOK_CASES["unregistered"][seq]

    # 解析专业（班级名形如 "2023级-计算机1班" -> "计算机"）
    import re
    m = re.search(r"级-([^\d]+)", class_name)
    major = m.group(1) if m else class_name

    class_code = CLASS_CODES[grade][class_name]
    sid = f"{grade}{class_code}{seq:05d}"
    name = rng.choice(SURNAMES) + rng.choice(GIVEN)
    return {
        "student_id": sid,
        "name": name,
        "gender": rng.choice(["男", "女"]),
        "grade": grade,
        "class_name": class_name,
        "major": major,
        "password_hash": hash_password(DEFAULT_STUDENT_PASSWORD),
        "gpa": gpa,
        "fail_count": fail_count,
        "library_visits": library_visits,
        "study_hours": study_hours,
        "attendance_rate": attendance_rate,
        "homework_timely_rate": homework_timely_rate,
        "credit_progress": credit_progress,
        "competition_awards": competition_awards,
        "psy_score": psy_score,
        "psy_counsel_times": psy_counsel_times,
        "mood_check": mood_check,
        "stress_index": stress_index,
        "physical_test_score": physical_test_score,
        "sleep_regularity": sleep_regularity,
        "second_class_count": second_class_count,
        "volunteer_hours": volunteer_hours,
        "club_role": club_role,
        "social_connection": social_connection,
        "canteen_regularity": canteen_regularity,
        "dorm_regularity": dorm_regularity,
        "consumption_daily": consumption_daily,
        "breakfast_rate": breakfast_rate,
        "leave_abnormal": leave_abnormal,
        "political_status": political_status,
        "discipline_violation": discipline_violation,
        "integrity_record": integrity_record,
        "collective_participation": collective_participation,
        "grit": grit,
        "self_control": self_control,
        "engagement": engagement,
        "metacognition": metacognition,
        "self_perception": self_perception,
        "learning_motivation": learning_motivation,
        "continuous_absence": continuous_absence,
        "night_not_home": night_not_home,
        "consumption_drop": consumption_drop,
        "grade_drop": grade_drop,
        # ---- 手册硬规则原始字段 ----
        "fitness_test_status": fitness_status,
        "fitness_history": json.dumps(fitness_history, ensure_ascii=False),
        "thesis_gpa": thesis_gpa,
        "fail_grade_consecutive": fail_grade_consecutive,
        "fail_grade_total": fail_grade_total,
        "campus_run_records": json.dumps(campus_run_records, ensure_ascii=False),
        "campus_run_valid_count": None,     # 由判定引擎从逐次记录核算后回填
        "campus_run_year_km": None,         # 同上
        "pe_course_score": pe_course_score,
        "run_violation_count": run_violation_count,
        "course_absence_ratio": course_absence_ratio,
        "homework_miss_ratio": homework_miss_ratio,
        "absence_hours": absence_hours,
        "leave_days": leave_days,
        "inactive_weeks_consecutive": inactive_weeks_consecutive,
        "inactive_weeks_total": inactive_weeks_total,
        "unregistered_weeks": unregistered_weeks,
        "labor_hours": labor_hours,
        "social_practice_done": social_practice_done,
        "aesthetic_detail": json.dumps(aesthetic_detail, ensure_ascii=False),
        "academic_tech_score": academic_tech_score,
        "dorm_hygiene_score": dorm_hygiene_score,
        # ---- 手册判定结果（由 analysis_core.handbook_rules 回填）----
        "handbook_hits": None,
        "handbook_reminders": None,
    }


def generate_dataset(per_class: int = 60) -> list[dict]:
    """生成完整模拟数据集。"""
    records = []
    seq = 1
    for grade, classes in GRADE_CLASSES.items():
        for cls in classes:
            for _ in range(per_class):
                records.append(generate_student(seq, grade, cls))
                seq += 1
    return records


def write_csv(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def main(per_class: int = 60, to_db: bool = True) -> list[dict]:
    """生成数据：写入 CSV 并初始化数据库。"""
    random.seed(20260811)
    np.random.seed(20260811)
    db_init.init_database(force=True)
    records = generate_dataset(per_class=per_class)
    csv_path = os.path.join(os.path.dirname(__file__), "raw_student_data.csv")
    write_csv(records, csv_path)
    if to_db:
        orm_ops.bulk_upsert_students(records)
    print(f"已生成 {len(records)} 条模拟数据 -> {csv_path}")
    if to_db:
        print(f"已写入数据库：{db_init.DB_PATH}")
        # 自动生成分析结果（聚类、综合评分、风险等级等）
        print("正在执行分析流水线...")
        analysis_pipeline.run_pipeline(save=True)
        print("分析完成。")
    return records


if __name__ == "__main__":
    main()
