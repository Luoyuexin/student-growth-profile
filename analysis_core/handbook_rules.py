"""analysis_core 本科手册硬性规则判定模块。

依据《本科生手册》整理出的 12 条可量化规则（见 config_setting.HANDBOOK_RULES），
对单个学生输出「是否需要被提醒 / 督促」，以及对应的客观提醒话术。

设计原则：
1. 本模块只产出「提醒」，不产出「正常 / 关注 / 高危」状态标签——那仍由
   analysis_core/risk_level.py 负责，两者口径互不干扰；
2. 只输出「未达标 / 需跟进」的条目，已达标项不逐条罗列（避免页面变成手册条文清单）；
3. 校园跑有效次数不信任外部字段，一律由逐次原始记录按手册口径核算
   （配速 4~10、单次男 1.5km / 女 1.2km、每日 ≤2 次、每周 ≤8 次）；
4. 全部阈值取自 config_setting，便于按当学期通知调整；
5. 另提供 enrich_record()，把手册派生指标注入记录，供六维评分使用。
"""

from __future__ import annotations

import json
from math import ceil
from typing import Dict, List, Optional, Tuple

from config_setting import (
    HANDBOOK_REMINDERS,
    HANDBOOK_RULES,
    cohort_to_level,
)

RULES = HANDBOOK_RULES
TPL = HANDBOOK_REMINDERS

# 处分档位严重度（用于在 S2 / S3 中取最高档）
SEVERITY = {"警告": 1, "严重警告": 2, "记过": 3, "留校察看": 4}

# 处分档位 -> 品德素养「纪律处分」二级指标折算分
PENALTY_SCORE = {"警告": 70.0, "严重警告": 55.0, "记过": 40.0, "留校察看": 20.0}
NO_PENALTY_SCORE = 100.0

# 体测等级 -> 折算分（供「体测等级」二级指标使用；免测按手册 50 分、缺考按 0 分）
FITNESS_LEVEL_SCORE = {
    "优秀": 100.0,
    "良好": 85.0,
    "及格": 70.0,
    "不及格": 40.0,
    "免测": 50.0,
    "缺考": 0.0,
}

# 是否输出「鼓励」类话术。按业务方要求：只提醒未达标项，不罗列已达标项，
# 故默认关闭；如需在页面上展示正向反馈，把此开关置 True 即可。
EMIT_PRAISE = False


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _num(value, default: float = 0.0) -> float:
    """安全转 float（None / 异常值 -> 默认值）。"""
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int(value, default: int = 0) -> int:
    return int(round(_num(value, default)))


def _json_load(value, default):
    """读取 JSON 文本字段（已是 list/dict 则直接返回）。"""
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# P2 校园跑：有效次数核算
# ---------------------------------------------------------------------------
def compute_valid_runs(records: list, gender: str, week_range: Optional[Tuple[int, int]] = None) -> Dict:
    """按手册口径核算校园跑有效次数。

    有效条件（需同时满足）：
      - 配速落在 pace_range（4~10 分钟/公里）内；
      - 单次里程 ≥ 性别下限（男 1.5km / 女 1.2km）；
      - 同一自然日最多计 daily_max 次、同一周最多计 weekly_max 次。

    week_range=(lo, hi) 时只核算该区间的记录（用于区分「本学期」与「整个学年」）。
    返回 {有效次数, 无效次数, 总次数, 有效公里, 无效原因}。
    """
    p2 = RULES["P2"]
    lo_km = p2["single_km"].get(gender, p2["single_km"]["男"])
    pace_lo, pace_hi = p2["pace_range"]
    daily_max, weekly_max = p2["daily_max"], p2["weekly_max"]

    recs = list(records or [])
    if week_range:
        w_lo, w_hi = week_range
        recs = [r for r in recs if w_lo <= _int(r.get("week"), -1) <= w_hi]

    day_cnt: Dict[Tuple[int, int], int] = {}
    week_cnt: Dict[int, int] = {}
    reasons: Dict[str, int] = {}
    valid, total_km = 0, 0.0

    def _bad(key: str) -> None:
        reasons[key] = reasons.get(key, 0) + 1

    for r in recs:
        week = _int(r.get("week"), -1)
        day = _int(r.get("day"), -1)
        km = _num(r.get("km"), 0.0)
        pace = _num(r.get("pace"), 0.0)

        if pace < pace_lo or pace > pace_hi:
            _bad("配速不在有效区间")
            continue
        if km < lo_km:
            _bad("单次里程不足")
            continue
        if day_cnt.get((week, day), 0) >= daily_max:
            _bad("当日次数超上限")
            continue
        if week_cnt.get(week, 0) >= weekly_max:
            _bad("当周次数超上限")
            continue

        day_cnt[(week, day)] = day_cnt.get((week, day), 0) + 1
        week_cnt[week] = week_cnt.get(week, 0) + 1
        valid += 1
        total_km += km

    return {
        "有效次数": valid,
        "无效次数": len(recs) - valid,
        "总次数": len(recs),
        "有效公里": round(total_km, 2),
        "无效原因": reasons,
    }


# ---------------------------------------------------------------------------
# P1 体测：等级与折算
# ---------------------------------------------------------------------------
def fitness_grade(score: float, status: str = "正常") -> str:
    """按手册四级给出体测等级：优秀 / 良好 / 及格 / 不及格（免测、缺考单列）。"""
    if status == "免测":
        return "免测"
    if status == "无故缺考":
        return "缺考"
    for line, name in RULES["P1"]["grade_lines"]:
        if score >= line:
            return name
    return "不及格"


def fitness_grade_score(score: float, status: str = "正常") -> float:
    """把体测等级折算为 0~100 分，供六维「体测等级」二级指标使用。"""
    return FITNESS_LEVEL_SCORE.get(fitness_grade(score, status), 40.0)


def fitness_graduation_score(history: Dict, level: int) -> float:
    """手册口径：体测毕业评定 = 毕业当年成绩 × 50% + 其他学年平均 × 50%。"""
    if not history:
        return 0.0
    cur = _num(history.get(str(level)), 0.0)
    others = [_num(v) for k, v in history.items() if k != str(level)]
    if not others:
        return round(cur, 1)
    return round(cur * 0.5 + (sum(others) / len(others)) * 0.5, 1)


# ---------------------------------------------------------------------------
# 纪律处分：S2 缺席学时 / S3 离校天数 取最高档
# ---------------------------------------------------------------------------
def discipline_penalty(record: dict, include_legacy: bool = True) -> Tuple[Optional[str], float]:
    """返回 (最高处分档位, 品德素养折算分)。无处分返回 (None, 100.0)。

    include_legacy=True 时，把既有「违纪处分次数」字段一并计入（用于品德素养折算）；
    判定「处分尚未解除」这类毕业条件时建议传 False，只认手册 S2 / S3 的实际处分。
    """
    best: Optional[str] = None

    hours = _num(record.get("absence_hours"), 0.0)
    for th, name in RULES["S2"]["thresholds"]:
        if hours >= th:
            best = name          # thresholds 升序，循环结束时即为最高档

    days = _num(record.get("leave_days"), 0.0)
    for th, name in RULES["S3"]["leave_thresholds"]:
        if days >= th:
            if best is None or SEVERITY.get(name, 0) > SEVERITY.get(best, 0):
                best = name

    # 兼容既有「违纪处分次数」字段（旧口径），无更重处分时按警告计
    if best is None and include_legacy and _num(record.get("discipline_violation"), 0.0) > 0:
        best = "警告"

    return best, (PENALTY_SCORE.get(best, NO_PENALTY_SCORE) if best else NO_PENALTY_SCORE)


# ---------------------------------------------------------------------------
# EC 第二课堂：五部分计分（手册 4.2）
# ---------------------------------------------------------------------------
def score_second_classroom(record: dict) -> Dict:
    """按手册第二部分「第二课堂成绩单」五部分计分，总分 600。

    各部分均设上限（config_setting.HANDBOOK_RULES.EC.parts_max）：
      道德与思想素质 180 / 学术与科技创新 120 / 体育与健康 100 /
      美学与人文素养 100 / 劳动与社会实践 100
    """
    ec = RULES["EC"]
    caps = ec["parts_max"]
    penalty_name, penalty_score = discipline_penalty(record)

    # 1) 道德与思想素质（180）：思想表现 + 集体参与 + 诚信，扣除处分影响
    moral_base = (
        _num(record.get("political_status"), 0.0) / 100.0 * 120.0
        + _num(record.get("collective_participation"), 0.0) * 30.0
        + _num(record.get("integrity_record"), 0.0) * 30.0
    )
    penalty_deduct = {"警告": 20.0, "严重警告": 40.0, "记过": 70.0, "留校察看": 110.0}
    moral = max(0.0, min(caps["道德与思想素质"], moral_base - penalty_deduct.get(penalty_name, 0.0)))

    # 2) 学术与科技创新（120）：竞赛获奖 + 学术科技活动得分
    academic = min(caps["学术与科技创新"],
                   _num(record.get("competition_awards"), 0.0) * 12.0
                   + _num(record.get("academic_tech_score"), 0.0))

    # 3) 体育与健康（100）：体测总分（60%）+ 校园跑完成率（40%）
    run = compute_valid_runs(
        _json_load(record.get("campus_run_records"), []),
        record.get("gender") or "男",
        RULES["P2"]["semester_weeks"],
    )
    target = RULES["P2"]["semester_target"] or 40
    run_ratio = min(1.0, run["有效次数"] / target)
    sports = min(caps["体育与健康"],
                 _num(record.get("physical_test_score"), 0.0) * 0.6 + run_ratio * 40.0)

    # 4) 美学与人文素养（100）：参赛/观众/讲座/网课/拓展/展览（各项设上限）
    aes = _json_load(record.get("aesthetic_detail"), {}) or {}
    b1 = min(30.0, _num(aes.get("参赛次数")) * 10.0 + _num(aes.get("观众次数")) * 5.0)
    b2 = min(30.0, _num(aes.get("讲座次数")) * 10.0)
    b3 = min(30.0, _num(aes.get("网课学时")) * 5.0)
    ext = min(20.0, _num(aes.get("比赛拓展分"))) + min(20.0, _num(aes.get("展览参观次数")) * 4.0)
    aesthetic = min(caps["美学与人文素养"], b1 + b2 + b3 + ext)

    # 5) 劳动与社会实践（100）：志愿服务 + 劳动实践 + 社会实践 + 宿舍卫生 + 第二课堂活动
    vol_hours = _num(record.get("volunteer_hours"), 0.0)
    vol = 0.0
    if vol_hours >= ec["volunteer_hours"]:
        vol = ec["volunteer_score"] + min(
            ec["volunteer_extra_cap"],
            (vol_hours - ec["volunteer_hours"]) * ec["volunteer_extra_per_hour"],
        )
    labor = ec["labor_score"] if _num(record.get("labor_hours"), 0.0) >= ec["labor_hours"] else 0.0
    social = ec["social_practice_score"] if _int(record.get("social_practice_done"), 0) else 0.0
    dorm = _dorm_score(_num(record.get("dorm_hygiene_score"), 0.0))
    activity = min(20.0, _num(record.get("second_class_count"), 0.0) * 1.0)
    labor_part = min(caps["劳动与社会实践"], vol + labor + social + dorm + activity)

    parts = {
        "道德与思想素质": round(moral, 1),
        "学术与科技创新": round(academic, 1),
        "体育与健康": round(sports, 1),
        "美学与人文素养": round(aesthetic, 1),
        "劳动与社会实践": round(labor_part, 1),
    }
    parts_max = {k: float(caps[k]) for k in parts}
    return {
        "五部分": parts,
        "上限": parts_max,
        "总分": round(sum(parts.values()), 1),
        "满分": float(ec["total_max"]),
        "美学本学期得分": round(aesthetic, 1),
        "处分档位": penalty_name,
    }


def _dorm_score(score: float) -> float:
    """宿舍卫生检查分 -> 计分档（优 20 / 良 16 / 中 12 / 合格 10 / 不合格 0）。"""
    cap = RULES["EC"]["parts_max"]["劳动与社会实践"]
    for name, lo, hi, sc in RULES["EC"]["dorm_hygiene"]:
        if lo <= score < hi or (name == "优" and score >= hi):
            return min(float(sc), cap)
    return 0.0


# ---------------------------------------------------------------------------
# 派生指标注入（供六维评分使用）
# ---------------------------------------------------------------------------
def enrich_record(record: dict) -> dict:
    """把手册派生指标注入记录副本，供六维评分使用（不修改原始字段）。

    注入内容：
      campus_run_valid_count / campus_run_year_km / campus_run_ratio /
      physical_test_grade_score / discipline_penalty_score
    """
    rec = dict(record)
    gender = rec.get("gender") or "男"
    recs = _json_load(rec.get("campus_run_records"), [])
    p2 = RULES["P2"]
    sem = compute_valid_runs(recs, gender, p2["semester_weeks"])
    year = compute_valid_runs(recs, gender, p2["year_weeks"])
    target = p2["semester_target"] or 40

    rec["campus_run_valid_count"] = sem["有效次数"]
    rec["campus_run_year_km"] = year["有效公里"]
    rec["campus_run_ratio"] = round(min(1.0, sem["有效次数"] / target), 3)
    rec["physical_test_grade_score"] = fitness_grade_score(
        _num(rec.get("physical_test_score"), 0.0),
        rec.get("fitness_test_status") or "正常",
    )
    rec["discipline_penalty_score"] = discipline_penalty(rec)[1]
    return rec


# ---------------------------------------------------------------------------
# 12 条规则判定：只输出需要提醒的项
# ---------------------------------------------------------------------------
def judge(record: dict) -> Dict:
    """对一个学生执行手册规则判定。

    返回 {"hits": [规则编号...], "reminders": [{"type","rule","text"}...]}。
    只输出「未达标 / 需跟进」的条目；全部达标时 reminders 为空列表。
    """
    hits: List[str] = []
    reminders: List[Dict] = []

    def add(rule: str, key: str, rtype: str = "提醒", **kw) -> None:
        hits.append(rule)
        reminders.append({"type": rtype, "rule": rule, "text": TPL[key].format(**kw)})

    grade = _int(record.get("grade"), 0)
    level = cohort_to_level(grade) if grade else 1
    gpa = _num(record.get("gpa"), 0.0)
    credit = _num(record.get("credit_progress"), 1.0)
    run_records = _json_load(record.get("campus_run_records"), [])
    run_history = _json_load(record.get("fitness_history"), {}) or {}
    penalty_name, _penalty_score = discipline_penalty(record)
    # 仅手册 S2 / S3 的实际处分（不含旧口径违纪次数），用于「处分是否已解除」判定
    active_penalty, _ = discipline_penalty(record, include_legacy=False)

    # ================= 学业类 A =================
    # A1 学业警示：课程最高平均 GPA < 2.00
    if gpa < RULES["A1"]["gpa_line"]:
        add("A1", "gpa_low", gpa=f"{gpa:.2f}")

    # A2 年级晋级（大一~大三；大四转为毕业 / 学位条件核查）
    a2 = RULES["A2"]
    if level in a2["credit_ratio"]:
        th = a2["credit_ratio"][level]
        if credit < th:
            add("A2", "credit_warn", ratio=f"{credit:.0%}", grade=level, threshold=f"{th:.0%}")

    # A3 退学红线：大一学分 < 1/3、连续 2 次未晋级、累计 3 次未晋级
    a3 = RULES["A3"]
    if level == 1 and credit < a3["freshman_credit_ratio"]:
        add("A3", "credit_freshman", ratio=f"{credit:.0%}")
    cons = _int(record.get("fail_grade_consecutive"), 0)
    total = _int(record.get("fail_grade_total"), 0)
    if cons >= a3["consecutive_fail"]:
        add("A3", "unpass_consecutive", n=cons)
    if total >= a3["total_fail"]:
        add("A3", "unpass_total", n=total)

    # A4 毕业条件（毕业年级）：体测大一大二平均、处分是否解除
    if level >= 4:
        a4 = RULES["A4"]
        vals = [
            _num(run_history.get(str(y)))
            for y in a4["fitness_avg_years"]
            if run_history.get(str(y)) is not None
        ]
        if vals:
            avg = sum(vals) / len(vals)
            if avg < a4["fitness_avg_line"]:
                add("A4", "graduation_fitness", avg=f"{avg:.1f}")
        if active_penalty is not None:
            add("A4", "graduation_discipline")

    # A5 学士学位（毕业年级）：课程最高平均 GPA 与毕设 GPA 均需 ≥ 2.00
    if level >= 4:
        thesis = record.get("thesis_gpa")
        if thesis is not None and _num(thesis) < RULES["A5"]["thesis_gpa_line"]:
            add("A5", "thesis_gpa_low", gpa=f"{_num(thesis):.2f}")

    # ================= 体质健康类 P =================
    # P1 体测：四级判定 / 免测按 50 / 无故缺考按 0 / 毕业评定 < 50
    p1 = RULES["P1"]
    fscore = _num(record.get("physical_test_score"), 0.0)
    fstatus = record.get("fitness_test_status") or "正常"
    flv = fitness_grade(fscore, fstatus)
    if fstatus == "免测":
        add("P1", "fitness_exempt", score=f"{p1['exempt_score']:.0f}")
    elif fstatus == "无故缺考":
        add("P1", "fitness_absent")
    elif flv == "不及格":
        add("P1", "fitness_fail", score=f"{fscore:.1f}")
    elif EMIT_PRAISE:
        add("P1", "fitness_level", rtype="鼓励", score=f"{fscore:.1f}", level=flv)
    if level >= 4 and fstatus == "正常" and run_history:
        g = fitness_graduation_score(run_history, level)
        if g < p1["graduation_line"]:
            add("P1", "fitness_graduation", score=f"{g:.1f}")

    # P2 校园跑：有效次数（逐次记录核算）+ 学年累计里程 + 体育课一票否决
    p2 = RULES["P2"]
    gender = record.get("gender") or "男"
    sem = compute_valid_runs(run_records, gender, p2["semester_weeks"])
    year = compute_valid_runs(run_records, gender, p2["year_weeks"])
    target = p2["semester_target"] or 40
    if sem["有效次数"] < target:
        need = target - sem["有效次数"]
        weekly = max(p2["weekly_min"], ceil(need / 4))
        add("P2", "run_progress", done=sem["有效次数"], target=target, weekly=weekly)
        if _num(record.get("pe_course_score"), 100.0) < 60.0:
            add("P2", "run_pe_fail", done=sem["有效次数"], target=target)
    elif EMIT_PRAISE:
        add("P2", "run_done", rtype="鼓励", done=sem["有效次数"], target=target)

    km_target = p2["year_km"].get(gender, 130.0)
    if year["有效公里"] < km_target:
        add("P2", "run_year_km",
            km=f"{year['有效公里']:.1f}", target=f"{km_target:.0f}",
            gap=f"{km_target - year['有效公里']:.1f}")

    # P3 跑步诚信：代跑 / 骑车打卡
    viol = _int(record.get("run_violation_count"), 0)
    if viol >= RULES["P3"]["violation_threshold"]:
        add("P3", "run_violation", n=viol)

    # ================= 学籍纪律类 S =================
    # S1 课程考核资格：缺课 > 1/3 或欠交作业 > 1/4
    s1 = RULES["S1"]
    ar = _num(record.get("course_absence_ratio"), 0.0)
    if ar > s1["absence_ratio"]:
        add("S1", "absence_s1", ratio=f"{ar:.0%}")
    hm = _num(record.get("homework_miss_ratio"), 0.0)
    if hm > s1["homework_miss_ratio"]:
        add("S1", "homework_s1", ratio=f"{hm:.0%}")

    # S2 缺席（旷课）处分：20 / 30 / 40 / 60 学时（取最高档）
    hours = _num(record.get("absence_hours"), 0.0)
    s2_level = None
    for th, name in RULES["S2"]["thresholds"]:
        if hours >= th:
            s2_level = name
    if s2_level:
        add("S2", "absence_s2", hours=f"{hours:g}", level=s2_level)

    # S3 未经批准离校 / 学籍情形
    s3 = RULES["S3"]
    days = _num(record.get("leave_days"), 0.0)
    s3_level = None
    for th, name in s3["leave_thresholds"]:
        if days >= th:
            s3_level = name
    if s3_level:
        add("S3", "leave_s3", days=_int(days), level=s3_level)
    inc = _int(record.get("inactive_weeks_consecutive"), 0)
    if inc >= s3["inactive_consecutive_weeks"]:
        add("S3", "inactive_s3", kind="连续", weeks=inc)
    acc = _int(record.get("inactive_weeks_total"), 0)
    if acc >= s3["inactive_total_weeks"]:
        add("S3", "inactive_s3", kind="累计", weeks=acc)
    unreg = _int(record.get("unregistered_weeks"), 0)
    if unreg >= s3["unregistered_weeks"]:
        add("S3", "unregistered_s3", weeks=unreg)

    # ================= 第二课堂 EC =================
    ec = RULES["EC"]
    vol_hours = _num(record.get("volunteer_hours"), 0.0)
    if vol_hours < ec["volunteer_hours"]:
        add("EC", "ec_volunteer", hours=f"{vol_hours:g}")
    labor_hours = _num(record.get("labor_hours"), 0.0)
    if labor_hours < ec["labor_hours"]:
        add("EC", "ec_labor", hours=f"{labor_hours:g}")
    if not _int(record.get("social_practice_done"), 0):
        add("EC", "ec_social")
    aes_score = score_second_classroom(record)["美学本学期得分"]
    if aes_score < ec["aesthetic_per_semester"]:
        add("EC", "ec_aesthetic", score=f"{aes_score:.0f}")

    return {"hits": sorted(set(hits)), "reminders": reminders}
