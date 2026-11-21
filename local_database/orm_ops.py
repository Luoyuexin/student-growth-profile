"""local_database ORM 操作模块

以学生 / 教师 / 备注三张表为核心的轻量 ORM 封装。
所有对外读取接口默认经过脱敏与权限过滤，保证隐私安全要求。
"""

from __future__ import annotations

import datetime
import json
from typing import Dict, List, Optional

import streamlit as st

from local_database import db_init
from config_setting import (
    ROLE_STUDENT,
    ROLE_TEACHER,
    mask_name,
    mask_id,
    TEACHER_CLASS_SCOPE,
)

STUDENT_COLUMNS = [
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
    # 手册硬规则原始字段
    "fitness_test_status", "fitness_history", "thesis_gpa",
    "fail_grade_consecutive", "fail_grade_total",
    "campus_run_records", "campus_run_valid_count", "campus_run_year_km",
    "pe_course_score", "run_violation_count",
    "course_absence_ratio", "homework_miss_ratio",
    "absence_hours", "leave_days",
    "inactive_weeks_consecutive", "inactive_weeks_total", "unregistered_weeks",
    "labor_hours", "social_practice_done", "aesthetic_detail",
    "academic_tech_score", "dorm_hygiene_score",
    "cluster_label", "composite_score",
    "score_study", "score_health", "score_social", "score_life",
    "score_morality", "score_non_cog",
    "risk_level", "risk_score",
    # 分维度等级与成长类型（一维 / 多维聚类产物）
    "level_overall",
    "level_study", "level_health", "level_social",
    "level_life", "level_morality", "level_non_cog",
    "cluster_type",
    # 手册判定结果
    "handbook_hits", "handbook_reminders",
    "last_updated",
]


def _hash_obj(obj) -> int:
    """把 list/dict 等结构稳定序列化后取哈希，供 @st.cache_data 使用。"""
    return hash(json.dumps(obj, sort_keys=True, default=str))


# ---------------------------------------------------------------------------
# 写入操作
# ---------------------------------------------------------------------------
def _mask_record(rec: Dict, with_raw_id: bool = False) -> Dict:
    """按隐私要求脱敏一条记录。

    展示字段（name / student_id）打码；with_raw_id=True 时额外挂一个内部主键
    `_raw_student_id`，供后续按主键查询使用（仅服务端 / 已授权会话内使用，不用于展示）。
    """
    if with_raw_id:
        rec["_raw_student_id"] = rec.get("student_id", "")
    rec["name"] = mask_name(rec.get("name", ""))
    rec["student_id"] = mask_id(rec.get("student_id", ""))
    return rec


def upsert_student(record: Dict) -> None:
    """插入或更新一条学生记录。

    若记录包含 `name` 等完整字段，则使用 INSERT ... ON CONFLICT 全量写入；
    若仅为分析结果更新（缺少 name），则使用 UPDATE 避免触发 NOT NULL 约束。
    """
    conn = db_init.get_connection()
    try:
        cols = [c for c in STUDENT_COLUMNS if c in record]
        update_cols = [c for c in cols if c != "student_id"]

        # 缺少 name 说明是部分更新，直接 UPDATE 即可
        if "name" not in record:
            set_sql = ", ".join(f"{c}=?" for c in update_cols)
            sql = f"UPDATE students SET {set_sql} WHERE student_id=?"
            values = [record.get(c) for c in update_cols] + [record.get("student_id")]
        else:
            placeholders = ", ".join("?" for _ in cols)
            update_sql = ", ".join(f"{c}=excluded.{c}" for c in update_cols)
            sql = (
                f"INSERT INTO students ({', '.join(cols)}) VALUES ({placeholders}) "
                f"ON CONFLICT(student_id) DO UPDATE SET {update_sql}"
            )
            values = [record.get(c) for c in cols]

        conn.execute(sql, values)
        conn.commit()
    finally:
        conn.close()


def bulk_upsert_students(records: List[Dict]) -> None:
    """批量写入学生记录（事务提交）"""
    conn = db_init.get_connection()
    try:
        for record in records:
            cols = [c for c in STUDENT_COLUMNS if c in record]
            placeholders = ", ".join("?" for _ in cols)
            update_cols = [c for c in cols if c != "student_id"]
            update_sql = ", ".join(f"{c}=excluded.{c}" for c in update_cols)
            sql = (
                f"INSERT INTO students ({', '.join(cols)}) VALUES ({placeholders}) "
                f"ON CONFLICT(student_id) DO UPDATE SET {update_sql}"
            )
            values = [record.get(c) for c in cols]
            conn.execute(sql, values)
        conn.commit()
    finally:
        conn.close()


def save_teacher_note(student_id: str, teacher_id: str, note: str, mark_level: str) -> None:
    """保存教师对学生的备注 / 标记"""
    conn = db_init.get_connection()
    try:
        conn.execute(
            "INSERT INTO teacher_notes (student_id, teacher_id, note, mark_level, update_time) "
            "VALUES (?, ?, ?, ?, ?)",
            (student_id, teacher_id, note, mark_level, _now()),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 聚类模型元信息（等级边界 / 成长类型名）
# ---------------------------------------------------------------------------
def upsert_cluster_meta(
    grade: int,
    scope: str,
    k: int,
    centers,
    boundaries,
    labels,
    details=None,
) -> None:
    """写入某年级某作用域的聚类模型（INSERT OR REPLACE）。

    details: 可选，定 K 过程明细（tested_k / inertias / silhouettes / scores），
    供教师端配置页绘制「定 K 过程曲线图」。
    """
    conn = db_init.get_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO cluster_meta "
            "(grade, scope, k, centers, boundaries, labels, details, update_time) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(grade), scope, int(k),
                json.dumps(centers, ensure_ascii=False),
                json.dumps(boundaries, ensure_ascii=False),
                json.dumps(labels, ensure_ascii=False),
                json.dumps(details, ensure_ascii=False) if details is not None else None,
                _now(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_cluster_meta(grade: int, scope: str) -> Optional[Dict]:
    """读取单条聚类模型元信息，不存在返回 None"""
    conn = db_init.get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM cluster_meta WHERE grade=? AND scope=?",
            (int(grade), scope),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        for key in ("centers", "boundaries", "labels", "details"):
            try:
                d[key] = json.loads(d.get(key)) if d.get(key) is not None else None
            except (TypeError, ValueError):
                d[key] = None
        return d
    finally:
        conn.close()


def list_cluster_meta(grade: Optional[int] = None) -> List[Dict]:
    """列出聚类模型元信息（可按年级过滤）"""
    conn = db_init.get_connection()
    try:
        if grade is None:
            rows = conn.execute(
                "SELECT * FROM cluster_meta ORDER BY grade, scope"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM cluster_meta WHERE grade=? ORDER BY scope",
                (int(grade),),
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            for key in ("centers", "boundaries", "labels", "details"):
                try:
                    d[key] = json.loads(d.get(key)) if d.get(key) is not None else None
                except (TypeError, ValueError):
                    d[key] = None
            result.append(d)
        return result
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 认证
# ---------------------------------------------------------------------------
def authenticate_student(student_id: str, password: str) -> Optional[Dict]:
    """学生登录认证。成功返回原始记录（含敏感字段，仅限本人会话使用），失败返回 None。"""
    conn = db_init.get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM students WHERE student_id = ?", (student_id,)
        ).fetchone()
        if not row:
            return None
        if row["password_hash"] != db_init.hash_password(password):
            return None
        return dict(row)
    finally:
        conn.close()


def authenticate_teacher(teacher_id: str, password: str) -> Optional[Dict]:
    """教师登录认证。成功返回教师记录，失败返回 None"""
    conn = db_init.get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM teachers WHERE teacher_id = ?", (teacher_id,)
        ).fetchone()
        if not row:
            return None
        if row["password_hash"] != db_init.hash_password(password):
            return None
        return dict(row)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 读取操作（权限 + 脱敏）
# ---------------------------------------------------------------------------
def get_student_self(student_id: str) -> Optional[Dict]:
    """学生查看本人完整画像（本人授权，不脱敏）"""
    conn = db_init.get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM students WHERE student_id = ?", (student_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_student_masked(student_id: str) -> Optional[Dict]:
    """教师视角查看单个学生（脱敏 + 仅显示管辖班级）返回脱敏记录"""
    conn = db_init.get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM students WHERE student_id = ?", (student_id,)
        ).fetchone()
        if not row:
            return None
        return _mask_record(dict(row), with_raw_id=True)
    finally:
        conn.close()


def list_students_by_classes(
    class_names: List[str], keyword: str = "", sort_by: str = "composite_score", desc: bool = True
) -> List[Dict]:
    """教师查看管辖班级学生列表（脱敏 + 搜索 + 排序）

    keyword 同时匹配脱敏前的姓名 / 学号（便于教师用真实信息检索），
    但返回结果仍做脱敏展示。
    """
    conn = db_init.get_connection()
    try:
        placeholders = ", ".join("?" for _ in class_names)
        sql = f"SELECT * FROM students WHERE class_name IN ({placeholders})"
        params = list(class_names)
        if keyword:
            sql += " AND (name LIKE ? OR student_id LIKE ?)"
            params += [f"%{keyword}%", f"%{keyword}%"]
        if sort_by in STUDENT_COLUMNS:
            sql += f" ORDER BY {sort_by} {'DESC' if desc else 'ASC'}"
        rows = conn.execute(sql, params).fetchall()
        return [_mask_record(dict(r), with_raw_id=True) for r in rows]
    finally:
        conn.close()


def get_class_distribution(class_names: List[str]) -> Dict[str, int]:
    """班级整体风险等级分布（用于饼图 / 柱状图）。"""
    conn = db_init.get_connection()
    try:
        placeholders = ", ".join("?" for _ in class_names)
        rows = conn.execute(
            f"SELECT risk_level, COUNT(*) AS c FROM students "
            f"WHERE class_name IN ({placeholders}) AND risk_level IS NOT NULL "
            f"GROUP BY risk_level",
            list(class_names),
        ).fetchall()
        return {row["risk_level"]: row["c"] for row in rows}
    finally:
        conn.close()


def get_risk_students(class_names: List[str], levels: List[str]) -> List[Dict]:
    """获取高危 / 关注学生列表（脱敏）。"""
    conn = db_init.get_connection()
    try:
        placeholders = ", ".join("?" for _ in class_names)
        lps = ", ".join("?" for _ in levels)
        rows = conn.execute(
            f"SELECT * FROM students WHERE class_name IN ({placeholders}) "
            f"AND risk_level IN ({lps}) ORDER BY risk_score DESC",
            list(class_names) + list(levels),
        ).fetchall()
        result = []
        for r in rows:
            result.append(_mask_record(dict(r), with_raw_id=True))
        return result
    finally:
        conn.close()


def get_reminded_students(
    class_names: List[str], rule_ids: Optional[List[str]] = None
) -> List[Dict]:
    """获取存在手册提醒的学生列表（脱敏 + 仅管辖班级）。

    rule_ids 为空时返回所有存在提醒的学生；否则只返回命中指定规则的学生
    （如 ["A1"] 只返回存在学业警示提醒的学生）。
    仅返回「需要提醒」的学生，不返回状态标签，符合"只提醒不论断"的设计。
    """
    conn = db_init.get_connection()
    try:
        placeholders = ", ".join("?" for _ in class_names)
        sql = (
            f"SELECT * FROM students WHERE class_name IN ({placeholders}) "
            f"AND handbook_reminders IS NOT NULL AND handbook_reminders NOT IN ('', '[]')"
        )
        params: List = list(class_names)
        if rule_ids:
            like = " OR ".join("handbook_hits LIKE ?" for _ in rule_ids)
            sql += f" AND ({like})"
            params += [f'%"{rid}"%' for rid in rule_ids]
        sql += " ORDER BY grade, class_name, student_id"
        rows = conn.execute(sql, params).fetchall()
        return [_mask_record(dict(r), with_raw_id=True) for r in rows]
    finally:
        conn.close()


def get_handbook_reminder_stats(class_names: List[str]) -> Dict[str, int]:
    """统计管辖班级各手册规则的提醒人数（教师端概览用）。

    返回 {规则编号: 人数}，按人数降序。
    """
    conn = db_init.get_connection()
    try:
        placeholders = ", ".join("?" for _ in class_names)
        rows = conn.execute(
            f"SELECT handbook_hits FROM students WHERE class_name IN ({placeholders}) "
            f"AND handbook_hits IS NOT NULL AND handbook_hits NOT IN ('', '[]')",
            list(class_names),
        ).fetchall()
        counter: Dict[str, int] = {}
        for r in rows:
            try:
                for rid in json.loads(r["handbook_hits"]):
                    counter[rid] = counter.get(rid, 0) + 1
            except (TypeError, ValueError):
                continue
        return dict(sorted(counter.items(), key=lambda kv: kv[1], reverse=True))
    finally:
        conn.close()


def get_teacher_notes(student_id: str) -> List[Dict]:
    """获取某学生的教师备注历史。"""
    conn = db_init.get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM teacher_notes WHERE student_id = ? ORDER BY update_time DESC",
            (student_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_all_student_ids() -> List[str]:
    """获取全部学生 ID（用于分析流水线遍历）。"""
    conn = db_init.get_connection()
    try:
        rows = conn.execute("SELECT student_id FROM students").fetchall()
        return [r["student_id"] for r in rows]
    finally:
        conn.close()


def get_raw_student(student_id: str) -> Optional[Dict]:
    """获取未脱敏的学生原始记录（仅供分析内核 / 报表内部使用）。"""
    conn = db_init.get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM students WHERE student_id = ?", (student_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


@st.cache_data(ttl=300, show_spinner=False, hash_funcs={list: _hash_obj})
def get_records_for_analysis(class_names: List[str]) -> List[Dict]:
    """获取管辖班级学生的分析用记录（含年级 / 专业 / 六维得分 / 风险等级）。

    返回结果不含姓名 / 学号等 PII，仅供后端聚合计算（学院宏观研判等）使用。
    """
    from config_setting import DIMENSION_ORDER

    score_cols = ", ".join(f"score_{d}" for d in DIMENSION_ORDER)
    conn = db_init.get_connection()
    try:
        placeholders = ", ".join("?" for _ in class_names)
        rows = conn.execute(
            f"SELECT grade, class_name, major, {score_cols}, "
            f"composite_score, risk_level "
            f"FROM students WHERE class_name IN ({placeholders})",
            list(class_names),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_raw_students_by_classes(class_names: List[str]) -> List[Dict]:
    """获取管辖班级学生的原始（含 PII，已授权）完整记录，用于本地聚合计算分析树。

    仅在教师已授权管辖范围内调用，结果不对外传输。
    """
    conn = db_init.get_connection()
    try:
        placeholders = ", ".join("?" for _ in class_names)
        sql = f"SELECT * FROM students WHERE class_name IN ({placeholders})"
        rows = conn.execute(sql, list(class_names)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_raw_students_by_class(class_name: str) -> List[Dict]:
    """获取单个班级的原始完整记录（本机聚合计算用，不对外传输）。"""
    conn = db_init.get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM students WHERE class_name = ?", (class_name,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_raw_students_by_grade(grade: int) -> List[Dict]:
    """获取整个年级的原始完整记录（本机聚合计算用，不对外传输）。"""
    conn = db_init.get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM students WHERE grade = ?", (int(grade),)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_dimension_means_by_cluster_type(cluster_type: str, grade: int) -> Dict[str, float]:
    """同一年级、同一成长类型学生的各维度得分均值（用于「同类型对比」）。

    返回六维 {维度键: 均值}；无同类样本时各维返回 0.0。
    """
    from config_setting import DIMENSION_ORDER

    score_cols = {d: f"score_{d}" for d in DIMENSION_ORDER}
    conn = db_init.get_connection()
    try:
        sql = "SELECT " + ", ".join(f"AVG({c}) AS {c}" for c in score_cols.values()) + \
              " FROM students WHERE cluster_type = ? AND grade = ?"
        row = conn.execute(sql, (cluster_type, int(grade))).fetchone()
        means = {}
        for d, col in score_cols.items():
            means[d] = float(row[col]) if row and row[col] is not None else 0.0
        return means
    finally:
        conn.close()


def get_dimension_means(class_name: str = None, grade: int = None) -> Dict[str, float]:
    """计算指定范围（班级或年级）的各维度平均分，用于雷达图参考线。

    返回六维 {study, health, social, life, morality, non_cog} 的均值（无数据返回 0.0）。
    """
    from config_setting import DIMENSION_ORDER

    score_cols = {d: f"score_{d}" for d in DIMENSION_ORDER}
    conn = db_init.get_connection()
    try:
        sql = "SELECT " + ", ".join(score_cols.values()) + " FROM students WHERE 1=1"
        params: List = []
        if class_name:
            sql += " AND class_name = ?"
            params.append(class_name)
        if grade is not None:
            sql += " AND grade = ?"
            params.append(grade)
        rows = conn.execute(sql, params).fetchall()
        means = {d: 0.0 for d in DIMENSION_ORDER}
        if not rows:
            return means
        for d, col in score_cols.items():
            vals = [r[col] for r in rows if r[col] is not None]
            means[d] = float(sum(vals) / len(vals)) if vals else 0.0
        return means
    finally:
        conn.close()


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
