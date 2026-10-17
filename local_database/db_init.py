"""local_database 数据库初始化模块。

负责 SQLite 数据库建表、初始化默认账号，并提供连接管理。
数据库文件位于本地，不上传云端，满足隐私安全要求。
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

# 数据库文件路径（本地隔离存储）
DB_DIR = Path(__file__).resolve().parent
DB_PATH = DB_DIR / "student_growth.db"

# 建表 SQL
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS students (
    student_id          TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    gender              TEXT,
    grade               INTEGER,
    class_name          TEXT,
    major               TEXT,
    password_hash       TEXT,
    -- 学业发展维度
    gpa                 REAL,
    fail_count          INTEGER,
    library_visits      INTEGER,
    study_hours         REAL,
    attendance_rate     REAL,        -- 出勤率 0~1
    homework_timely_rate REAL,       -- 作业及时率 0~1
    credit_progress     REAL,        -- 学分完成进度 0~1
    competition_awards  INTEGER,
    -- 身心健康维度
    psy_score           REAL,        -- 心理测评得分 0~100（越高越好）
    psy_counsel_times   INTEGER,     -- 心理咨询次数
    mood_check          REAL,        -- 情绪打卡分 0~100（越高越好）
    stress_index        REAL,        -- 压力指数 0~100（越高越差）
    physical_test_score REAL,        -- 体测成绩 0~100
    sleep_regularity    REAL,        -- 睡眠规律性 0~1
    -- 社会参与维度
    second_class_count  INTEGER,
    volunteer_hours     REAL,
    club_role           REAL,        -- 社团任职（0/1）
    social_connection   REAL,        -- 人际交往指数 0~100
    -- 生活适应维度
    canteen_regularity  REAL,        -- 饮食规律性 0~1
    dorm_regularity     REAL,        -- 作息规律性 0~1
    consumption_daily   REAL,        -- 日均消费额（元/天）
    breakfast_rate      REAL,        -- 早餐就餐率 0~1
    leave_abnormal      INTEGER,     -- 离校异常次数
    -- 品德素养维度
    political_status    REAL,        -- 思想政治表现 0~100
    discipline_violation REAL,       -- 违纪处分次数（越低越好）
    integrity_record    REAL,        -- 诚信记录（0/1）
    collective_participation REAL,   -- 集体参与率 0~1
    -- 非认知能力维度（行为代理 + 模拟，0~100）
    grit                REAL,        -- 恒毅力
    self_control        REAL,        -- 自制力
    engagement          REAL,        -- 投入度
    metacognition       REAL,        -- 后设认知自我调整力
    self_perception     REAL,        -- 自我感知度
    learning_motivation REAL,        -- 学习动机
    -- 风险指标
    continuous_absence  INTEGER,
    night_not_home      INTEGER,
    consumption_drop    REAL,
    grade_drop          REAL,
    -- 手册硬规则原始字段（对应 config_setting.HANDBOOK_RULES，仅供提醒判定使用）
    fitness_test_status TEXT,        -- P1 体测状态：正常 / 免测 / 无故缺考
    fitness_history     TEXT,        -- P1/A4 体测历年总分 JSON {"1": 78.5, "2": 81.0}
    thesis_gpa          REAL,        -- A5 毕业设计（论文）GPA
    fail_grade_consecutive INTEGER,  -- A3 连续未晋级次数
    fail_grade_total    INTEGER,     -- A3 累计未晋级次数
    campus_run_records  TEXT,        -- P2 校园跑逐次记录 JSON [{"week":1,"day":1,"km":1.6,"pace":6.0}]
    campus_run_valid_count INTEGER,  -- P2 本学期有效次数（由原始记录核算得到）
    campus_run_year_km  REAL,        -- P2 学年累计有效公里
    pe_course_score     REAL,        -- P2 体育课成绩（有效次数不达标按不及格处理）
    run_violation_count INTEGER,     -- P3 跑步违规（代跑 / 骑车打卡）次数
    course_absence_ratio REAL,       -- S1 单门课程最高缺课比例 0~1
    homework_miss_ratio REAL,        -- S1 单门课程最高欠交作业比例 0~1
    absence_hours       REAL,        -- S2 学期累计缺席（旷课）学时
    leave_days          INTEGER,     -- S3 未经批准连续离校天数
    inactive_weeks_consecutive INTEGER,  -- S3 连续不参加活动周数
    inactive_weeks_total INTEGER,    -- S3 累计不参加活动周数
    unregistered_weeks  INTEGER,     -- S3 逾期未注册周数
    labor_hours         REAL,        -- EC 劳动实践时长
    social_practice_done INTEGER,    -- EC 寒暑假社会实践完成（0/1）
    aesthetic_detail    TEXT,        -- EC 美育明细 JSON（参赛/观众/讲座/网课/拓展/展览）
    academic_tech_score REAL,        -- EC 学术与科技创新得分
    dorm_hygiene_score  REAL,        -- EC 宿舍卫生检查分（0~100）
    -- 分析结果：六维得分 + 综合分
    cluster_label       INTEGER,
    composite_score     REAL,
    score_study         REAL,
    score_health        REAL,
    score_social        REAL,
    score_life          REAL,
    score_morality      REAL,
    score_non_cog       REAL,
    risk_level          TEXT,
    risk_score          REAL,
    -- 分维度等级（一维 K-means 质心有序映射：待改进/合格/良好/优秀/卓越）
    level_overall       TEXT,        -- 综合等级（对综合分一维聚类）
    level_study         TEXT,
    level_health        TEXT,
    level_social        TEXT,
    level_life          TEXT,
    level_morality      TEXT,
    level_non_cog       TEXT,
    -- 多维聚类的可读成长类型（全面发展型 / 学业偏科型 / 实践活跃型 …）
    cluster_type        TEXT,
    -- 手册判定结果（仅产出提醒，不产出状态标签）
    handbook_hits       TEXT,        -- 命中的规则编号 JSON，如 ["A1", "S2"]
    handbook_reminders  TEXT,        -- 提醒话术 JSON [{"type":"提醒","rule":"A1","text":"..."}]
    last_updated        TEXT
);

-- 分年级聚类模型元信息（质心 / 等级边界 / 类型名）
-- scope: 'multi'（多维分型）| 'composite'（综合分级）| 'dim:<维度键>'（单维分级）
CREATE TABLE IF NOT EXISTS cluster_meta (
    grade       INTEGER NOT NULL,
    scope       TEXT NOT NULL,
    k           INTEGER,
    centers     TEXT,       -- 升序质心（一维）或六维质心矩阵（multi）JSON
    boundaries  TEXT,       -- 相邻质心中点 JSON（一维分级边界，用于"距上一等级差多少分"）
    labels      TEXT,       -- 等级文本（一维）或成长类型名（multi）JSON
    details     TEXT,       -- 定 K 过程明细 JSON（tested_k / inertias / silhouettes / scores）
    update_time TEXT,
    PRIMARY KEY (grade, scope)
);

CREATE TABLE IF NOT EXISTS teachers (
    teacher_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    password_hash TEXT,
    managed_classes TEXT  -- 逗号分隔的班级名
);

CREATE TABLE IF NOT EXISTS teacher_notes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id   TEXT NOT NULL,
    teacher_id   TEXT NOT NULL,
    note         TEXT,
    mark_level   TEXT,        -- 高危 / 关注 / 正常
    update_time  TEXT
);

CREATE INDEX IF NOT EXISTS idx_students_class ON students(class_name);
CREATE INDEX IF NOT EXISTS idx_students_grade ON students(grade);
"""

# 默认演示账号（明文密码仅用于本地演示，生产应强制改密）
DEFAULT_TEACHERS = [
    ("T2023001", "王老师", "teacher123", "2023级-计算机1班,2023级-计算机2班"),
    ("T2023002", "李老师", "teacher123", "2024级-软件1班,2024级-软件2班"),
]

DEFAULT_STUDENT_PASSWORD = "stu123456"


def get_connection() -> sqlite3.Connection:
    """获取数据库连接，开启行级访问（返回 dict-like 行）。"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _simple_hash(password: str) -> str:
    """轻量密码哈希（演示用，生产请替换为 bcrypt/argon2）。"""
    import hashlib

    salt = "student_growth_salt"
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()


def init_database(force: bool = False) -> None:
    """初始化数据库：建表并写入默认教师账号。

    force=True 时会删除已有数据库重建（用于演示重置）。
    """
    if force and DB_PATH.exists():
        os.remove(DB_PATH)

    conn = get_connection()
    try:
        conn.executescript(SCHEMA_SQL)
        # 旧库轻量迁移：cluster_meta 缺 details 列时原位补列（无需整库重建）
        cur = conn.execute("PRAGMA table_info(cluster_meta)")
        cols = {row["name"] for row in cur.fetchall()}
        if cols and "details" not in cols:
            conn.execute("ALTER TABLE cluster_meta ADD COLUMN details TEXT")
        # 写入默认教师账号（若为空）
        cur = conn.execute("SELECT COUNT(*) AS c FROM teachers")
        if cur.fetchone()["c"] == 0:
            for tid, name, pwd, scope in DEFAULT_TEACHERS:
                conn.execute(
                    "INSERT INTO teachers (teacher_id, name, password_hash, managed_classes) "
                    "VALUES (?, ?, ?, ?)",
                    (tid, name, _simple_hash(pwd), scope),
                )
        conn.commit()
    finally:
        conn.close()


def hash_password(password: str) -> str:
    return _simple_hash(password)


def column_exists(table: str, column: str) -> bool:
    """检查表中是否存在某列（用于旧库 schema 兼容性检测）。"""
    if not DB_PATH.exists():
        return False
    conn = get_connection()
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return any(row["name"] == column for row in cur.fetchall())
    finally:
        conn.close()


def reset_database() -> None:
    """重置数据库（保留结构，清空数据）。"""
    if DB_PATH.exists():
        os.remove(DB_PATH)
    init_database()


if __name__ == "__main__":
    init_database(force=True)
    print(f"数据库已初始化：{DB_PATH}")
