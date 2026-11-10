"""data_source CSV 手动导入模块（对应《作品书》5.3「当前手动导入数据集」）。

职责：解析 CSV -> 字段校验与规范化 -> 备份数据库 -> 写入 SQLite ->
调用 analysis_core.pipeline.run_pipeline 全量重算六维评分 / 聚类 / 风险等级。

设计约束（与两份 PDF 口径一致）：
- 全程本地处理，不调取校内真实隐私数据；当前演示数据为团队自建模拟数据集。
- 绝不调用 generate_mock_data.main()（其内部 init_database(force=True) 会删库重建）。
- 分析结果字段一律由后端重算，不信任上传文件里的同名列。
- 教务 / 一卡通 / 第二课堂接口仅为适配器预留，不在此实现自动同步。
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis_core import pipeline as analysis_pipeline  # noqa: E402
from analysis_core.data_clean import NUMERIC_FIELDS  # noqa: E402
from config_setting import TEACHER_CLASS_SCOPE  # noqa: E402
from data_source.adapters import (  # noqa: E402
    CsvDatasetAdapter, FIELD_CONTRACT, TEMPLATE_PATH,
)
from local_database import db_init, orm_ops  # noqa: E402

# 必填字段：缺一即阻断导入
REQUIRED_FIELDS = ["student_id", "name", "grade", "class_name"]

# 手册硬规则依赖的 JSON 字段，缺失时补空结构，保证判定引擎不报错
JSON_DEFAULTS = {
    "campus_run_records": "[]",
    "fitness_history": "{}",
    "aesthetic_detail": "{}",
}

# 只允许后端计算，上传文件里的同名列一律清空
ANALYSIS_ONLY_FIELDS = ["handbook_hits", "handbook_reminders"]

# 分年级 K-means 的建议最小样本量（作品书 3.4.3：按年级独立建模）
MIN_GRADE_SAMPLES = 30


def _known_classes() -> set:
    """当前教师账号可管辖的班级名集合，用于提示「导入后教师端看不到」。"""
    try:
        conn = db_init.get_connection()
        try:
            rows = conn.execute("SELECT managed_classes FROM teachers").fetchall()
        finally:
            conn.close()
        classes = {
            c.strip()
            for row in rows
            for c in (row[0] or "").split(",")
            if c.strip()
        }
        if classes:
            return classes
    except Exception:
        pass
    return {c for scope in TEACHER_CLASS_SCOPE.values() for c in scope}


def parse_csv_bytes(data: bytes) -> Tuple[List[Dict], List[str]]:
    """解析上传的 CSV 字节流，返回原始行与表头。"""
    adapter = CsvDatasetAdapter()
    return adapter.load(data), adapter.fieldnames(data)


def validate_records(rows: List[Dict], fieldnames: List[str]) -> Dict:
    """字段校验与规范化。errors 阻断导入，warnings 仅提示。"""
    errors: List[str] = []
    warnings: List[str] = []
    fn = [c.strip() for c in (fieldnames or []) if c and c.strip()]
    fn_set = set(fn)

    missing_required = [c for c in REQUIRED_FIELDS if c not in fn_set]
    if missing_required:
        errors.append("缺少必填列：" + "、".join(missing_required))
    if not rows:
        errors.append("CSV 中没有数据行。")

    unknown = [c for c in fn if c not in FIELD_CONTRACT]
    if unknown:
        warnings.append("已忽略非契约字段 " + str(len(unknown)) + " 个：" + "、".join(unknown[:8]))

    missing_optional = [c for c in FIELD_CONTRACT if c not in fn_set and c not in REQUIRED_FIELDS]
    if missing_optional:
        warnings.append(
            "缺少 " + str(len(missing_optional)) + " 个可选字段，"
            "将按 data_clean 规则填充（活动类补 0、其余按列中位数）。"
        )

    records: List[Dict] = []
    seen = set()
    for line_no, row in enumerate(rows, start=2):
        rec: Dict[str, Any] = {}
        for col in FIELD_CONTRACT:
            value = row.get(col, "")
            if isinstance(value, str):
                value = value.strip()
            rec[col] = None if value in (None, "") else value

        for col in REQUIRED_FIELDS:
            if not rec.get(col):
                errors.append("第 " + str(line_no) + " 行缺少必填值：" + col)

        sid = str(rec.get("student_id") or "")
        if sid and sid in seen:
            errors.append("学号重复：" + sid)
        if sid:
            seen.add(sid)

        try:
            rec["grade"] = int(float(rec["grade"]))
        except (TypeError, ValueError):
            errors.append("第 " + str(line_no) + " 行 grade 不是数字。")
            rec["grade"] = None

        for col in NUMERIC_FIELDS:
            value = rec.get(col)
            if value is None:
                continue
            try:
                rec[col] = float(value)
            except (TypeError, ValueError):
                warnings.append("第 " + str(line_no) + " 行 " + col + " 非数值，按缺失处理。")
                rec[col] = None

        for col, default in JSON_DEFAULTS.items():
            if not rec.get(col):
                rec[col] = default

        if not rec.get("password_hash"):
            # 统一写入演示账号密码的哈希；明文不打印、不回显
            rec["password_hash"] = db_init.hash_password(db_init.DEFAULT_STUDENT_PASSWORD)

        for col in ANALYSIS_ONLY_FIELDS:
            rec[col] = None

        records.append(rec)

    known = _known_classes()
    new_classes = sorted({
        str(r.get("class_name"))
        for r in records
        if r.get("class_name") and str(r.get("class_name")) not in known
    })
    if new_classes:
        warnings.append(
            "以下班级不在当前教师管辖范围，导入后教师端将看不到："
            + "、".join(new_classes[:8])
            + "。请同步 teachers.managed_classes 与 config_setting.TEACHER_CLASS_SCOPE。"
        )

    grade_counts = Counter(int(r["grade"]) for r in records if r.get("grade"))
    for grade, count in sorted(grade_counts.items()):
        if count < MIN_GRADE_SAMPLES:
            warnings.append(
                str(grade) + " 级仅 " + str(count) + " 人：分年级 K-means "
                "建议每年级不少于 " + str(MIN_GRADE_SAMPLES) + " 人，样本过少可能导致定 K 失真。"
            )

    return {
        "ok": len(errors) == 0,
        "errors": errors[:20],
        "warnings": warnings,
        "records": records,
        "row_count": len(records),
        "missing_optional": missing_optional,
        "unknown": unknown,
        "new_classes": new_classes,
        "grade_counts": dict(grade_counts),
    }


def inspect_upload(data: bytes) -> Dict:
    """供网页预览：解析 + 校验，不产生任何写操作。"""
    rows, fieldnames = parse_csv_bytes(data)
    result = validate_records(rows, fieldnames)
    result["fieldnames"] = fieldnames
    return result


def backup_database() -> Path:
    """写库前备份 SQLite 文件（带时间戳，已被 .gitignore 忽略）。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = db_init.DB_PATH.with_name("student_growth_备份_" + stamp + ".db")
    if db_init.DB_PATH.exists():
        shutil.copy2(db_init.DB_PATH, target)
    return target


def import_records(records: List[Dict], mode: str = "upsert") -> Dict:
    """落库并重算画像。mode: upsert（按学号追加/更新）| replace（清空学生表）。"""
    if mode not in ("upsert", "replace"):
        raise ValueError("mode 只能是 upsert 或 replace")
    if not records:
        raise ValueError("没有可导入的记录")

    db_init.init_database()  # 只建表，不删库
    backup = backup_database()

    if mode == "replace":
        conn = db_init.get_connection()
        try:
            conn.execute("DELETE FROM students")
            conn.execute("DELETE FROM cluster_meta")
            # teachers / teacher_notes 保留：账号与帮扶记录不随数据替换清空
            conn.commit()
        finally:
            conn.close()

    orm_ops.bulk_upsert_students(records)
    analysis_pipeline.run_pipeline(save=True)

    return {
        "mode": mode,
        "rows": len(records),
        "backup": str(backup),
        "db": str(db_init.DB_PATH),
    }


def build_template_bytes() -> bytes:
    """生成 68 字段导入模板（UTF-8 BOM，Excel 可直接打开）。"""
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=FIELD_CONTRACT)
    writer.writeheader()
    sample = {c: "" for c in FIELD_CONTRACT}
    sample.update({
        "student_id": "202300000000",
        "name": "示例学生",
        "gender": "男",
        "grade": "2023",
        "class_name": "2023级-计算机1班",
        "major": "计算机科学与技术",
        "gpa": "3.20",
        "fail_count": "0",
        "attendance_rate": "0.95",
        "homework_timely_rate": "0.92",
        "credit_progress": "0.88",
        "library_visits": "80",
        "study_hours": "180",
        "psy_score": "78",
        "physical_test_score": "82",
        "second_class_count": "10",
        "volunteer_hours": "30",
        "canteen_regularity": "0.85",
        "dorm_regularity": "0.80",
        "consumption_daily": "28.5",
        "political_status": "85",
        "integrity_record": "1",
    })
    writer.writerow(sample)
    return ("\ufeff" + buf.getvalue()).encode("utf-8")


def write_template_file(path: Path = TEMPLATE_PATH) -> Path:
    """把导入模板写到 data_source/import_template.csv（随代码交付）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_template_bytes())
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="StuProfile CSV 手动导入（当前数据入口；教务等接口仅预留）"
    )
    parser.add_argument("csv_path", nargs="?", help="要导入的 CSV 路径；省略则只生成模板")
    parser.add_argument("--mode", choices=["upsert", "replace"], default="upsert",
                        help="upsert=按学号追加/更新（默认），replace=清空学生表后全量替换")
    parser.add_argument("--make-template", action="store_true", help="重新生成字段模板")
    args = parser.parse_args(argv)

    template = write_template_file()
    print("导入模板：" + str(template))
    if not args.csv_path:
        print("未指定 CSV，仅生成模板。用法：python data_source/data_import.py data.csv --mode upsert")
        return 0

    source = Path(args.csv_path)
    if not source.exists():
        print("找不到文件：" + str(source))
        return 2

    preview = inspect_upload(source.read_bytes())
    for warn in preview["warnings"]:
        print("[提示] " + warn)
    if not preview["ok"]:
        for err in preview["errors"]:
            print("[错误] " + err)
        print("校验未通过，未写入任何数据。")
        return 2

    print("校验通过：" + str(preview["row_count"]) + " 条记录，开始导入 ...")
    result = import_records(preview["records"], mode=args.mode)
    print("导入完成：" + str(result["rows"]) + " 条；已重算六维评分 / 聚类 / 风险等级。")
    print("数据库备份：" + result["backup"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
