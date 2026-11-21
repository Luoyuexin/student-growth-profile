"""data_source 数据集适配器层

统一契约：任何数据源只需实现 ``DatasetAdapter.load() -> list[dict]``，
产出的记录字段与 ``data_source.generate_mock_data.FIELDS`` 字段契约对齐，
即可交给 ``analysis_core.pipeline.run_pipeline(records=...)`` 完成全量分析。
分析层因此与具体数据源解耦，替换数据源不需要改动评分 / 聚类代码。

当前状态：
- 已实现：MockDatasetAdapter（团队自建模拟数据）、CsvDatasetAdapter（手动导入 CSV）。
- 仅作设计预留、尚未实现自动同步：教务系统 / 一卡通 / 第二课堂平台适配器。
  真实接入须在合规授权框架下完成，不在本版本范围内；不调取校内真实隐私数据。
"""

from __future__ import annotations

import csv
import io
import os
import random
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

# 保证既能被 Streamlit 入口导入，也能作为命令行脚本独立运行
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_source import generate_mock_data as _mock  # noqa: E402

# 统一字段契约（68 字段），与模拟数据生成脚本、orm_ops.STUDENT_COLUMNS 对齐
FIELD_CONTRACT: List[str] = list(_mock.FIELDS)

DATA_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = DATA_DIR / "import_template.csv"


class DatasetAdapter(ABC):
    """数据源适配器抽象基类。"""

    name: str = "base"
    source_label: str = "未指定数据源"
    implemented: bool = True

    @abstractmethod
    def load(self, *args: Any, **kwargs: Any) -> List[Dict]:
        """返回符合 FIELD_CONTRACT 字段契约的学生记录列表。"""

    def describe(self) -> Dict:
        return {
            "name": self.name,
            "source_label": self.source_label,
            "implemented": self.implemented,
        }


class MockDatasetAdapter(DatasetAdapter):
    """团队自建模拟学情数据集（240 条 / 68 字段 / 固定随机种子）。

    仅生成内存记录，不写库、不触发分析流水线，写库由 data_import 统一负责。
    """

    name = "mock_dataset"
    source_label = "团队自建模拟学情数据集（240 条 / 68 字段，固定随机种子）"
    implemented = True

    def load(self, per_class: int = 60) -> List[Dict]:
        import numpy as np

        random.seed(20260811)
        np.random.seed(20260811)
        return _mock.generate_dataset(per_class=per_class)


class CsvDatasetAdapter(DatasetAdapter):
    """手动导入 CSV 数据集（当前版本的数据入口）。

    只负责读取与基础解析，字段校验、缺失填充、落库与重算由 data_import 完成。
    """

    name = "csv_file"
    source_label = "手动导入 CSV 数据集"
    implemented = True

    def load(self, data: bytes) -> List[Dict]:
        text = data.decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        return [dict(row) for row in reader]

    def fieldnames(self, data: bytes) -> List[str]:
        text = data.decode("utf-8-sig", errors="replace")
        return list(csv.DictReader(io.StringIO(text)).fieldnames or [])


class AcademicSystemAdapter(DatasetAdapter):
    """教务管理系统适配器（设计预留，尚未实现自动同步）。"""

    name = "academic_system"
    source_label = "教务管理系统（成绩 / GPA / 学分 / 考勤）"
    implemented = False

    def load(self, *args: Any, **kwargs: Any) -> List[Dict]:
        raise NotImplementedError(
            "教务系统接口按长期规划预留，尚未实现自动同步；"
            "当前请使用 CsvDatasetAdapter 手动导入数据集。真实接入需合规授权。"
        )


class CampusCardAdapter(DatasetAdapter):
    """校园一卡通适配器（设计预留，尚未实现自动同步）。"""

    name = "campus_card"
    source_label = "校园一卡通（消费 / 食堂 / 门禁 / 图书馆）"
    implemented = False

    def load(self, *args: Any, **kwargs: Any) -> List[Dict]:
        raise NotImplementedError(
            "一卡通接口按长期规划预留，尚未实现自动同步；"
            "当前请使用 CsvDatasetAdapter 手动导入数据集。真实接入需合规授权。"
        )


class SecondClassAdapter(DatasetAdapter):
    """第二课堂活动平台适配器（设计预留，尚未实现自动同步）。"""

    name = "second_class"
    source_label = "第二课堂活动平台（活动参与 / 积分）"
    implemented = False

    def load(self, *args: Any, **kwargs: Any) -> List[Dict]:
        raise NotImplementedError(
            "第二课堂平台接口按长期规划预留，尚未实现自动同步；"
            "当前请使用 CsvDatasetAdapter 手动导入数据集。真实接入需合规授权。"
        )


ADAPTER_REGISTRY = [
    MockDatasetAdapter,
    CsvDatasetAdapter,
    AcademicSystemAdapter,
    CampusCardAdapter,
    SecondClassAdapter,
]


def list_adapters() -> List[Dict]:
    """返回适配器清单及实现状态，供界面 / 文档展示，避免口头声称「已接入」。"""
    return [cls().describe() for cls in ADAPTER_REGISTRY]
