"""ai_insight 学情分析服务（项目创新点：AI 大模型驱动分析与可视化解读）。

封装 F1~F5 五大 AI 功能，落实作品书 4.4.2 的可溯源三道防线：
  ① 事实卡片化：先把数据结构化为带唯一编号的事实卡片（llm_client.build_*_facts）；
  ② 强制引用 + 后校验：要求每条数值结论标注 [Fxx]，输出后规则校验并剔除无据表述；
  ③ 知识锚定：手册提醒卡片携带规则编号，建议须锚定可量化条款。

全程经脱敏网关，调用失败自动优雅降级；每个函数返回 (文本, 成功标志)。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ai_insight import llm_client as client


def _system(prompts: Dict[str, str], key: str, fallback: str) -> str:
    return prompts.get(key, fallback)


def _validated_call(system: str, facts: list, task_instruction: str,
                    fallback_text: str) -> Tuple[str, bool]:
    """通用「事实卡片 -> 强制引用 -> 后校验」调用链。

    模型不可用 / 输出经校验后为空时，返回 (兜底文本, False)。
    """
    cards = client.facts_to_text(facts)
    user = (
        f"事实卡片（唯一可引用的数据源）：\n{cards}\n\n"
        f"{client.CITATION_INSTRUCTION}\n\n{task_instruction}"
    )
    text = client._call(system, user)
    if text is None:
        return fallback_text, False
    validated = client.validate_cited_answer(text, facts)
    clean = validated.get("text", "").strip()
    if not clean:
        return fallback_text, False
    return client.append_evidence_footer(validated, facts), True


# ---------------------------------------------------------------------------
# F1 学生成长诊断
# ---------------------------------------------------------------------------
def generate_student_diagnosis(record: Dict[str, Any]) -> Tuple[str, bool]:
    """F1 学生端 AI 成长诊断报告（自然语言 + 提升建议，结论可溯源）。"""
    prompts = client.load_prompts()
    system = _system(
        prompts, "student_diagnosis_system",
        "你是高校学生成长分析助手。仅依据给定事实卡片，用客观、专业的语言生成成长诊断报告，"
        "包含综合概述、优势、短板与提升建议。",
    )

    class_means = grade_means = None
    try:  # 本地聚合，用于事实卡片中的对比项，不上云原始记录
        from local_database import orm_ops
        class_means = orm_ops.get_dimension_means(class_name=record.get("class_name"))
        grade_means = orm_ops.get_dimension_means(grade=record.get("grade"))
    except Exception:
        pass

    facts = client.build_student_facts(
        record, class_means=class_means, grade_means=grade_means
    )

    def _fallback() -> str:
        from analysis_core.pipeline import get_personalized_advice
        return "\n".join(get_personalized_advice(record))

    return _validated_call(
        system, facts,
        "请生成一份结构化的「成长诊断报告」：综合概述、优势维度、短板维度、可操作的提升建议。",
        _fallback(),
    )


# ---------------------------------------------------------------------------
# F2 班级 / 年级研判
# ---------------------------------------------------------------------------
def generate_class_summary(agg: Dict[str, Any]) -> Tuple[str, bool]:
    """F2 教师端 AI 班级 / 年级研判摘要（基于聚合事实卡片）。"""
    prompts = client.load_prompts()
    system = _system(
        prompts, "class_summary_system",
        "你是高校辅导员学情分析助手。仅依据给定聚合事实卡片生成研判摘要，"
        "指出偏弱模块并给出育人建议。",
    )
    facts = client.build_class_facts(agg)
    return _validated_call(
        system, facts,
        "请先概括整体态势，再指出相对偏弱的模块，最后给出 2~3 条可操作的育人建议。",
        "（AI 未启用或调用失败，无法生成班级研判摘要，请配置 config_setting/llm_config.yaml）",
    )


# ---------------------------------------------------------------------------
# F3 图表解读
# ---------------------------------------------------------------------------
def generate_chart_interpret(record: Dict[str, Any], chart_type: str) -> Tuple[str, bool]:
    """F3 图表 AI 文字解读（雷达图 / 趋势图，结论标注证据编号）。"""
    prompts = client.load_prompts()
    system = _system(
        prompts, "chart_interpret_system",
        "你是数据解读助手。仅依据给定事实卡片，用一两句话专业解释图表含义。",
    )
    try:
        from local_database import orm_ops
        grade_means = orm_ops.get_dimension_means(grade=record.get("grade"))
    except Exception:
        grade_means = None
    facts = client.build_student_facts(record, grade_means=grade_means)
    return _validated_call(
        system, facts,
        f"图表类型：{chart_type}。请给出客观准确的文字解读，说明各维度的相对强弱。",
        "（AI 解读暂不可用，已为你保留图表原始数据）",
    )


# ---------------------------------------------------------------------------
# F4 高危帮扶话术
# ---------------------------------------------------------------------------
def generate_counsel_tips(record: Dict[str, Any]) -> Tuple[str, bool]:
    """F4 高危 / 关注学生干预建议与帮扶话术（可溯源、不标签化）。"""
    prompts = client.load_prompts()
    system = _system(
        prompts, "counsel_tips_system",
        "你是高校心理 / 学业帮扶助手。仅依据给定事实卡片，生成温和、专业的干预建议与帮扶话术。",
    )
    facts = client.build_student_facts(record)
    return _validated_call(
        system, facts,
        "请结合风险信号与手册提醒，给出沟通话术与 2~3 条干预建议，语气关怀、避免标签化。",
        "（AI 帮扶话术暂不可用，请辅导员结合风险指标人工研判）",
    )


# ---------------------------------------------------------------------------
# F5 对话式查询
# ---------------------------------------------------------------------------
def chat_answer(question: str, safe_context: Dict[str, Any]) -> Tuple[str, bool]:
    """F5 对话式查询：仅基于脱敏事实卡片 / 聚合卡片回答。

    safe_context 为学生脱敏记录（经网关）或班级聚合统计，无任何 PII。
    """
    prompts = client.load_prompts()
    system = _system(
        prompts, "chat_system",
        "你是高校学生成长助手，仅基于提供的事实卡片回答问题，不编造、不泄露个人信息。",
    )
    if "dimension_means" in safe_context or "risk_distribution" in safe_context:
        facts = client.build_class_facts(safe_context)
    else:
        facts = client.build_student_facts(safe_context)

    cards = client.facts_to_text(facts)
    user = (
        f"事实卡片（唯一可引用的数据源）：\n{cards}\n\n"
        f"{client.CITATION_INSTRUCTION}\n若卡片不足以回答，请如实说明。\n\n"
        f"用户提问：{question}"
    )
    text = client._call(system, user)
    if text is None:
        return "（AI 对话暂不可用，请检查 config_setting/llm_config.yaml 是否已正确配置云端 API）", False
    validated = client.validate_cited_answer(text, facts)
    if not validated.get("text", "").strip():
        # 校验后正文为空：明确回话，不要返回空白气泡
        # （此前算出的 clean 变量没被使用，导致助手消息渲染成一片空白）
        return (
            "这次没能给出可溯源的回答（回复里没有可用的事实卡片引用）。"
            "可以换个更具体的问法，例如「本班哪个维度最弱」「高风险学生有多少人」。",
            True,
        )
    return client.append_evidence_footer(validated, facts), True
