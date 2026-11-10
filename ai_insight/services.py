"""ai_insight 学情分析服务（项目创新点：AI 大模型驱动分析与可视化解读）。

封装 F1~F5 五大 AI 功能，落实作品书 4.4.2 的可溯源三道防线：
  ① 事实卡片化：先把数据结构化为带唯一编号的事实卡片（llm_client.build_*_facts）；
  ② 强制引用 + 后校验：要求每条数值结论标注 [Fxx]，输出后规则校验并剔除无据表述；
  ③ 知识锚定：手册提醒卡片携带规则编号，建议须锚定可量化条款。

全程经脱敏网关，调用失败自动优雅降级；每个函数返回 (文本, 成功标志)。

质量保障机制：
  - 修复重试：首次输出被校验大量剔除时，把被剔除的句子作为反馈再调用一次，
    让模型补引用重写，避免「报告被拦腰删得只剩标题」；
  - 多轮对话：chat_answer 接收 history，支持追问与指代消解；
  - 专用链路：维度诊断不再复用「一两句话」的图表解读，而走独立的事实卡片与任务指令。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ai_insight import llm_client as client

# 证据脚本的分隔行：历史消息传回模型前据此剥离，省 token 且避免徽章干扰
_EVIDENCE_FOOTER_MARK = "\n---\n"


def _system(prompts: Dict[str, str], key: str, fallback: str) -> str:
    return prompts.get(key, fallback)


def _sanitize_history(history: Optional[List[Dict[str, str]]],
                      max_turns: int = 6, max_chars: int = 900) -> List[Dict[str, str]]:
    """规整对话历史：只保留近几轮、剥离证据脚注、截断超长消息。

    历史仅用于理解上下文，无需携带「引用证据」索引（每轮正文已引用过卡片）。
    """
    cleaned: List[Dict[str, str]] = []
    for m in history or []:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = str(m.get("content") or "")
        content = content.split(_EVIDENCE_FOOTER_MARK, 1)[0].strip()
        if not content:
            continue
        cleaned.append({"role": role, "content": content[:max_chars]})
    return cleaned[-max_turns:]


def _validated_call(system: str, facts: list, task_instruction: str,
                    fallback_text: str,
                    history: Optional[List[Dict[str, str]]] = None,
                    max_tokens: Optional[int] = None) -> Tuple[str, bool]:
    """通用「事实卡片 -> 强制引用 -> 后校验 ->（必要时）带反馈修复重试」调用链。

    - 首次输出校验后为空，或被剔除的无据表述 >= 3 条时，触发一次修复重试：
      把被剔除的句子反馈给模型，要求补 [Fxx] 引用后完整重写；
    - 模型不可用 / 两次输出经校验后均为空时，返回 (兜底文本, False)。
    """
    cards = client.facts_to_text(facts)
    user = (
        f"事实卡片（唯一可引用的数据源）：\n{cards}\n\n"
        f"{client.CITATION_INSTRUCTION}\n\n{task_instruction}"
    )
    hist = _sanitize_history(history)
    text = client._call(system, user, max_tokens=max_tokens, history=hist)
    if text is None:
        return fallback_text, False

    validated = client.validate_cited_answer(text, facts)
    clean = validated.get("text", "").strip()
    dropped = validated.get("dropped", []) or []

    # 修复重试：正文被删空或剔除句较多，说明模型没遵守引用要求——
    # 把被剔除的句子原样反馈，要求补引用重写一次（保留原结构与结论）。
    if (not clean or len(dropped) >= 3) and dropped:
        feedback = (
            "你上一次的输出中，下列表述因含数值 / 等级断言但缺少 [Fxx] 证据引用"
            "被系统剔除：\n"
            + "\n".join(f"- {d}" for d in dropped[:8])
            + "\n\n请重新输出完整内容：保留原有结构与结论不变，"
              "为每条含数值或等级判断的句子在句末补上对应事实卡片编号（如 [F01]）。"
              "只输出修正后的正文，不要解释。"
        )
        text2 = client._call(system, f"{user}\n\n{feedback}",
                             max_tokens=max_tokens, history=hist)
        if text2:
            v2 = client.validate_cited_answer(text2, facts)
            clean2 = v2.get("text", "").strip()
            if len(clean2) > len(clean):
                validated, clean = v2, clean2

    if not clean:
        return fallback_text, False
    return client.append_evidence_footer(validated, facts), True


# ---------------------------------------------------------------------------
# F1 学生成长诊断
# ---------------------------------------------------------------------------
def generate_student_diagnosis(record: Dict[str, Any]) -> Tuple[str, bool]:
    """F1 学生端 AI 成长诊断报告（结构化五章报告，结论可溯源）。"""
    prompts = client.load_prompts()
    system = _system(
        prompts, "student_diagnosis_system",
        "你是高校学生成长分析助手。仅依据给定事实卡片，用客观、专业的语言生成结构化的成长诊断报告，"
        "包含总体研判、六维解读、优势、短板归因与分阶段提升行动方案。",
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

    # 报告篇幅较长：放宽 max_tokens，避免思考占额导致正文截断
    return _validated_call(
        system, facts,
        "请严格按系统提示词规定的五章结构，输出完整的「成长诊断报告」，"
        "每个章节标题下的内容不得为空。",
        _fallback(),
        max_tokens=4000,
    )


# ---------------------------------------------------------------------------
# F2 班级 / 年级研判
# ---------------------------------------------------------------------------
def generate_class_summary(agg: Dict[str, Any]) -> Tuple[str, bool]:
    """F2 教师端 AI 班级 / 年级研判摘要（基于聚合事实卡片）。"""
    prompts = client.load_prompts()
    system = _system(
        prompts, "class_summary_system",
        "你是高校辅导员学情分析助手。仅依据给定聚合事实卡片生成结构化研判摘要，"
        "指出偏弱模块并给出可落地的育人建议。",
    )
    facts = client.build_class_facts(agg)
    return _validated_call(
        system, facts,
        "请严格按系统提示词规定的四节结构，输出完整的「班级研判摘要」。",
        "（AI 未启用或调用失败，无法生成班级研判摘要，请配置 config_setting/llm_config.yaml）",
        max_tokens=3000,
    )


# ---------------------------------------------------------------------------
# F3 图表解读
# ---------------------------------------------------------------------------
def generate_chart_interpret(record: Dict[str, Any], chart_type: str) -> Tuple[str, bool]:
    """F3 图表 AI 文字解读（雷达图 / 趋势图，结论标注证据编号）。"""
    prompts = client.load_prompts()
    system = _system(
        prompts, "chart_interpret_system",
        "你是数据解读助手。仅依据给定事实卡片，用三五句话专业解读图表，"
        "说明各维度相对强弱并给出具体差值。",
    )
    try:
        from local_database import orm_ops
        grade_means = orm_ops.get_dimension_means(grade=record.get("grade"))
    except Exception:
        grade_means = None
    facts = client.build_student_facts(record, grade_means=grade_means)
    return _validated_call(
        system, facts,
        f"图表类型：{chart_type}。请按系统提示词要求给出客观准确的文字解读，"
        f"说明各维度的相对强弱与具体差值。",
        "（AI 解读暂不可用，已为你保留图表原始数据）",
    )


# ---------------------------------------------------------------------------
# F3+ 维度专项诊断（维度分页面专用，独立于图表解读）
# ---------------------------------------------------------------------------
def generate_dimension_diagnosis(record: Dict[str, Any], dim_key: str,
                                 dim_score: float, dim_level: str,
                                 sub_scores: Dict[str, float]) -> Tuple[str, bool]:
    """单个维度的 AI 专项诊断：聚焦该维度的二级指标、手册条款与横向对比。

    旧版此处复用 F3 图表解读链路，被「一两句话」的要求束缚，输出单薄；
    现改为独立链路：事实卡片注入该维度全部二级指标得分 + 维度均分对比。
    """
    from config_setting import DIMENSION_LABELS, DIMENSION_SUBSCORES

    prompts = client.load_prompts()
    label = DIMENSION_LABELS.get(dim_key, dim_key)
    system = _system(
        prompts, "dimension_diagnosis_system",
        f"你是高校学情分析助手。仅依据给定事实卡片，对一个学生的「{label}」维度"
        f"做专项诊断：先总体判断，再逐项分析二级指标强弱，最后给出 2~3 条"
        f"针对最弱项的可执行提升建议。含数值的结论标注卡片编号，"
        f"不编造数据、不出现任何个人信息。",
    )

    # 通用卡片：综合 / 六维 / 排名 / 风险 / 手册提醒（build_student_facts 自带）
    class_means = grade_means = None
    try:
        from local_database import orm_ops
        class_means = orm_ops.get_dimension_means(class_name=record.get("class_name"))
        grade_means = orm_ops.get_dimension_means(grade=record.get("grade"))
    except Exception:
        pass
    facts = client.build_student_facts(record, class_means=class_means,
                                       grade_means=grade_means)

    # 追加该维度全部二级指标卡片（F80+）
    from analysis_core import handbook_rules as hb_rules
    enriched = hb_rules.enrich_record(dict(record))
    next_fid = 80
    grade_field_means: Dict[str, float] = {}
    try:  # 本地统计年级各二级指标原始字段均值，用于逐项对比
        from local_database import orm_ops as _ops
        peers = _ops.get_raw_students_by_grade(int(record.get("grade") or 0))
        for (_sl, _w, field, _lo, _hi, _h) in DIMENSION_SUBSCORES.get(dim_key, []):
            vals = [float(p.get(field)) for p in peers if p.get(field) is not None]
            if vals:
                grade_field_means[field] = sum(vals) / len(vals)
    except Exception:
        grade_field_means = {}
    for (sub_label, w, field, _lo, _hi, higher) in DIMENSION_SUBSCORES.get(dim_key, []):
        s = float(sub_scores.get(sub_label, 0.0) or 0.0)
        text = f"{label}·{sub_label} 得分 {s:.1f} 分（权重 {w * 100:.0f}%"
        if not higher:
            text += "，反向计分"
        text += "）"
        # 原始值对比（同为原始口径才能相减，不能拿归一化得分与原始均值比）
        gm = grade_field_means.get(field)
        try:
            raw_v = enriched.get(field)
            if raw_v is not None and gm is not None:
                raw_f, gm_f = float(raw_v), float(gm)
                diff = raw_f - gm_f
                good = (diff > 0) if higher else (diff < 0)
                text += (f"，原始值 {raw_f:.2f} 较年级均值 {gm_f:.2f} "
                         f"{'占优' if good else '落后'} {abs(diff):.2f}")
        except (TypeError, ValueError):
            pass
        facts.append({"id": f"F{next_fid}", "text": text})
        next_fid += 1

    task = (
        f"待诊断维度：{label}（得分 {float(dim_score):.1f} 分，等级「{dim_level}」）。\n"
        f"请输出结构化诊断：\n"
        f"**总体判断**（2~3 句，含等级与年级位置）；\n"
        f"**二级指标逐项分析**（从强到弱，逐项给分并点评）；\n"
        f"**提升建议**（2~3 条，聚焦最弱的 1~2 个二级指标，"
        f"每条含具体做法与频次 / 数量目标，依据手册提醒时标注 [F4x]）。"
    )
    return _validated_call(
        system, facts, task,
        "（AI 维度诊断暂不可用，请参考下方规则式维度建议）",
        max_tokens=2600,
    )


# ---------------------------------------------------------------------------
# F4 高危帮扶话术
# ---------------------------------------------------------------------------
def generate_counsel_tips(record: Dict[str, Any]) -> Tuple[str, bool]:
    """F4 高危 / 关注学生干预建议与帮扶话术（可溯源、不标签化）。"""
    prompts = client.load_prompts()
    system = _system(
        prompts, "counsel_tips_system",
        "你是高校心理 / 学业帮扶助手。仅依据给定事实卡片，为辅导员生成"
        "「状态速览 → 沟通话术 → 分级干预建议 → 注意事项」的帮扶方案，"
        "语气温和、避免标签化。",
    )
    facts = client.build_student_facts(record)
    return _validated_call(
        system, facts,
        "请严格按系统提示词规定的四节结构输出帮扶方案；沟通话术要能直接照读。",
        "（AI 帮扶话术暂不可用，请辅导员结合风险指标人工研判）",
        max_tokens=3000,
    )


# ---------------------------------------------------------------------------
# F5 对话式查询
# ---------------------------------------------------------------------------
def chat_answer(question: str, safe_context: Dict[str, Any],
                history: Optional[List[Dict[str, str]]] = None) -> Tuple[str, bool]:
    """F5 对话式查询：仅基于脱敏事实卡片 / 聚合卡片回答，支持多轮。

    safe_context 为学生脱敏记录（经网关）或班级聚合统计，无任何 PII；
    亦可在 safe_context["fact_cards"] 预置现成卡片（学生端构建，绕开网关对
    手册提醒字段的裁剪）。history 为近几轮 {"role", "content"} 列表，
    用于理解「那第二呢」「为什么」这类追问。
    """
    prompts = client.load_prompts()
    system = _system(
        prompts, "chat_system",
        "你是高校学生成长助手，仅基于提供的事实卡片与对话历史回答问题，"
        "不编造、不泄露个人信息；方法类问题可给出具体建议但数值必须来自卡片。",
    )

    preset_cards = safe_context.get("fact_cards") if isinstance(safe_context, dict) else None
    if preset_cards:
        facts = list(preset_cards)
    elif "dimension_means" in safe_context or "risk_distribution" in safe_context:
        facts = client.build_class_facts(safe_context)
    else:
        facts = client.build_student_facts(safe_context)

    cards = client.facts_to_text(facts)
    user = (
        f"事实卡片（唯一可引用的数据源）：\n{cards}\n\n"
        f"{client.CITATION_INSTRUCTION}\n若卡片不足以回答，先答能答的部分再说明缺口。\n\n"
        f"用户提问：{question}"
    )
    text = client._call(system, user, history=_sanitize_history(history))
    if text is None:
        return "（AI 对话暂不可用，请检查 config_setting/llm_config.yaml 是否已正确配置云端 API）", False
    validated = client.validate_cited_answer(text, facts)
    clean = validated.get("text", "").strip()
    if not clean:
        # 校验后正文为空：先尝试一次修复重试，仍失败再给出引导话术
        retry = client._call(
            system,
            f"{user}\n\n注意：你上一次的回复未通过证据校验（没有任何 [Fxx] 引用）。"
            f"请重新回答本题：每条含数值或等级判断的句子句末标注卡片编号（如 [F01]）。",
            history=_sanitize_history(history),
        )
        if retry:
            validated = client.validate_cited_answer(retry, facts)
            clean = validated.get("text", "").strip()
        if not clean:
            return (
                "这次没能给出可溯源的回答（回复里没有可用的事实卡片引用）。"
                "可以换个更具体的问法，例如「本班哪个维度最弱」「高风险学生有多少人」。",
                True,
            )
    return client.append_evidence_footer(validated, facts), True
