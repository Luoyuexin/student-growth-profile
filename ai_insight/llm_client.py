"""ai_insight 云端大模型客户端。

本项目创新点：调用云端大语言模型（OpenAI 兼容接口）对学情数据做自然语言分析，
并将洞察嵌入雷达图 / 折线图等可视化呈现。

隐私合规（强制）：
- 所有发往云端的数据必先经 `safe_student_profile` / 聚合统计脱敏网关处理，
  剥离姓名、学号等 PII，仅传维度分值与班级 / 年级聚合统计。
- 即使调用方误传 PII，网关也会拦截。
- API Key 仅存于本地 `config_setting/llm_config.yaml`（已被 .gitignore 忽略），不进 git / 数据库。

优雅降级：
- 未启用 / 未配置 / 调用失败 时，所有生成函数返回 (文本, False)，由调用方回退规则式建议。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config_setting"
LLM_CONFIG_PATH = CONFIG_DIR / "llm_config.yaml"
PROMPTS_PATH = CONFIG_DIR / "prompts.yaml"

# 绝对禁止外传的字段（脱敏网关黑名单，作品书 4.4.3：四类个人标识字段完全剔除）
PII_FIELDS = {"name", "student_id", "password_hash", "gender"}

# 原始流水 / 自由文本大字段：即使不含 PII 也不直接外发，
# 只允许以「事实卡片」中的统计结果形式外传（作品书 4.4.2 防线①）。
RAW_BLOB_FIELDS = {
    "campus_run_records", "fitness_history", "aesthetic_detail",
    "handbook_reminders", "handbook_hits",
}

# 学生场景允许外发的字段白名单（分数字段 / 等级 / 类型 / 风险信号 / 聚合上下文）。
# services 层默认改走事实卡片；本白名单作为网关兜底，防止调用方误传原始记录。
STUDENT_SAFE_KEYS = {
    "composite_score", "risk_level", "risk_score", "cluster_type",
    "level_overall", "level_study", "level_health", "level_social",
    "level_life", "level_morality", "level_non_cog",
    "score_study", "score_health", "score_social", "score_life",
    "score_morality", "score_non_cog",
    "fail_count", "continuous_absence", "night_not_home",
    "consumption_drop", "grade_drop",
    "class_avg", "grade_avg", "risk_signals", "fact_cards",
}


@dataclass
class LLMConfig:
    enabled: bool = False
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.6
    max_tokens: int = 3000
    timeout: int = 90
    # 深度思考策略：auto（按模型自动判断）/ on（强制开启）/ off（关闭）
    # 注意：GLM-5.3、GLM-5.3-Flash 等推理模型不支持关闭思考，传 disabled 会直接报错，
    #       因此默认用 auto，由 _resolve_thinking() 按模型名判断。
    thinking: str = "auto"
    # 思考深度，仅部分 GLM 模型支持（low / high / max），留空则不发送该字段
    reasoning_effort: str = ""


# 已知「思考始终开启、不允许关闭」的模型名片段：这类模型若传 thinking={"type":"disabled"}
# 会被服务端直接拒绝，必须传 enabled（或干脆不传）。
THINKING_ALWAYS_ON = ("glm-5.3", "glm-5-3", "glm-5.2", "glm-5-2")


def _resolve_thinking(cfg: LLMConfig) -> Optional[Dict[str, Any]]:
    """按配置与模型名决定 thinking 字段；返回 None 表示不发送该字段。"""
    mode = (cfg.thinking or "auto").strip().lower()
    if mode == "off":
        return {"type": "disabled"}
    if mode == "on":
        return {"type": "enabled"}
    # auto：推理型模型强制开启，其余默认关闭以降低延迟
    model = (cfg.model or "").lower()
    if any(tag in model for tag in THINKING_ALWAYS_ON):
        return {"type": "enabled"}
    return {"type": "disabled"}


# 最近一次调用失败的原因（供界面与自测脚本展示，便于定位鉴权 / 模型未开通 / 限流等问题）
_LAST_ERROR: Optional[str] = None


def get_last_error() -> Optional[str]:
    """返回最近一次云端调用失败的原因；成功时为 None。"""
    return _LAST_ERROR


def _set_error(msg: Optional[str]) -> None:
    global _LAST_ERROR
    _LAST_ERROR = msg


def mask_key(key: str) -> str:
    """脱敏展示 API Key，避免在界面 / 日志中泄露完整密钥。"""
    key = (key or "").strip()
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}****{key[-4:]}"


def _parse_thinking(raw: Any) -> str:
    """解析 thinking 配置，兼容旧的布尔写法（true/false）。"""
    if isinstance(raw, bool):
        return "on" if raw else "off"
    text = str(raw or "").strip().lower()
    return text if text in ("auto", "on", "off") else "auto"


def load_llm_config() -> LLMConfig:
    """加载云端大模型配置；文件不存在时返回全默认（未启用）。"""
    if not LLM_CONFIG_PATH.exists():
        return LLMConfig()
    with open(LLM_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return LLMConfig(
        enabled=bool(data.get("enabled", False)),
        base_url=str(data.get("base_url", "")).strip(),
        api_key=str(data.get("api_key", "")).strip(),
        model=str(data.get("model", "")).strip(),
        temperature=float(data.get("temperature", 0.6)),
        max_tokens=int(data.get("max_tokens", 3000)),
        timeout=int(data.get("timeout", 90)),
        thinking=_parse_thinking(data.get("thinking", "auto")),
        reasoning_effort=str(data.get("reasoning_effort", "") or "").strip(),
    )


def load_prompts() -> Dict[str, str]:
    """加载提示词模板。"""
    if not PROMPTS_PATH.exists():
        return {}
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def is_enabled() -> bool:
    """云端大模型是否已正确配置。"""
    cfg = load_llm_config()
    return bool(cfg.enabled and cfg.base_url and cfg.api_key and cfg.model)


def safe_student_profile(record: Dict[str, Any]) -> Dict[str, Any]:
    """脱敏网关：剥离 PII 与原始流水大字段，仅保留白名单内的分值 / 等级上下文。

    作品书 4.4.3：姓名、学号、密码哈希、性别四类个人标识完全剔除；
    此外原始逐次记录（校园跑流水、体测历年 JSON、提醒话术等）不外发，
    只允许经事实卡片（build_student_facts）聚合后的统计结果上云。
    传入的 record 即使误含 PII，也会被拦截。None 值统一转为 0。
    """
    out = {}
    for k, v in record.items():
        if k in PII_FIELDS or k in RAW_BLOB_FIELDS:
            continue
        if k not in STUDENT_SAFE_KEYS:
            continue
        out[k] = 0 if v is None else _coerce(v)
    return out


def _coerce(obj: Any) -> Any:
    """将 numpy 等非原生类型转换为可 JSON 序列化的 Python 类型。"""
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _coerce(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce(v) for v in obj]
    return obj


def _as_text(value: Any) -> str:
    """把消息正文规整为纯字符串。

    防御性处理：Chat Completions 要求 content 为字符串。若上游误传
    list / tuple（例如字符串字面量后多写一个逗号，会变成单元素元组），
    服务端只会返回 400 InvalidParameter 且提示晦涩，难以定位。
    此处提前归一化，避免这类问题以"看起来像没配置"的方式静默失败。
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "".join(_as_text(v) for v in value)
    return "" if value is None else str(value)


def _create_payload(cfg: LLMConfig, system: str, user: str, max_tokens: Optional[int],
                    history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """组装一次对话请求体（OpenAI Chat Completions 兼容格式）。

    history：多轮对话历史（不含 system 与本轮 user），按时间升序，
             每项 {"role": "user"/"assistant", "content": "..."}。用于 F5 对话式查询，
             让模型理解追问与指代（如「那第二呢」「为什么」）。
    thinking：按模型自适应。GLM-5.3 / 5.3-Flash 等推理模型思考不可关闭，
              传 disabled 会被服务端拒绝，故对这类模型传 enabled。
    reasoning_effort：仅部分 GLM 模型支持（low / high / max），其他模型不发送。
    """
    messages: List[Dict[str, str]] = [{"role": "system", "content": _as_text(system)}]
    for m in history or []:
        role = "assistant" if m.get("role") == "assistant" else "user"
        content = _as_text(m.get("content"))
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": _as_text(user)})
    payload: Dict[str, Any] = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "max_tokens": int(max_tokens or cfg.max_tokens),
        "messages": messages,
    }
    thinking = _resolve_thinking(cfg)
    if thinking:
        payload["thinking"] = thinking
    if cfg.reasoning_effort:
        model = (cfg.model or "").lower()
        if any(tag in model for tag in ("glm-5.3", "glm-5-3", "glm-5.2", "glm-5-2")):
            payload["reasoning_effort"] = cfg.reasoning_effort
    return payload


def _http_post(url: str, headers: Dict[str, str], payload: Dict[str, Any],
               timeout: int) -> Tuple[int, str]:
    """最小 HTTP POST，返回 (状态码, 响应正文)。

    优先使用 requests（本机已具备）；若环境缺失则退回标准库 urllib，
    全程不依赖 openai SDK —— 避免 SDK 与 httpx 之间的版本冲突导致调用失败。
    """
    try:
        import requests  # type: ignore
    except Exception:
        requests = None  # type: ignore

    if requests is not None:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        return resp.status_code, resp.text

    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return int(r.status), r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return int(e.code), e.read().decode("utf-8", "replace")


# 常见 HTTP 状态码的中文排查提示
_HTTP_HINTS = {
    400: "请求参数被拒绝：模型 ID 或参数字段可能不被该模型支持",
    401: "鉴权失败：API Key 无效或格式不正确，请重新复制方舟控制台的 Key",
    403: "无权限或账号异常：请到方舟控制台检查模型开通状态与账号余额",
    404: "模型未开通或模型 ID 写错：请在方舟控制台「开通模型」后重试",
    429: "触发限流或免费额度已用尽：请稍后重试或检查额度",
    500: "服务端异常，请稍后重试",
    503: "服务端繁忙，请稍后重试",
}


def _describe_http_error(status: int, text: str) -> str:
    """把 HTTP 错误整理成可直接看懂的中文排查信息。"""
    hint = _HTTP_HINTS.get(status, "")
    try:
        err = (json.loads(text) or {}).get("error") or {}
        code = str(err.get("code", "") or "")
        msg = str(err.get("message", "") or "")
        detail = f"{code}: {msg}" if (code or msg) else text[:200]
    except Exception:
        detail = text[:200]
    parts = [f"HTTP {status}"]
    if hint:
        parts.append(hint)
    parts.append(f"服务端返回：{detail}")
    return " ｜ ".join(parts)


def _call(system: str, user: str, max_tokens: Optional[int] = None,
          history: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
    """调用云端 OpenAI 兼容接口（火山方舟 / DeepSeek / 通义等通用）。

    history：多轮对话历史（可选），见 _create_payload 说明。
    任何异常（网络、鉴权、模型未开通、限流、超时）均被捕获并返回 None，
    由上层做优雅降级；失败原因记录在 `get_last_error()` 中，便于快速定位。
    """
    cfg = load_llm_config()
    if not (cfg.enabled and cfg.base_url and cfg.api_key and cfg.model):
        _set_error(
            "未配置完整：请检查 config_setting/llm_config.yaml 的 "
            "enabled / base_url / api_key / model 四项是否均已填写"
        )
        return None

    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    payload = _create_payload(cfg, system, user, max_tokens, history)

    def _do_request(body: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """执行一次请求，返回 (文本, 错误说明)。"""
        try:
            status, text = _http_post(url, headers, body, cfg.timeout)
        except Exception as e:
            return None, f"网络请求失败：{type(e).__name__}: {e}"
        if status != 200:
            return None, _describe_http_error(status, text)
        try:
            data = json.loads(text)
            message = data["choices"][0]["message"]
            content = message.get("content")
            # 推理型模型（GLM-5.3 等）思考内容可能落在 reasoning_content，
            # 若正文为空则回退取它，避免误判为「返回内容为空」。
            if not content:
                content = message.get("reasoning_content")
        except Exception as e:
            return None, f"响应解析失败：{type(e).__name__}: {e} ｜ 原文：{text[:300]}"
        if not content:
            return None, "模型返回内容为空"
        return str(content).strip(), None

    text, err = _do_request(payload)
    if err is None:
        _set_error(None)
        return text

    # 兼容性回退：个别模型不识别 thinking / reasoning_effort 字段（返回 400），
    # 去掉这两个可选字段后重试一次，保证换模型时不至于完全不可用。
    optional = [k for k in ("thinking", "reasoning_effort") if k in payload]
    if optional and "400" in err:
        fallback = {k: v for k, v in payload.items() if k not in optional}
        text2, err2 = _do_request(fallback)
        if err2 is None:
            _set_error(None)
            return text2
        _set_error(f"{err} ｜ 去掉 {'/'.join(optional)} 参数重试后仍失败：{err2}")
        return None

    _set_error(err)
    return None


def test_connection() -> Tuple[bool, str]:
    """连通性自测：发送一次最小请求，返回 (是否成功, 说明信息)。

    失败信息会明确指出鉴权失败 / 模型未开通 / 限流 / 网络等具体原因，
    便于接入阶段快速定位问题。
    """
    cfg = load_llm_config()
    if not cfg.enabled:
        return False, "配置中 enabled 仍为 false，请改为 true 后重试"
    if not (cfg.base_url and cfg.api_key and cfg.model):
        return False, "base_url / api_key / model 存在空值，请补全后重试"

    text = _call("你是一个连通性测试助手。", "请只回复两个字：可用", max_tokens=32)
    if text:
        return True, (
            f"调用成功 ｜ 模型 {cfg.model} ｜ 密钥 {mask_key(cfg.api_key)} ｜ 返回：{text[:60]}"
        )
    return False, f"调用失败：{get_last_error()}"


def _json_block(data: Any) -> str:
    return json.dumps(_coerce(data), ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 可溯源机制（作品书 4.4.2 三道防线）
#   ① 事实卡片化：六维得分 / 等级 / 风险 / 手册命中项 -> 带唯一编号的卡片
#   ② 强制引用 + 后校验：要求模型每条结论标注 [Fxx]，输出后规则校验，
#      剔除「含数值断言却无有效证据引用」的句子
#   ③ 知识锚定：手册提醒卡片携带规则编号（A1 / P2 …），建议必须可回溯
# ---------------------------------------------------------------------------
import re as _re

_CITATION_RE = _re.compile(
    r"[【\[\(]\s*F\d{2,3}(?:\s*[,，、;；/]\s*F\d{2,3})*\s*[】\])]"
)
_CITATION_ID_RE = _re.compile(r"F\d{2,3}")


def _cited_ids(text: str) -> list:
    """取出文本中所有引用块内的卡片编号。

    兼容一个方括号里写多个编号的写法（[F01, F10] / 【F01、F10】）——
    模型很爱这么合并引用，早期只认单个 [F01]，会把这类句子判成「无引用」而整段删除。
    """
    return [
        i for block in _CITATION_RE.findall(text or "")
        for i in _CITATION_ID_RE.findall(block)
    ]
_NUMBER_CLAIM_RE = _re.compile(r"\d")

# 数据断言特征：「X 分 / X% / 第 X 名 / 等级」等对既有数据的量化陈述，
# 这类句子必须有 [Fxx] 证据引用，否则剔除。
_DATA_CLAIM_RE = _re.compile(
    r"\d+(?:\.\d+)?\s*(?:分(?!钟)|％|%|名(?!单|称)|位|GPA|绩点)"
    r"|等级[^。；;]{0,6}[待合良优正关高]"
    r"|较[^。；;]{0,8}(?:高|低|多|少)[^。；;]{0,6}\d"
)
# 建议性数量单位：次 / 小时 / 分钟 / 天 / 周 / 公里 / 本 / 篇 / 项 等行动建议中的数量，
# 属于给学生的行动建议而非数据断言，予以放行（修复「每周锻炼 3 次」被误杀的问题）。
_SUGGESTION_UNIT_RE = _re.compile(
    r"\d+(?:\.\d+)?\s*(?:次|小时|分钟|天|周|公里|千米|km|本|篇|项|条|个|门|组|步)"
)
# 建议语境提示词：出现这类措辞时，句子更像行动建议而非数据陈述。
_SUGGESTION_HINT_RE = _re.compile(
    r"建议|可以|不妨|尝试|坚持|争取|尽量|每周|每天|日常|安排|计划|参与|参加|保持|养成|目标定为"
)


def _fnum(v, default=0.0) -> float:
    """宽松转 float（None / 非数值返回 default），供卡片构建共用。"""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _rank_facts(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    """计算综合评分在年级 / 班级中的排名与百分位（本地统计，结果无 PII）。

    排名卡片让模型能回答「我排在什么位置」「前百分之多少」这类问题，
    也是诊断报告判断「相对位置」的关键依据；本地库缺失时静默跳过。
    """
    out: List[Tuple[str, str]] = []
    try:
        from local_database import orm_ops
        score = _fnum(record.get("composite_score"))
        grade = record.get("grade")
        if grade:
            peers = [float(p.get("composite_score") or 0)
                     for p in orm_ops.get_raw_students_by_grade(int(grade))]
            if len(peers) > 1:
                rank = sum(1 for s in peers if s > score) + 1
                pct = rank / len(peers) * 100
                out.append(("F05", f"综合评分在年级 {len(peers)} 名学生中位列第 {rank} 名"
                                   f"（约前 {pct:.0f}%）"))
        cls = record.get("class_name")
        if cls:
            peers = [float(p.get("composite_score") or 0)
                     for p in orm_ops.get_raw_students_by_class(cls)]
            if len(peers) > 1:
                rank = sum(1 for s in peers if s > score) + 1
                pct = rank / len(peers) * 100
                out.append(("F06", f"综合评分在班级 {len(peers)} 名学生中位列第 {rank} 名"
                                   f"（约前 {pct:.0f}%）"))
    except Exception:
        pass
    return out


def _sub_metric_facts(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    """每个维度取最强 / 最弱二级指标各一条（本地计算，结果无 PII）。

    仅六维总分时模型只能给「学业偏弱」这类笼统结论；加入二级指标强弱项后，
    报告才能落到「具体哪一项拖了后腿、强在哪一项」的粒度。
    """
    out: List[Tuple[str, str]] = []
    try:
        from analysis_core import handbook_rules as hb_rules
        from analysis_core import scoring as _scoring
        from config_setting import DIMENSION_LABELS as _DL, DIMENSION_ORDER
        enriched = hb_rules.enrich_record(dict(record))
        all_scores = _scoring.compute_dimension_scores(enriched)
        fid = 60
        for d in DIMENSION_ORDER:
            _dim_score, subs = all_scores.get(d, (0.0, {}))
            if not subs:
                continue
            best_label, best_v = max(subs.items(), key=lambda x: x[1])
            worst_label, worst_v = min(subs.items(), key=lambda x: x[1])
            if best_label == worst_label:
                continue
            out.append((f"F{fid}", f"二级指标优势项：{_DL[d]}·{best_label} {best_v:.1f} 分"
                                   f"（本维度最高）"))
            fid += 1
            out.append((f"F{fid}", f"二级指标薄弱项：{_DL[d]}·{worst_label} {worst_v:.1f} 分"
                                   f"（本维度最低）"))
            fid += 1
    except Exception:
        pass
    return out


def build_student_facts(
    record: Dict[str, Any],
    class_means: Optional[Dict[str, float]] = None,
    grade_means: Optional[Dict[str, float]] = None,
) -> List[Dict[str, str]]:
    """把一条学生记录结构化为带唯一编号的事实卡片。

    返回 [{"id": "F01", "text": "..."}, ...]，文本中已不含任何 PII。
    模型只被允许引用卡片中的既有数值，从源头压缩幻觉空间。
    卡片覆盖：综合 / 六维 / 排名百分位 / 风险信号 / 二级指标强弱项 / 手册提醒。
    """
    from config_setting import DIMENSION_ORDER, DIMENSION_LABELS

    def _f(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    facts: List[Dict[str, str]] = []

    def add(fid: str, text: str) -> None:
        facts.append({"id": fid, "text": text})

    add("F01", f"综合评分 {_f(record.get('composite_score')):.1f} 分（满分 100）")
    if record.get("level_overall"):
        add("F02", f"综合等级为「{record['level_overall']}」")
    add("F03", f"风险等级为「{record.get('risk_level', '正常')}」，"
               f"风险信号累计 {_f(record.get('risk_score')):.0f} 项")
    if record.get("cluster_type"):
        add("F04", f"成长类型聚类结果为「{record['cluster_type']}」")
    for fid, text in _rank_facts(record):
        add(fid, text)

    # 六维得分与等级（F10~F15）
    means_src = grade_means or class_means or {}
    for i, d in enumerate(DIMENSION_ORDER):
        fid = f"F{10 + i:d}"
        score = _f(record.get(f"score_{d}"))
        level = record.get(f"level_{d}")
        text = f"{DIMENSION_LABELS[d]}得分 {score:.1f} 分"
        if level:
            text += f"，等级「{level}」"
        if means_src and means_src.get(d) is not None:
            diff = score - _f(means_src.get(d))
            text += f"，较年级均值{'高' if diff >= 0 else '低'} {abs(diff):.1f} 分"
        add(fid, text)

    # 触发的风险信号（F20+）
    signals = []
    if _f(record.get("fail_count")) >= 2:
        signals.append(f"挂科 {int(_f(record.get('fail_count')))} 门")
    if _f(record.get("continuous_absence")) >= 5:
        signals.append(f"连续缺勤 {int(_f(record.get('continuous_absence')))} 天")
    if _f(record.get("night_not_home")) >= 3:
        signals.append(f"夜间未归 {int(_f(record.get('night_not_home')))} 次")
    if _f(record.get("consumption_drop")) >= 0.4:
        signals.append(f"消费骤减 {_f(record.get('consumption_drop')) * 100:.0f}%")
    if _f(record.get("grade_drop")) >= 0.25:
        signals.append(f"成绩环比下降 {_f(record.get('grade_drop')) * 100:.0f}%")
    for j, sig in enumerate(signals):
        add(f"F{20 + j}", f"风险信号：{sig}")

    # 二级指标强弱项（F60+，本地计算）
    for fid, text in _sub_metric_facts(record):
        add(fid, text)

    # 手册命中项（③ 知识锚定：携带规则编号与系统生成的客观话术，无 PII）
    hits = record.get("handbook_hits")
    reminders = record.get("handbook_reminders")
    try:
        if isinstance(hits, str):
            hits = json.loads(hits)
        if isinstance(reminders, str):
            reminders = json.loads(reminders)
    except (TypeError, ValueError):
        hits, reminders = [], []
    for k, item in enumerate(reminders or []):
        fid = f"F{40 + k}"
        rule = item.get("rule", "") if isinstance(item, dict) else ""
        text = item.get("text", "") if isinstance(item, dict) else str(item)
        add(fid, f"手册规则 {rule} 的系统提醒：{text}")

    return facts


def build_class_facts(agg: Dict[str, Any]) -> List[Dict[str, str]]:
    """把班级 / 年级聚合统计结构化为事实卡片（F2 研判 / F5 对话用）。

    卡片覆盖：样本规模、综合均分、各维均值、最强 / 最弱模块、
    风险分布、手册规则命中统计（agg 可选携带）。
    """
    facts: List[Dict[str, str]] = [
        {"id": "F01", "text": f"样本学生 {int(agg.get('student_count', 0))} 人"}
    ]
    comp = agg.get("composite_mean")
    if comp is not None:
        try:
            facts.append({"id": "F02",
                          "text": f"综合评分均值 {float(comp):.1f} 分（满分 100）"})
        except (TypeError, ValueError):
            pass
    means = agg.get("dimension_means", {}) or {}
    for i, (dim, val) in enumerate(means.items()):
        facts.append({"id": f"F{10 + i}", "text": f"{dim}平均 {float(val):.1f} 分"})
    # 最强 / 最弱模块：让研判摘要能直接「点名」短板而不是罗列一遍均值
    if len(means) >= 2:
        try:
            ordered = sorted(means.items(), key=lambda kv: float(kv[1]))
            weak_label, weak_v = ordered[0]
            strong_label, strong_v = ordered[-1]
            facts.append({"id": "F20",
                          "text": f"相对最弱模块：{weak_label}（平均 {float(weak_v):.1f} 分）"})
            facts.append({"id": "F21",
                          "text": f"相对最强模块：{strong_label}（平均 {float(strong_v):.1f} 分）"
                                  f"，与最弱模块相差 {float(strong_v) - float(weak_v):.1f} 分"})
        except (TypeError, ValueError):
            pass
    dist = agg.get("risk_distribution", {}) or {}
    dist_text = "、".join(f"{k} {int(v)} 人" for k, v in dist.items()) or "暂无"
    facts.append({"id": "F30", "text": f"风险等级分布：{dist_text}"})
    total = sum(int(v) for v in dist.values())
    if total and int(dist.get("高危", 0)) + int(dist.get("关注", 0)) > 0:
        need = int(dist.get("高危", 0)) + int(dist.get("关注", 0))
        facts.append({"id": "F31",
                      "text": f"需重点关注 / 干预的学生合计 {need} 人，"
                              f"占样本 {need / total * 100:.0f}%"})
    # 手册规则命中统计（teacher 端聚合时可选携带，key=规则号，value=人数）
    for k, (rule, cnt) in enumerate((agg.get("rule_hits") or {}).items()):
        facts.append({"id": f"F{40 + k}",
                      "text": f"手册规则 {rule} 命中 {int(cnt)} 人"})
    return facts


def facts_to_text(facts: List[Dict[str, str]]) -> str:
    """事实卡片 -> 喂给模型的编号清单文本。"""
    return "\n".join(f"[{f['id']}] {f['text']}" for f in facts)


CITATION_INSTRUCTION = (
    "引用要求：你的每一条含有具体数值或等级判断的结论，必须在句末标注其支撑事实卡片编号"
    "（如 [F01]、[F12]）；只能引用下列卡片中出现的数值，不得编造卡片外的数据；"
    "建议类内容若依据某条手册提醒，必须标注对应 [F4x] 编号。"
    "同一句话依据多张卡片时，可合并标注在一个方括号内（如 [F01, F12]），"
    "但务必与结论写在同一句的句末。"
)


def validate_cited_answer(text: str, facts: List[Dict[str, str]]) -> Dict[str, Any]:
    """后校验（防线②）：按行 / 句检查证据引用完整性。

    - 「数据断言」（X 分 / X% / 第 X 名 / 等级判断等）无有效 [Fxx] 引用 -> 剔除；
    - 「建议性数量」（如"每周锻炼 3 次""保证 8 小时睡眠"）是行动建议而非
      数据陈述，即使无引用也放行 —— 避免误杀合理建议；
    - 引用了不存在编号的 -> 删除该引用标记；
    - 标题行 / 不含数值的过渡句保留。
    返回 {text, used_ids, dropped}。
    """
    valid_ids = {f["id"] for f in facts}
    used: List[str] = []
    dropped: List[str] = []

    def _strip_invalid(line: str) -> str:
        """去掉引用了不存在编号的引用块；块内仍有有效编号则重建成 [F01, F10]。"""
        def repl(m):
            ids = [i for i in _CITATION_ID_RE.findall(m.group(0)) if i in valid_ids]
            return "[" + ", ".join(ids) + "]" if ids else ""
        return _CITATION_RE.sub(repl, line)

    def _is_suggestion_quantity(sentence: str) -> bool:
        """判断含数字的句子是否属于「建议性数量」而非「数据断言」。

        规则：命中建议语境提示词 且 数字后接行动类单位 且 不含数据断言特征。
        例如「建议每周锻炼 3 次」-> 建议性数量，放行；
        「你的综合评分为 78.5 分」-> 数据断言，必须带引用。
        """
        if _DATA_CLAIM_RE.search(sentence):
            return False
        return bool(_SUGGESTION_HINT_RE.search(sentence)
                    and _SUGGESTION_UNIT_RE.search(sentence))

    kept_lines: List[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            kept_lines.append(raw_line)
            continue
        valid_hits = [i for i in _cited_ids(line) if i in valid_ids]

        # ① 整行已标注有效证据：视为该段共享这些引用，直接放行。
        #    模型习惯「一段话末尾统一标 [F01][F10]」，若仍按分句逐个校验，
        #    引用会被句号切到单独的片段里，导致整段正文被判无据而删空
        #    （教师端研判摘要只剩标题、对话助手回复空白都是这个原因）。
        if valid_hits:
            rebuilt = _strip_invalid(line)
            if rebuilt:
                kept_lines.append(rebuilt)
                used.extend(valid_hits)
            continue

        # ② 整行没有任何有效引用：逐句剔除「含数值的数据断言」，动作建议放行
        parts = _re.split(r"(?<=[。！；;])", line)
        good_parts = []
        for p in parts:
            p_stripped = p.strip()
            if not p_stripped:
                continue
            has_number = bool(_NUMBER_CLAIM_RE.search(_CITATION_RE.sub("", p)))
            if has_number and not _is_suggestion_quantity(p):
                dropped.append(p_stripped)
                continue
            good_parts.append(p)
        rebuilt = "".join(good_parts).strip()
        rebuilt = _strip_invalid(rebuilt)
        if rebuilt:
            kept_lines.append(rebuilt)

    clean_text = "\n".join(l for l in kept_lines if l is not None).strip()
    used = sorted(set(used))
    if not clean_text:
        # 极端情况：模型输出全部无证据 -> 不留空，交由上层规则兜底
        clean_text = ""
    return {"text": clean_text, "used_ids": used, "dropped": dropped}


def append_evidence_footer(validated: Dict[str, Any], facts: List[Dict[str, str]]) -> str:
    """在正文后附「校验徽章 + 引用证据」索引，让每条结论可溯源到事实卡片。"""
    text = validated.get("text", "")
    used = validated.get("used_ids", [])
    dropped = validated.get("dropped", [])
    if not used:
        return text
    fact_map = {f["id"]: f["text"] for f in facts}
    lines = [
        "",
        "---",
        # 校验徽章：证据链可见（引用卡片数 / 校验状态 / 剔除无据表述数）
        f"> ✅ **校验徽章**：引用证据卡片 {len(used)} 条 · 校验通过 · "
        f"无据表述已剔除 {len(dropped)} 条",
        "",
        "**引用证据**",
    ]
    for fid in used:
        if fid in fact_map:
            lines.append(f"- [{fid}] {fact_map[fid]}")
    return text + "\n" + "\n".join(lines)
