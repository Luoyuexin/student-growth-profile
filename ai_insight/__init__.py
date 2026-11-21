"""ai_insight AI 分析模块（项目创新点之一）

提供基于云端大模型的学情分析与可视化解读能力
详见 llm_client.py（客户端与脱敏网关）与 services.py（各功能实现）
"""

from ai_insight import llm_client, services

__all__ = ["llm_client", "services"]
