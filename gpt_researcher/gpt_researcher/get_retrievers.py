"""Retriever selection for GPT Researcher.

重写: 原来的实现引用了不存在的顶层 `retrievers` 模块。
现委托给 gpt_researcher/actions/retriever.py(逻辑与原版一致):
headers["retrievers"] / headers["retriever"] → cfg.retrievers → cfg.retriever → 默认 tavily。
"""

from gpt_researcher.actions.retriever import get_retrievers

__all__ = ["get_retrievers"]
