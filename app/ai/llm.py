"""LLM access through LangChain chat models, with graceful offline fallback.

`get_llm()` returns None when no provider is configured — every AI feature then falls back to the deterministic
heuristic engine, so the whole platform works without an API key (useful for tests, demos and air-gapped setups).
"""
from __future__ import annotations

import logging
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from app.config import Settings, get_settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLM:
    """Thin wrapper over a LangChain BaseChatModel with text + structured-output helpers."""

    def __init__(self, chat_model, name: str):
        self.chat = chat_model
        self.name = name

    def text(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        res = self.chat.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        c = res.content
        return c if isinstance(c, str) else "".join(b.get("text", "") for b in c if isinstance(b, dict))

    def structured(self, schema: Type[T], system: str, user: str) -> T:
        from langchain_core.messages import HumanMessage, SystemMessage
        runnable = self.chat.with_structured_output(schema)
        out = runnable.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return out if isinstance(out, schema) else schema.model_validate(out)


def get_llm(settings: Optional[Settings] = None) -> Optional[LLM]:
    s = settings or get_settings()
    try:
        if s.llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return LLM(ChatAnthropic(model=s.llm_model, temperature=0, max_tokens=2500), f"anthropic:{s.llm_model}")
        if s.llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            return LLM(ChatOpenAI(model=s.llm_model, temperature=0), f"openai:{s.llm_model}")
    except Exception as e:  # missing package / API key
        log.warning("LLM provider '%s' unavailable (%s) — using offline heuristic engine.", s.llm_provider, e)
    return None
