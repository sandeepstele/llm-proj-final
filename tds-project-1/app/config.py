"""Environment-based configuration for LLM agent, LiteLLM, LangSmith, GPTCache, etc."""
from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

# OpenAI (no proxy)
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE: str = os.getenv(
    "OPENAI_API_BASE",
    "https://api.openai.com/v1",
)

# Gemini (via LiteLLM)
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# LangSmith
LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "llm-proj-final")

# LiteLLM / OpenRouter
LITELLM_MODEL: str = os.getenv("LITELLM_MODEL", "openai/gpt-4o-mini")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

# GPTCache
GPT_CACHE_ENABLED: bool = os.getenv("GPT_CACHE_ENABLED", "false").lower() in ("true", "1", "yes")

# Agent backend: langchain | pydantic_ai
_backend = os.getenv("AGENT_BACKEND", "langchain").strip().lower()
AGENT_BACKEND: Literal["langchain", "pydantic_ai"] = (
    "pydantic_ai" if _backend == "pydantic_ai" else "langchain"
)

# Runtime
RUNNING_IN_CODESPACES: bool = "CODESPACES" in os.environ
RUNNING_IN_DOCKER: bool = os.path.exists("/.dockerenv")


def use_openrouter() -> bool:
    """True if LITELLM_MODEL uses openrouter/..."""
    return LITELLM_MODEL.strip().lower().startswith("openrouter/")


def chat_base_url() -> str:
    """Base URL for chat/embeddings. OpenRouter or OpenAI. No proxy."""
    if use_openrouter():
        return "https://openrouter.ai/api/v1"
    return (OPENAI_API_BASE or "https://api.openai.com/v1").rstrip("/")


def chat_api_key() -> str:
    """API key for chat/embeddings. OpenRouter or OpenAI."""
    if use_openrouter():
        return OPENROUTER_API_KEY or ""
    return OPENAI_API_KEY or ""


def chat_model() -> str:
    """Model for chat/embeddings. OpenRouter: LITELLM_MODEL without prefix; else OPENAI_CHAT_MODEL or gpt-4o-mini."""
    if use_openrouter():
        m = LITELLM_MODEL.strip().lower().replace("openrouter/", "", 1)
        return m or "openai/gpt-4o-mini"
    return os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def litellm_api_key() -> str:
    """API key for LiteLLM. OpenRouter or OpenAI."""
    return chat_api_key()


def litellm_api_base() -> str | None:
    """API base for LiteLLM. Only set when using OpenAI (non-OpenRouter)."""
    if use_openrouter():
        return None
    return OPENAI_API_BASE or None


def setup_litellm_env() -> None:
    """Set LiteLLM env (api_key, api_base). Call before creating ChatLiteLLM. No proxy."""
    if use_openrouter():
        if OPENROUTER_API_KEY and "OPENROUTER_API_KEY" not in os.environ:
            os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
    else:
        if OPENAI_API_KEY and "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        base = OPENAI_API_BASE or "https://api.openai.com/v1"
        if "OPENAI_API_BASE" not in os.environ:
            os.environ["OPENAI_API_BASE"] = base


def setup_langsmith_env() -> None:
    """Set LangSmith env vars so LangChain traces there when enabled."""
    if LANGCHAIN_TRACING_V2.lower() in ("true", "1", "yes") and LANGCHAIN_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
        if LANGCHAIN_PROJECT:
            os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
