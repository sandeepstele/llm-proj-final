"""Environment-based configuration for LLM agent, LiteLLM, LangSmith, GPTCache, etc."""
from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

# Existing
AIPROXY_TOKEN: str = os.getenv("AIPROXY_TOKEN", "")
OPENAI_BASE_URL: str = os.getenv(
    "OPENAI_BASE_URL",
    "http://aiproxy.sanand.workers.dev/openai/v1",
)

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


def litellm_api_key() -> str:
    """API key for LiteLLM: OpenRouter key when using openrouter, else AIPROXY_TOKEN."""
    if use_openrouter():
        return OPENROUTER_API_KEY or ""
    return AIPROXY_TOKEN or ""


def litellm_api_base() -> str | None:
    """API base for LiteLLM: only set when using current proxy (non-OpenRouter)."""
    if use_openrouter():
        return None
    return OPENAI_BASE_URL or None


def setup_litellm_env() -> None:
    """Set env vars so LiteLLM picks up api_key/api_base. Call before creating ChatLiteLLM."""
    if use_openrouter():
        if OPENROUTER_API_KEY and "OPENROUTER_API_KEY" not in os.environ:
            os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
    else:
        if AIPROXY_TOKEN and "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = AIPROXY_TOKEN
        if OPENAI_BASE_URL and "OPENAI_API_BASE" not in os.environ:
            os.environ["OPENAI_API_BASE"] = OPENAI_BASE_URL


def setup_langsmith_env() -> None:
    """Set LangSmith env vars so LangChain traces there when enabled."""
    if LANGCHAIN_TRACING_V2.lower() in ("true", "1", "yes") and LANGCHAIN_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
        if LANGCHAIN_PROJECT:
            os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
