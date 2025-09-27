# LLM Automation Agent

Execute plain-English tasks via an LLM-powered agent. No proxy — OpenAI, Gemini, or OpenRouter only.

[![CI](https://github.com/your-username/llm-proj-final/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/llm-proj-final/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

### Built with

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)](https://python.langchain.com/)
[![LiteLLM](https://img.shields.io/badge/LiteLLM-1C3C3C?logo=openai&logoColor=white)](https://docs.litellm.ai/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-1C3C3C)](https://openrouter.ai/)
[![Pydantic AI](https://img.shields.io/badge/Pydantic_AI-E92063)](https://ai.pydantic.dev/)
[![GPTCache](https://img.shields.io/badge/GPTCache-1C3C3C)](https://github.com/zilliztech/gpt-cache)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

---

## Overview

The **LLM Automation Agent** accepts plain-English task descriptions over an API, dispatches them to a tool-calling agent (LangChain or Pydantic AI), and returns results. Tools cover SQL (SQLite/DuckDB), OCR, image extraction, audio transcription, embeddings, web scraping, Markdown/HTML, CSV filtering, and more. All paths are restricted to `/data`; no file deletion. LLM calls use **OpenAI**, **Gemini**, or **OpenRouter** — no proxy.

## Features

- **`POST /run?task=<plain-English>`** — Run the agent on a task. Returns `{ "status": "success", "message": "..." }` or `500` on error.
- **`GET /read?path=<file path>`** — Read a file under `/data` to verify outputs (403 if outside `/data`, 404 if missing).
- **Path guards** — All file I/O restricted to `/data` via `ensure_local_path` / `enforce_data_directory`.
- **Dual backends** — **LangChain** (default) or **Pydantic AI** via `AGENT_BACKEND`.
- **Optional GPTCache** — Cache LLM responses when `GPT_CACHE_ENABLED=true`.
- **Docker** — Dockerfile and Podman-compatible; run with `OPENAI_API_KEY` or `OPENROUTER_API_KEY`.

## Quick start

```bash
git clone https://github.com/your-username/llm-proj-final.git
cd llm-proj-final
pip install uv
uv venv
source venv/bin/activate   # Windows: .\venv\Scripts\activate
uv pip install -r requirements.txt -r requirements-dev.txt
```

Set `OPENAI_API_KEY` or `OPENROUTER_API_KEY` (and `LITELLM_MODEL` if using OpenRouter). Use a `.env` in the project root or in `app/`.

```bash
cd app && uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- **API:** `http://localhost:8000`
- **Docs:** `http://localhost:8000/docs`

## Configuration

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required for default OpenAI model). |
| `OPENAI_API_BASE` | OpenAI API base URL (default: `https://api.openai.com/v1`). |
| `GOOGLE_API_KEY` | Google API key when using Gemini models via LiteLLM. |
| `LITELLM_MODEL` | Model string, e.g. `openai/gpt-4o-mini` (default) or `openrouter/openai/gpt-4o-mini`. |
| `OPENROUTER_API_KEY` | Required when `LITELLM_MODEL` uses `openrouter/...`. |
| `LANGCHAIN_TRACING_V2` | Set to `true` to enable LangSmith tracing. |
| `LANGCHAIN_API_KEY` | LangSmith API key (when tracing enabled). |
| `LANGCHAIN_PROJECT` | LangSmith project name (default: `llm-proj-final`). |
| `GPT_CACHE_ENABLED` | Set to `true` to enable GPTCache for LangChain LLM calls. |
| `AGENT_BACKEND` | `langchain` (default) or `pydantic_ai`. |

## Docker

```bash
docker build -t llm-proj-final .
docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY -p 8000:8000 llm-proj-final
```

With OpenRouter:

```bash
docker run --rm -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY -e LITELLM_MODEL=openrouter/openai/gpt-4o-mini -p 8000:8000 llm-proj-final
```

Podman:

```bash
podman run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY -p 8000:8000 llm-proj-final
```

## API

| Endpoint | Description |
|----------|-------------|
| `POST /run?task=<plain-English>` | Run the agent on a task. Returns `{ "status": "success", "message": "..." }` or `500` on error. |
| `GET /read?path=<file path>` | Read a file under `/data`. Returns `403` if path is outside `/data`, `404` if not found. |

## CI

GitHub Actions runs on push and pull requests: **Ruff** (check + format), **pytest**, and **Docker build**. The agent smoke test uses OpenRouter; add an `OPENROUTER_API_KEY` repository secret (or use `OPENAI_API_KEY` with OpenAI) to run it. If neither is set, the smoke test is skipped.

See [.github/workflows/ci.yml](.github/workflows/ci.yml).

## Project structure

```
llm-proj-final/
├── app/
│   ├── main.py           # FastAPI app; /run, /read
│   ├── config.py         # Env-based config (LiteLLM, LangSmith, GPTCache, AGENT_BACKEND)
│   ├── funtion_tasks.py  # Tool implementations (path guards, no deletion)
│   ├── agent_langchain.py
│   ├── agent_pydantic_ai.py
│   └── task_to_embed.txt
├── tests/
│   └── test_api.py
├── .github/workflows/
│   └── ci.yml
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Commit changes with meaningful messages.
4. Open a pull request.

## License & credits

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

- **Author:** [Sandeep S](https://github.com/sandeepstele)
- **Original development:** [ANdIeCOOl](https://github.com/ANdIeCOOl)
