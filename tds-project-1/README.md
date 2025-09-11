# TDS-Project-1

LLM-based automation agent with **LangChain** (agentic loop), **LangSmith** (tracing), **LiteLLM** (LLM routing), **OpenRouter**, **Pydantic AI** (optional backend), and **GPTCache** (LLM response caching).

## Commands to run

**Terminal 1** — Start the app:

```bash
pip install uv
uv venv
source venv/bin/activate   # Windows: .\venv\Scripts\activate
cd app
uv run main.py
```

You should see the app started successfully. API: `http://localhost:8000`. Docs: `http://localhost:8000/docs`.

**Terminal 2** (optional):

```bash
uv run evaluate.py
```

## Environment variables

No proxy. All LLM calls use **OpenAI** or **Gemini** (or **OpenRouter**).

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

Create a `.env` in the `app/` directory (or project root) and set the variables above as needed.

## Docker

Build and run:

```bash
docker build -t llm-proj-final .
docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY -p 8000:8000 llm-proj-final
```

Or with OpenRouter:

```bash
docker run --rm -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY -e LITELLM_MODEL=openrouter/openai/gpt-4o-mini -p 8000:8000 llm-proj-final
```

Or use Podman:

```bash
podman run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY -p 8000:8000 llm-proj-final
```

## CI

GitHub Actions runs on push and pull requests: **Ruff** (check + format), **pytest**, and **Docker build**. The agent smoke test uses OpenRouter; add an `OPENROUTER_API_KEY` repository secret (or use `OPENAI_API_KEY` with OpenAI) to run it. If neither is set, the smoke test is skipped (e.g. in forks).

## API

- **`POST /run?task=<plain-English>`** — Run the agent on a task. Returns `{ "status": "success", "message": "..." }` or `500` on error.
- **`GET /read?path=<file path>`** — Read a file under `/data` to verify outputs.

## Project structure

```
app/
├── main.py              # FastAPI app; /run dispatches to agent, /read unchanged
├── config.py            # Env-based config (LiteLLM, LangSmith, GPTCache, AGENT_BACKEND)
├── funtion_tasks.py     # Tool implementations (path guards, no deletion)
├── agent_langchain.py   # LangChain agent + LC tools (LiteLLM)
├── agent_pydantic_ai.py # Pydantic AI agent (optional backend)
└── task_to_embed.txt    # Task descriptions
```

## License

MIT.
