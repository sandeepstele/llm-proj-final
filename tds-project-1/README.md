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

| Variable | Purpose |
|----------|---------|
| `AIPROXY_TOKEN` | API key for OpenAI-compatible proxy (required for default model). |
| `OPENAI_BASE_URL` | Proxy base URL (default: `http://aiproxy.sanand.workers.dev/openai/v1`). |
| `LANGCHAIN_TRACING_V2` | Set to `true` to enable LangSmith tracing. |
| `LANGCHAIN_API_KEY` | LangSmith API key (when tracing enabled). |
| `LANGCHAIN_PROJECT` | LangSmith project name (default: `llm-proj-final`). |
| `LITELLM_MODEL` | Model string, e.g. `openai/gpt-4o-mini` (default) or `openrouter/openai/gpt-4o-mini`. |
| `OPENROUTER_API_KEY` | Required when `LITELLM_MODEL` uses `openrouter/...`. |
| `GPT_CACHE_ENABLED` | Set to `true` to enable GPTCache for LangChain LLM calls. |
| `AGENT_BACKEND` | `langchain` (default) or `pydantic_ai`. |

Create a `.env` in the `app/` directory (or project root) and set the variables above as needed.

## Docker

Build and run:

```bash
docker build -t llm-proj-final .
docker run --rm -e AIPROXY_TOKEN=$AIPROXY_TOKEN -p 8000:8000 llm-proj-final
```

Or use Podman:

```bash
podman run --rm -e AIPROXY_TOKEN=$AIPROXY_TOKEN -p 8000:8000 llm-proj-final
```

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
