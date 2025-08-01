# /// script
# dependencies = [
#   "fastapi",
#   "requests",
#   "python-dotenv",
#   "uvicorn",
#   "beautifulsoup4",
#   "markdown",
#   "requests<3",
#   "duckdb",
#   "numpy",
#   "python-dateutil",
#   "docstring-parser",
#   "httpx",
#   "pydantic",
#   "pytesseract",
#   "pillow",
#   "ffmpeg-python",
#   "pydub",
#   "SpeechRecognition",
#   "langchain",
#   "langchain-core",
#   "langchain-openai",
#   "langchain-community",
#   "langchain-classic",
#   "langsmith",
#   "langchain-litellm",
#   "litellm",
#   "pydantic-ai",
#   "gptcache",
# ]
# ///
import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
import uvicorn

load_dotenv()

# Config and LangSmith / GPTCache setup before any agent use
from config import (
    AGENT_BACKEND,
    GPT_CACHE_ENABLED,
    setup_langsmith_env,
    RUNNING_IN_CODESPACES,
    RUNNING_IN_DOCKER,
)

setup_langsmith_env()

if GPT_CACHE_ENABLED:
    try:
        from gptcache.processor.pre import get_prompt
        from gptcache.manager.factory import get_data_manager
        from langchain_community.cache import GPTCache

        try:
            from langchain_core.globals import set_llm_cache
        except ImportError:
            from langchain.globals import set_llm_cache

        def _init_gptcache(cache_obj, llm_str: str = ""):
            cache_obj.init(
                pre_embedding_func=get_prompt,
                data_manager=get_data_manager(manager="map", data_dir=f"./gptcache_map_{llm_str or 'default'}"),
            )

        set_llm_cache(GPTCache(init_func=_init_gptcache))
        logging.getLogger(__name__).info("GPTCache enabled for LangChain LLM calls.")
    except Exception as e:
        logging.getLogger(__name__).warning("GPTCache init failed: %s. Continuing without cache.", e)

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_local_path(path: str) -> str:
    """Ensure the path uses './data/...' locally, but '/data/...' in Docker."""
    if (not RUNNING_IN_CODESPACES) and RUNNING_IN_DOCKER:
        return path
    return path.lstrip("/")


@app.post("/run")
async def run_task(task: str = Query(..., description="Plain-English task description")):
    logger.info("run_task: %s", task[:200] if len(task) > 200 else task)
    try:
        if AGENT_BACKEND == "pydantic_ai":
            from agent_pydantic_ai import run_agent as run_pydantic_ai_agent
            result = run_pydantic_ai_agent(task)
        else:
            from agent_langchain import run_agent
            result = run_agent(task)

        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "Agent error."))
        return {"status": "success", "message": result.get("message", "Task executed successfully.")}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error executing task")
        raise HTTPException(status_code=500, detail="An internal error occurred while executing the task.")


@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str = Query(..., description="Path to the file to read")):
    logger.info("read_file: %s", path)
    output_file_path = ensure_local_path(path)
    if not os.path.exists(output_file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    with open(output_file_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
