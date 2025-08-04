"""LangChain tool-calling agent with LiteLLM. Wraps funtion_tasks as LC tools."""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_classic.agents.agent import AgentExecutor
from langchain_classic.agents.tool_calling_agent.base import create_tool_calling_agent

from config import (
    setup_litellm_env,
    LITELLM_MODEL,
    litellm_api_key,
    litellm_api_base,
)
from funtion_tasks import (
    format_file_with_prettier,
    query_database,
    extract_specific_text_using_llm,
    get_similar_text_using_embeddings,
    extract_text_from_image,
    extract_specific_content_and_create_index,
    process_and_write_logfiles,
    sort_json_by_keys,
    count_occurrences,
    install_and_run_script,
    fetch_data_from_api_and_save,
    clone_git_repo_and_commit,
    scrape_webpage,
    compress_image,
    transcribe_audio,
    convert_markdown_to_html,
    filter_csv,
    download_file,
)

logger = logging.getLogger(__name__)

_agent_executor: Optional[AgentExecutor] = None


def _query_database(
    db_file: str,
    output_file: str,
    query: str,
    query_params: list,
) -> str:
    """Execute SQL on SQLite and write result to file. query_params: list of values in order."""
    query_database(db_file, output_file, query, tuple(query_params))
    return f"Query executed; result written to {output_file}"


def _fetch_data_from_api_and_save(
    url: str,
    output_file: str,
    generated_prompt: str,
    params: Optional[dict] = None,
) -> str:
    """Fetch API (GET or POST) and save JSON to file."""
    fetch_data_from_api_and_save(url, output_file, generated_prompt, params)
    return f"Data saved to {output_file}"


def _install_and_run_script(
    package: str,
    args: list,
    script_url: str,
) -> str:
    """Install package, download script from URL, run with uv."""
    install_and_run_script(package, args, script_url=script_url)
    return "Script executed successfully"


def _download_file_tool(url: str, output_path: str) -> str:
    """Download a file from the given URL and save it to output_path."""
    return download_file(url, output_path)


def _build_tools() -> list:
    def _tool(name: str, desc: str, func: Any) -> Any:
        return StructuredTool.from_function(name=name, description=desc, func=func)

    return [
        _tool("download_file", "Download a file from the given URL and save it to output_path.", _download_file_tool),
        _tool("install_and_run_script", "Install package, download script from URL, run with uv. Use when task says download or https.", _install_and_run_script),
        _tool("format_file_with_prettier", "Format a file using Prettier. Create sample markdown if missing.", format_file_with_prettier),
        _tool("query_database", "Execute a SQL query on SQLite and write result to output file.", _query_database),
        _tool("extract_specific_text_using_llm", "Extract specific text from a file using an LLM; write to output file.", extract_specific_text_using_llm),
        _tool("get_similar_text_using_embeddings", "Find most similar lines in a file using embeddings; write to output file.", get_similar_text_using_embeddings),
        _tool("extract_text_from_image", "Extract text from an image (OCR + LLM); write to output file.", extract_text_from_image),
        _tool("extract_specific_content_and_create_index", "Index files by extension; extract content marker (e.g. H1); write index JSON.", extract_specific_content_and_create_index),
        _tool("process_and_write_logfiles", "Process N most recent log files, write X lines each to output file.", process_and_write_logfiles),
        _tool("sort_json_by_keys", "Sort JSON array by given keys; write to output file.", sort_json_by_keys),
        _tool("count_occurrences", "Count date components or regex pattern in file; write count to output file.", count_occurrences),
        _tool("fetch_data_from_api_and_save", "Fetch data from API (GET or POST) and save JSON to file.", _fetch_data_from_api_and_save),
        _tool("clone_git_repo_and_commit", "Clone a git repo, add all, commit with message.", clone_git_repo_and_commit),
        _tool("scrape_webpage", "Scrape a webpage and save prettified HTML to file.", scrape_webpage),
        _tool("compress_image", "Compress or resize image; save with given quality (1-95).", compress_image),
        _tool("transcribe_audio", "Transcribe audio (MP3/WAV) to text; write to output file.", transcribe_audio),
        _tool("convert_markdown_to_html", "Convert Markdown file to HTML.", convert_markdown_to_html),
        _tool("filter_csv", "Filter CSV by column=value; write matching rows as JSON to output file.", filter_csv),
    ]


def _get_llm():
    from langchain_litellm import ChatLiteLLM

    setup_litellm_env()
    return ChatLiteLLM(
        model=LITELLM_MODEL,
        api_key=litellm_api_key() or None,
        api_base=litellm_api_base(),
    )


def _get_agent_executor() -> AgentExecutor:
    global _agent_executor
    if _agent_executor is not None:
        return _agent_executor

    llm = _get_llm()
    tools = _build_tools()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Use the available tools to complete the user's task. All paths must be under /data. Do not delete files.",
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    _agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        return_intermediate_steps=False,
        handle_parsing_errors=True,
    )
    return _agent_executor


def run_agent(task: str) -> dict[str, Any]:
    """Run the LangChain agent on a plain-English task. Returns {status, message}."""
    exe = _get_agent_executor()
    try:
        result = exe.invoke({"input": task})
        out = result.get("output") or "Done."
        return {"status": "success", "message": out}
    except Exception as e:
        logger.exception("Agent run failed")
        return {"status": "error", "message": str(e)}
