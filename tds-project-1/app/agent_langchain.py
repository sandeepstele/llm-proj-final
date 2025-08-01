"""LangChain tool-calling agent with LiteLLM. Wraps funtion_tasks as LC tools."""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
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

# Lazy imports to avoid loading LangChain/LiteLLM before config
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
    return [
        tool(
            _download_file_tool,
            name="download_file",
            description="Download a file from the given URL and save it to output_path.",
        ),
        tool(
            _install_and_run_script,
            name="install_and_run_script",
            description="Install a package, download a script from a URL, run with uv. Use when task says download or https.",
        ),
        tool(
            format_file_with_prettier,
            name="format_file_with_prettier",
            description="Format a file using Prettier. Create sample markdown if missing.",
        ),
        tool(
            _query_database,
            name="query_database",
            description="Execute a SQL query on SQLite and write result to output file.",
        ),
        tool(
            extract_specific_text_using_llm,
            name="extract_specific_text_using_llm",
            description="Extract specific text from a file using an LLM; write to output file.",
        ),
        tool(
            get_similar_text_using_embeddings,
            name="get_similar_text_using_embeddings",
            description="Find most similar lines in a file using embeddings; write to output file.",
        ),
        tool(
            extract_text_from_image,
            name="extract_text_from_image",
            description="Extract text from an image (OCR + LLM); write to output file.",
        ),
        tool(
            extract_specific_content_and_create_index,
            name="extract_specific_content_and_create_index",
            description="Index files by extension; extract content marker (e.g. H1); write index JSON.",
        ),
        tool(
            process_and_write_logfiles,
            name="process_and_write_logfiles",
            description="Process N most recent log files, write X lines each to output file.",
        ),
        tool(
            sort_json_by_keys,
            name="sort_json_by_keys",
            description="Sort JSON array by given keys; write to output file.",
        ),
        tool(
            count_occurrences,
            name="count_occurrences",
            description="Count date components or regex pattern in file; write count to output file.",
        ),
        tool(
            _fetch_data_from_api_and_save,
            name="fetch_data_from_api_and_save",
            description="Fetch data from API (GET or POST) and save JSON to file.",
        ),
        tool(
            clone_git_repo_and_commit,
            name="clone_git_repo_and_commit",
            description="Clone a git repo, add all, commit with message.",
        ),
        tool(
            scrape_webpage,
            name="scrape_webpage",
            description="Scrape a webpage and save prettified HTML to file.",
        ),
        tool(
            compress_image,
            name="compress_image",
            description="Compress or resize image; save with given quality (1-95).",
        ),
        tool(
            transcribe_audio,
            name="transcribe_audio",
            description="Transcribe audio (MP3/WAV) to text; write to output file.",
        ),
        tool(
            convert_markdown_to_html,
            name="convert_markdown_to_html",
            description="Convert Markdown file to HTML.",
        ),
        tool(
            filter_csv,
            name="filter_csv",
            description="Filter CSV by column=value; write matching rows as JSON to output file.",
        ),
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
                "You are an intelligent agent that understands and parses tasks. "
                "You identify the best tool(s) to use and may chain multiple steps. "
                "All file paths must be under /data. Never delete files.",
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
        out = result.get("output") or "Task completed."
        return {"status": "success", "message": out}
    except Exception as e:
        logger.exception("Agent run failed")
        return {"status": "error", "message": str(e)}
