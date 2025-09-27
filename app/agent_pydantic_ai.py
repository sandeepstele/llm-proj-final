"""Pydantic AI agent (optional backend). Same tools as LangChain, OpenAI-compatible model."""
from __future__ import annotations

import logging
from typing import Any, Optional

from config import (
    LITELLM_MODEL,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENROUTER_API_KEY,
    use_openrouter,
)
from funtion_tasks import (
    format_file_with_prettier,
    query_database,
    run_sql_query_on_database,
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

_agent: Any = None


def _query_db(db_file: str, output_file: str, query: str, query_params: list) -> str:
    query_database(db_file, output_file, query, tuple(query_params))
    return f"Query executed; result written to {output_file}"


def _fetch_api(url: str, output_file: str, generated_prompt: str, params: Optional[dict] = None) -> str:
    fetch_data_from_api_and_save(url, output_file, generated_prompt, params)
    return f"Data saved to {output_file}"


def _install_run(package: str, args: list, script_url: str) -> str:
    install_and_run_script(package, args, script_url=script_url)
    return "Script executed successfully"


def _get_model():
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    if use_openrouter():
        base = "https://openrouter.ai/api/v1"
        key = OPENROUTER_API_KEY or ""
    else:
        base = OPENAI_API_BASE or "https://api.openai.com/v1"
        key = OPENAI_API_KEY or ""

    client = AsyncOpenAI(api_key=key, base_url=base)
    model_name = "gpt-4o-mini"
    if use_openrouter() and LITELLM_MODEL.startswith("openrouter/"):
        model_name = LITELLM_MODEL.split("/", 1)[-1] if "/" in LITELLM_MODEL else "openai/gpt-4o-mini"
    return OpenAIChatModel(model_name, provider=OpenAIProvider(openai_client=client))


def _get_agent():
    global _agent
    if _agent is not None:
        return _agent

    from pydantic_ai import Agent

    model = _get_model()
    _agent = Agent(
        model,
        output_type=str,
        system_prompt="Use the available tools to complete the user's task. All paths must be under /data. Do not delete files.",
    )

    # tool_plain: no deps
    @_agent.tool_plain
    def download_file_tool(url: str, output_path: str) -> str:
        """Download a file from the given URL and save it to output_path."""
        return download_file(url, output_path)

    @_agent.tool_plain
    def install_and_run_script_tool(package: str, args: list, script_url: str) -> str:
        """Install package, download script from URL, run with uv. Use when task says download or https."""
        return _install_run(package, args, script_url)

    @_agent.tool_plain
    def format_file_with_prettier_tool(file_path: str, prettier_version: str) -> str:
        """Format a file using Prettier. Create sample markdown if missing."""
        format_file_with_prettier(file_path, prettier_version)
        return "Formatted successfully."

    @_agent.tool_plain
    def query_database_tool(db_file: str, output_file: str, query: str, query_params: list) -> str:
        """Execute SQL on SQLite and write result to file. query_params: list of values in order."""
        return _query_db(db_file, output_file, query, query_params)

    @_agent.tool_plain
    def run_sql_query_on_database_tool(
        database_file: str, query: str, output_file: str, is_sqlite: bool = True
    ) -> str:
        """Run SQL on SQLite or DuckDB; write all result rows to output file. is_sqlite selects backend."""
        run_sql_query_on_database(database_file, query, output_file, is_sqlite)
        return f"SQL executed; rows written to {output_file}"

    @_agent.tool_plain
    def extract_specific_text_using_llm_tool(
        input_file: str,
        output_file: str,
        task: str,
        max_chars: Optional[int] = None,
    ) -> str:
        """Extract specific text from a file using an LLM; write to output file. max_chars: truncate input if set."""
        extract_specific_text_using_llm(input_file, output_file, task, max_chars=max_chars)
        return f"Extracted; written to {output_file}"

    @_agent.tool_plain
    def get_similar_text_using_embeddings_tool(
        input_file: str, output_file: str, no_of_similar_texts: int
    ) -> str:
        """Find most similar lines in a file using embeddings; write to output file."""
        get_similar_text_using_embeddings(input_file, output_file, no_of_similar_texts)
        return f"Similar texts written to {output_file}"

    @_agent.tool_plain
    def extract_text_from_image_tool(
        image_path: str,
        output_file: str,
        task: str,
        strip_spaces: bool = False,
        ocr_only: bool = False,
    ) -> str:
        """Extract text from an image (OCR + LLM); write to output file. strip_spaces: remove spaces (A8-style). ocr_only: skip LLM, write OCR only."""
        extract_text_from_image(image_path, output_file, task, strip_spaces=strip_spaces, ocr_only=ocr_only)
        return f"Extracted; written to {output_file}"

    @_agent.tool_plain
    def extract_specific_content_and_create_index_tool(
        input_file: str, output_file: str, extension: str, content_marker: str
    ) -> str:
        """Index files by extension; extract content marker (e.g. H1); write index JSON."""
        extract_specific_content_and_create_index(input_file, output_file, extension, content_marker)
        return f"Index written to {output_file}"

    @_agent.tool_plain
    def process_and_write_logfiles_tool(
        input_file: str, output_file: str, num_logs: int = 10, num_of_lines: int = 1
    ) -> str:
        """Process N most recent log files, write X lines each to output file."""
        process_and_write_logfiles(input_file, output_file, num_logs, num_of_lines)
        return f"Log lines written to {output_file}"

    @_agent.tool_plain
    def sort_json_by_keys_tool(input_file: str, output_file: str, keys: list) -> str:
        """Sort JSON array by given keys; write to output file."""
        sort_json_by_keys(input_file, output_file, keys)
        return f"Sorted JSON written to {output_file}"

    @_agent.tool_plain
    def count_occurrences_tool(
        input_file: str,
        output_file: str,
        date_component: Optional[str] = None,
        target_value: Optional[int] = None,
        custom_pattern: Optional[str] = None,
    ) -> str:
        """Count date components or regex pattern in file; write count to output file."""
        count_occurrences(input_file, output_file, date_component, target_value, custom_pattern)
        return f"Count written to {output_file}"

    @_agent.tool_plain
    def fetch_data_from_api_and_save_tool(
        url: str, output_file: str, generated_prompt: str, params: Optional[dict] = None
    ) -> str:
        """Fetch data from API (GET or POST) and save JSON to file."""
        return _fetch_api(url, output_file, generated_prompt, params)

    @_agent.tool_plain
    def clone_git_repo_and_commit_tool(repo_url: str, output_dir: str, commit_message: str) -> str:
        """Clone a git repo, add all, commit with message."""
        clone_git_repo_and_commit(repo_url, output_dir, commit_message)
        return "Repo cloned and committed."

    @_agent.tool_plain
    def scrape_webpage_tool(url: str, output_file: str) -> str:
        """Scrape a webpage and save prettified HTML to file."""
        scrape_webpage(url, output_file)
        return f"Scraped HTML written to {output_file}"

    @_agent.tool_plain
    def compress_image_tool(input_file: str, output_file: str, quality: int = 50) -> str:
        """Compress or resize image; save with given quality (1-95)."""
        compress_image(input_file, output_file, quality)
        return f"Compressed image written to {output_file}"

    @_agent.tool_plain
    def transcribe_audio_tool(
        input_file: str, output_file: str, language: str = "en-US"
    ) -> str:
        """Transcribe audio (MP3/WAV) to text; write to output file. language: e.g. en-US."""
        transcribe_audio(input_file, output_file, language=language)
        return f"Transcript written to {output_file}"

    @_agent.tool_plain
    def convert_markdown_to_html_tool(input_file: str, output_file: str) -> str:
        """Convert Markdown file to HTML."""
        convert_markdown_to_html(input_file, output_file)
        return f"HTML written to {output_file}"

    @_agent.tool_plain
    def filter_csv_tool(input_file: str, column: str, value: str, output_file: str) -> str:
        """Filter CSV by column=value; write matching rows as JSON to output file."""
        filter_csv(input_file, column, value, output_file)
        return f"Filtered JSON written to {output_file}"

    return _agent


def run_agent(task: str) -> dict[str, Any]:
    """Run the Pydantic AI agent on a plain-English task. Returns {status, message}."""
    agent = _get_agent()
    try:
        result = agent.run_sync(task)
        out = result.output or "Done."
        return {"status": "success", "message": out}
    except Exception as e:
        logger.exception("Pydantic AI agent run failed")
        return {"status": "error", "message": str(e)}
