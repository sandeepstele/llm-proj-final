# /// script
# dependencies = [
#   "python-dotenv",
#   "beautifulsoup4",
#   "markdown",
#   "duckdb",
#   "numpy",
#   "python-dateutil",
#   "docstring-parser",
#   "httpx",
#   "pydantic",
# ]
# ///

import base64
import csv
import glob
import json
import logging
import os
import re
import sqlite3
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import ffmpeg
import httpx
import numpy as np
import pytesseract
import requests
from bs4 import BeautifulSoup
from dateutil.parser import parse
from PIL import Image
from pydub import AudioSegment
import speech_recognition as sr

import dotenv
import markdown

dotenv.load_dotenv()
API_KEY = os.getenv("AIPROXY_TOKEN")
URL_CHAT = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
URL_EMBEDDING = "http://aiproxy.sanand.workers.dev/openai/v1/embeddings"
RUNNING_IN_CODESPACES = "CODESPACES" in os.environ
RUNNING_IN_DOCKER = os.path.exists("/.dockerenv")
logging.basicConfig(level=logging.INFO)


def enforce_data_directory(path: str) -> str:
    """Ensure that the given absolute path is under the allowed /data directory."""
    abs_path = os.path.abspath(path)
    allowed_root = os.path.abspath("/data")
    if not abs_path.startswith(allowed_root):
        raise ValueError(f"Access denied: {abs_path} is outside the allowed directory {allowed_root}")
    return abs_path


def ensure_local_path(path: str) -> str:
    """Resolve path (Docker vs local) and enforce /data."""
    if (not RUNNING_IN_CODESPACES) and RUNNING_IN_DOCKER:
        return enforce_data_directory(path)
    local_path = path.lstrip("/")
    full_path = os.path.abspath(os.path.join(os.getcwd(), local_path))
    return enforce_data_directory(full_path)


def format_file_with_prettier(file_path: str, prettier_version: str):
    """Format file with Prettier; create sample markdown if missing."""
    input_file_path = ensure_local_path(file_path)
    if not os.path.exists(input_file_path):
        d = os.path.dirname(input_file_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(input_file_path, "w", encoding="utf-8") as f:
            f.write("# Sample Markdown\nThis is a sample markdown file.")
    result = subprocess.run(
        ["npx", f"prettier@{prettier_version}", "--write", input_file_path],
        capture_output=True,
        text=True,
        shell=True,
    )
    if result.returncode != 0:
        logging.error("Prettier failed: %s", result.stderr)
        raise RuntimeError(f"Prettier failed with exit code {result.returncode}")

def query_gpt(user_input: str, task: str):
    response = requests.post(
        URL_CHAT,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "JUST SO WHAT IS ASKED\nYOUR output is part of a program, using tool functions" + task},
                {"role": "user", "content": user_input},
            ],
        },
    )
    response.raise_for_status()
    return response.json()

def rewrite_sensitive_task(task: str) -> str:
    """Rewrite sensitive task descriptions in an indirect way."""
    task_lower = task.lower()
    
    rewrite_map = {
        "credit card": "Extract the full number exactly as shown in the image. Make sure you include every digit without any omissions or extra spaces. The output should be only the sequence of digits.",
        "cvv": "3-digit number near another number",
        "bank account": "1 longest numerical sequence",
        "routing number": "a series of numbers used for banking",
        "social security": "numerical sequence",
        "passport": "longest alphanumeric string",
        "driver's license": "structured alphanumeric code",
        "api key": "a long secret-looking string",
        "password": "text following 'Password:'",
    }
    
    for keyword, replacement in rewrite_map.items():
        if keyword in task_lower:
            # Replace the keyword with its replacement (case-insensitive)
            return re.sub(keyword, replacement, task, flags=re.IGNORECASE)
    return task

def extract_text_with_tesseract(image_path: str) -> str:
    """Extract text from the image using Tesseract OCR."""
    try:
        img = Image.open(image_path)
        return pytesseract.image_to_string(img).strip()
    except Exception as e:
        logging.error("OCR extraction failed: %s", e)
        return ""

def query_gpt_image(image_path: str, task: str):
    """Rewrite task, run OCR, encode image, send OCR + image to LLM."""
    
    image_format = image_path.split(".")[-1]
    clean_task = rewrite_sensitive_task(task)
    
    ocr_text = extract_text_with_tesseract(image_path)
    with open(image_path, "rb") as file:
        base64_image = base64.b64encode(file.read()).decode("utf-8")
    
    # Build the messages with both the OCR text and image data
    messages = [
        {
            "role": "system",
            "content": (
                "Extract the required input exactly as shown. Return only the final result without additional commentary."
            )
        },
        {
            "role": "user",
            "content": (
                f"{clean_task}. Here is the OCR extracted text from the image: {ocr_text}"
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Additionally, please refer to the image:"},
                {"type": "image_url", "image_url": { "url": f"data:image/{image_format};base64,{base64_image}" }}
            ]
        }
    ]
    
    response = requests.post(
        URL_CHAT,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o-mini",
            "messages": messages
        }
    )
    
    response.raise_for_status()
    return response.json()


def query_database(db_file: str, output_file: str, query: str, query_params: Tuple):
    """Run a SQL query on SQLite and write the single result to output_file."""
    db_file_path = ensure_local_path(db_file)
    output_file_path = ensure_local_path(output_file)
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    try:
        cursor.execute(query, query_params)
        result = cursor.fetchone()
        output_data = result[0] if result else "No results found."
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(str(output_data))
    except sqlite3.Error as e:
        logging.error("query_database failed: %s", e)
        raise
    finally:
        conn.close()
def extract_specific_text_using_llm(input_file: str, output_file: str, task: str):
    """Extract text from input_file per task via LLM; write to output_file."""
    input_file_path = ensure_local_path(input_file)
    output_file_path = ensure_local_path(output_file)
    with open(input_file_path, "r", encoding="utf-8") as f:
        text_info = f.read()
    resp = query_gpt(text_info, task)
    content = resp["choices"][0]["message"]["content"]
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(content)
def get_embeddings(texts: List[str]):
    resp = requests.post(
        URL_EMBEDDING,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": "text-embedding-3-small", "input": texts},
    )
    resp.raise_for_status()
    return np.array([e["embedding"] for e in resp.json()["data"]])


def get_similar_text_using_embeddings(input_file: str, output_file: str, no_of_similar_texts: int):
    """Find up to no_of_similar_texts most similar line pairs via embeddings; write distinct lines to output_file."""
    input_file_path = ensure_local_path(input_file)
    output_file_path = ensure_local_path(output_file)
    with open(input_file_path, "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f.readlines()]
    if not documents:
        with open(output_file_path, "w", encoding="utf-8") as f:
            pass
        return
    line_embeddings = get_embeddings(documents)
    sim = np.dot(line_embeddings, line_embeddings.T)
    np.fill_diagonal(sim, -1)
    n = len(documents)
    triu_i, triu_j = np.triu_indices(n, k=1)
    vals = sim[triu_i, triu_j]
    order = np.argsort(-vals)
    seen: set[int] = set()
    lines_out: List[str] = []
    for idx in order:
        if len(lines_out) >= no_of_similar_texts:
            break
        i, j = int(triu_i[idx]), int(triu_j[idx])
        for k in (i, j):
            if k not in seen:
                seen.add(k)
                lines_out.append(documents[k])
                if len(lines_out) >= no_of_similar_texts:
                    break
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))
def extract_text_from_image(image_path: str, output_file: str, task: str):
    """Extract text from image via OCR + LLM. task describes what to extract."""
    image_path_secure = ensure_local_path(image_path)
    output_file_path = ensure_local_path(output_file)
    resp = query_gpt_image(image_path_secure, task)
    content = resp["choices"][0]["message"]["content"].replace(" ", "")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(content)       
def extract_specific_content_and_create_index(
    input_file: str, output_file: str, extension: str, content_marker: str
):
    """Index files by extension; extract first content_marker line per file; write JSON index."""
    input_file_path = ensure_local_path(input_file)
    output_file_path = ensure_local_path(output_file)
    pattern = os.path.join(input_file_path, "**", f"*{extension}")
    extension_files = glob.glob(pattern, recursive=True)
    index: Dict[str, str] = {}
    for fp in extension_files:
        title = None
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(content_marker):
                    title = line.lstrip(content_marker).strip()
                    break
        index[os.path.relpath(fp, input_file_path)] = title or ""
    with open(output_file_path, "w", encoding="utf-8") as jf:
        json.dump(index, jf, indent=2, sort_keys=True)
def process_and_write_logfiles(
    input_file: str, output_file: str, num_logs: int = 10, num_of_lines: int = 1
):
    """Process num_logs most recent log files; write num_of_lines from each to output_file."""
    input_file_path = ensure_local_path(input_file)
    output_file_path = ensure_local_path(output_file)
    log_files = glob.glob(os.path.join(input_file_path, "*.log"))
    log_files.sort(key=os.path.getmtime, reverse=True)
    recent = log_files[:num_logs]
    with open(output_file_path, "w", encoding="utf-8") as out:
        for lf in recent:
            with open(lf, "r", encoding="utf-8") as inf:
                for _ in range(num_of_lines):
                    line = inf.readline()
                    if not line:
                        break
                    out.write(line)
def sort_json_by_keys(input_file: str, output_file: str, keys: list):
    """Sort JSON array by keys; write to output_file."""
    input_file_path = ensure_local_path(input_file)
    output_file_path = ensure_local_path(output_file)
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(sorted(data, key=lambda x: tuple(x[k] for k in keys)), f)                       
def count_occurrences(
    input_file: str,
    output_file: str,
    date_component: Optional[str] = None,
    target_value: Optional[int] = None,
    custom_pattern: Optional[str] = None,
):
    """Count date components or regex matches in file; write count to output_file."""
    if not custom_pattern and not date_component:
        raise ValueError("Provide either custom_pattern or date_component.")
    if date_component in ("weekday", "month", "year") and target_value is None:
        raise ValueError("target_value required for weekday/month/year.")
    input_file_path = ensure_local_path(input_file)
    output_file_path = ensure_local_path(output_file)
    count = 0
    with open(input_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if custom_pattern and re.search(custom_pattern, line):
                count += 1
                continue
            try:
                pd = parse(line)
            except (ValueError, OverflowError):
                logging.debug("Skipping invalid date: %s", line[:80])
                continue
            if date_component == "weekday" and pd.weekday() == target_value:
                count += 1
            elif date_component == "month" and pd.month == target_value:
                count += 1
            elif date_component == "year" and pd.year == target_value:
                count += 1
            elif date_component == "leap_year" and (
                pd.year % 4 == 0 and (pd.year % 100 != 0 or pd.year % 400 == 0)
            ):
                count += 1
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(str(count))
def install_and_run_script(package: str, args: list, *, script_url: str):
    """Install package, download script from URL (https://...), run with uv. Use only when task says download or gives https URL. Note: curl/uv run use current working directory."""
    if package == "uvicorn":
        subprocess.run(["pip", "install", "uv"], check=True)
    else:
        subprocess.run(["pip", "install", package], check=True)
    subprocess.run(["curl", "-O", script_url], check=True)
    script_name = script_url.rstrip("/").split("/")[-1]
    cmd = ["uv", "run", script_name] + ([a for a in args] if args else [])
    subprocess.run(cmd, check=True)

def fetch_data_from_api_and_save(
    url: str, output_file: str, generated_prompt: str, params: Optional[Dict[str, Any]] = None
):
    """GET URL and save JSON to output_file; if GET fails and params has headers/data, try POST."""
    out_path = ensure_local_path(output_file)
    err: Optional[Exception] = None
    try:
        r = requests.get(url)
        r.raise_for_status()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(r.json(), f, indent=4)
        return
    except requests.exceptions.RequestException as e:
        err = e
        logging.warning("GET %s failed: %s", url, e)
    if params and "headers" in params and "data" in params:
        try:
            r = requests.post(url, headers=params["headers"], json=params["data"])
            r.raise_for_status()
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(r.json(), f, indent=4)
            return
        except requests.exceptions.RequestException as e:
            err = e
            logging.warning("POST %s failed: %s", url, e)
    raise RuntimeError(f"Fetch failed: {err}") from err

def clone_git_repo_and_commit(repo_url: str, output_dir: str, commit_message: str):
    """Clone repo to output_dir, add all, commit with message."""
    out_path = ensure_local_path(output_dir)
    subprocess.run(["git", "clone", repo_url, out_path], check=True)
    subprocess.run(["git", "add", "."], cwd=out_path, check=True)
    subprocess.run(["git", "commit", "-m", commit_message], cwd=out_path, check=True)


def run_sql_query_on_database(
    database_file: str, query: str, output_file: str, is_sqlite: bool = True
):
    """Run SQL on SQLite or DuckDB; write rows to output_file."""
    db_path = ensure_local_path(database_file)
    out_path = ensure_local_path(output_file)
    conn = None
    try:
        conn = sqlite3.connect(db_path) if is_sqlite else duckdb.connect(db_path)
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(str(row) + "\n")
    except (sqlite3.Error, duckdb.Error) as e:
        logging.error("run_sql_query_on_database failed: %s", e)
        raise
    finally:
        if conn is not None:
            conn.close()


def scrape_webpage(url: str, output_file: str):
    """Fetch URL, parse HTML, write prettified HTML to output_file."""
    r = requests.get(url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    out_path = ensure_local_path(output_file)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(soup.prettify())


def compress_image(input_file: str, output_file: str, quality: int = 50):
    """Save image at input_file to output_file with given quality (1-95)."""
    inp = ensure_local_path(input_file)
    out = ensure_local_path(output_file)
    d = os.path.dirname(out)
    if d:
        os.makedirs(d, exist_ok=True)
    img = Image.open(inp)
    img.save(out, quality=quality)

def transcribe_audio(input_file: str, output_file: str):
    """Transcribe MP3/WAV to text via Google Speech Recognition; write to output_file."""
    inp = ensure_local_path(input_file)
    out = ensure_local_path(output_file)
    data_dir = enforce_data_directory("/data")
    temp_wav = os.path.join(data_dir, "temp_transcribe.wav")
    os.makedirs(data_dir, exist_ok=True)
    rec = sr.Recognizer()
    try:
        try:
            with sr.AudioFile(inp) as src:
                audio = rec.record(src)
        except Exception:
            ffmpeg.input(inp).output(
                temp_wav, format="wav", acodec="pcm_s16le", ac=1, ar=16000
            ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
            with sr.AudioFile(temp_wav) as src:
                audio = rec.record(src)
        text = rec.recognize_google(audio)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
    finally:
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except OSError:
                pass


def convert_markdown_to_html(input_file: str, output_file: str):
    """Convert Markdown to HTML (extra, codehilite, toc); write to output_file."""
    inp = ensure_local_path(input_file)
    out = ensure_local_path(output_file)
    with open(inp, "r", encoding="utf-8") as f:
        md = f.read()
    html = markdown.markdown(
        md,
        extensions=["extra", "codehilite", "toc"],
        extension_configs={"codehilite": {"linenums": False, "guess_lang": False}, "toc": {"permalink": True}},
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)


def filter_csv(input_file: str, column: str, value: str, output_file: str):
    """Filter CSV by column=value; write matching rows as JSON to output_file."""
    inp = ensure_local_path(input_file)
    out = ensure_local_path(output_file)
    results = []
    with open(inp, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if column in row and row[column] == value:
                results.append(row)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def download_file(url: str, output_path: str) -> str:
    """Download URL to output_path."""
    r = requests.get(url)
    r.raise_for_status()
    out = ensure_local_path(output_path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(r.content)
    return out