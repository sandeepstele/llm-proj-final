# LLM-Based-Workflow-automation 

**LLM-Based-Workflow-automation** is a FastAPI-based automation agent that leverages Large Language Models (LLMs) to execute tasks described in plain English. It integrates structured tool functions, such as transcribing audio, processing CSV files, running SQL queries, scraping webpages, and more, based on user-defined tasks.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Endpoints](#endpoints)
- [Extending the Project](#extending-the-project)
- [Dependencies](#dependencies)
- [License](#license)

## Overview

llmauto interprets natural language instructions and dynamically selects the appropriate tool functions to execute. It follows these steps:
1. Converts Python tool functions into OpenAI-compatible schemas.
2. Queries an LLM API to match a user's task with available functions.
3. Executes the corresponding functions and returns results.
4. Supports operations like file processing, transcription, database queries, and web scraping.

This project is containerized using Docker for seamless deployment.

## Features

- **Natural Language Processing:** Users can specify tasks in plain English, and the system maps them to executable functions.
- **Data Processing:** Handles Markdown conversion, CSV filtering, SQL queries, and data extraction.
- **Audio and Image Processing:** Transcribes audio files and extracts text from images.
- **Web Scraping:** Retrieves structured data from web pages.
- **Automation and Integration:** Can be extended to handle CI/CD pipelines and custom workflows.
- **Containerized Deployment:** Easily deployable via Docker.

## Architecture

The system comprises:
- **FastAPI Backend:** Exposes RESTful endpoints for task execution and file management.
- **LLM API Integration:** Communicates with an LLM API to interpret and execute tasks.
- **Tool Functions:** Predefined functions for data processing, OCR, audio transcription, and more.
- **Dockerized Setup:** Ensures compatibility across environments.

## Installation

### Prerequisites
- Docker installed.
- Docker Hub account (for publishing the image).

### Build the Docker Image
```sh
docker build --no-cache -t sandeepstele/llmauto .
```

### Run the Docker Container
```sh
docker run -p 8000:8000 --name fastapi-app sandeepstele/llmauto
```

## Usage

### Example: Download and Transcribe an Audio File

#### Step 1: Download Audio File
```sh
curl -X POST "http://localhost:8000/run" \
-H "Content-Type: application/json" \
--data-urlencode "task=Download the audio file from 'https://raw.githubusercontent.com/sandeepstele/llm-proj-final/main/Harvard%20list%2001.wav' and save it as './data/harvard_audio.wav'."
```

#### Step 2: Transcribe the Audio File
```sh
curl -X POST "http://localhost:8000/run" \
-H "Content-Type: application/json" \
--data-urlencode "task=Transcribe the audio file './data/harvard_audio.wav' and save the result to './data/output/transcription.txt'."
```

#### Step 3: Read Transcription Output
```sh
curl -G "http://localhost:8000/read" --data-urlencode "path=./data/output/transcription.txt"
```

### Example: Download and Filter a CSV File

#### Step 1: Download CSV File
```sh
curl -X POST "http://localhost:8000/run" \
-H "Content-Type: application/json" \
--data-urlencode "task=Download the CSV file from 'https://people.sc.fsu.edu/~jburkardt/data/csv/addresses.csv' and save it as './data/addresses.csv'."
```

#### Step 2: Read CSV Content
```sh
curl -G "http://localhost:8000/read" --data-urlencode "path=./data/addresses.csv"
```

## Endpoints

### `POST /run`
**Description:** Processes a plain-English task and executes the corresponding tool function(s).

**Example Request:**
```sh
curl -X POST "http://localhost:8000/run" \
-H "Content-Type: application/json" \
--data-urlencode "task=Transcribe the audio file './data/harvard_audio.wav' and save the result to './data/output/transcription.txt'."
```

### `GET /read`
**Description:** Reads and returns the content of a specified file.

**Example Request:**
```sh
curl -G "http://localhost:8000/read" --data-urlencode "path=./data/addresses.csv"
```

## Extending the Project

To add a new tool function:
1. Define the Python function.
2. Convert it to an OpenAI-compatible schema.
3. Register it in `function_mappings`.
4. Rebuild the Docker image.

## Dependencies

- **FastAPI** – API framework.
- **Uvicorn** – ASGI server.
- **Requests** – HTTP client.
- **python-dotenv** – Environment variable management.
- **BeautifulSoup4** – Web scraping.
- **Markdown** – Markdown processing.
- **DuckDB & SQLite3** – Database handling.
- **Numpy** – Numerical computing.
- **SpeechRecognition** – Audio transcription.
- **Pytesseract** – OCR processing.
- **FFmpeg-python & Pydub** – Audio processing.

## License

This project is licensed under the [MIT License](LICENSE).

