# LLM-Based Automation Agent

## Overview
The **LLM-Based Automation Agent** is designed to execute plain-English tasks using a Large Language Model (LLM) and integrate into a Continuous Integration (CI) pipeline. This automation agent processes structured and unstructured tasks while ensuring security and compliance constraints. 

## Features
- Accepts plain-English task descriptions via an API.
- Parses, interprets, and executes tasks using **GPT-4o-Mini**.
- Ensures security constraints (e.g., no external file access, no data deletion).
- Supports multi-step operations and structured automation workflows.
- Provides verifiable results via a dedicated endpoint.
- Fully containerized with **Docker** and **Podman** compatibility.

## API Endpoints
### `POST /run?task=<task description>`
Executes a task described in natural language.
- **Success Response:** `200 OK`
- **Task Error Response:** `400 Bad Request`
- **Agent Error Response:** `500 Internal Server Error`

### `GET /read?path=<file path>`
Retrieves the content of a specified file to verify output correctness.
- **Success Response:** `200 OK` with file content.
- **File Not Found Response:** `404 Not Found`

## Installation & Setup
### Prerequisites
- Python 3.8+
- Docker / Podman
- AI Proxy Token (Environment Variable: `AIPROXY_TOKEN`)

### Clone the Repository
```sh
git clone https://github.com/your-username/llm-based-automation-agent.git
cd llm-based-automation-agent
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

### Run the Application
```sh
export AIPROXY_TOKEN=your_token_here
python app.py
```

### Run with Docker
```sh
docker build -t llm-automation-agent .
docker run --rm -e AIPROXY_TOKEN=$AIPROXY_TOKEN -p 8000:8000 llm-automation-agent
```

## Supported Tasks
- Install and run files to generate required data.
- Format Markdown files using `prettier@3.4.2`.
- Count data from a date file and store results.
- Sort data.
- Extract log files.
- Index Markdown files based on H1 headings.
- Extract the email address from an email file.
- Extract number and text from an image.
- Find the most similar comments using embeddings.
- Query in an SQLite database.
- Prevent access outside `/data`.
- Prevent file deletion.
- Handle data fetching, Git commits, SQL queries, web scraping, image compression, audio transcription, Markdown-to-HTML conversion, and CSV filtering.


## Deployment
### Run with Podman
```sh
podman run --rm -e AIPROXY_TOKEN=$AIPROXY_TOKEN -p 8000:8000 sandeepstele/llmauto
```

### Docker Hub Repository
The Docker image for this project is hosted at:
[Docker Hub - sandeepstele/llmauto](https://hub.docker.com/r/sandeepstele/llmauto/tags)


## Contribution Guidelines
1. Fork the repository.
2. Create a feature branch.
3. Commit changes with meaningful messages.
4. Open a pull request.

## Author
Developed by [Sandeep S](https://github.com/sandeepstele).

## Credits
This project is based on initial development by [ANdIeCOOl](https://github.com/ANdIeCOOl).

## License
This project is licensed under the **MIT License**. See `LICENSE` for details.

## Credits
This project is based on initial development by [ANdIeCOOl](https://github.com/ANdIeCOOl).

.

