# Use Debian-based slim Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Install system dependencies required for the application
RUN apt-get update && apt-get install -y --fix-missing --no-install-recommends \
    gcc \
    g++ \
    make \
    libffi-dev \
    musl-dev \
    libopenblas-dev \
    sqlite3 \
    libsqlite3-dev \
    libmagic-dev \
    tesseract-ocr \
    ffmpeg \
    flac \
    curl \
    git \
    nodejs \
    npm \
    libsndfile1 \
    libasound2 \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Prettier globally using npm
RUN npm install -g prettier

# Copy the FastAPI application code into the container
COPY app /app

# Install all required Python dependencies

RUN pip install --no-cache-dir \
    uv \
    pytesseract \
    python-dotenv \
    beautifulsoup4 \
    markdown \
    duckdb \
    numpy \
    python-dateutil \
    docstring-parser \
    httpx \
    pydantic \
    SpeechRecognition \
    fastapi \
    requests \
    pillow \
    uvicorn \
    ffmpeg-python \
    pydub 
    
RUN pip install --no-cache-dir \
    ffmpeg-python 
# Expose the port that the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]