# 1) Use a lightweight Python image
FROM python:3.11-slim AS base

# 2) Set working directory
WORKDIR /app

# 3) Install system dependencies (if you need them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4) Copy dependency files first (for better caching)
COPY requirements.txt .

# 5) Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6) Copy the rest of your code
COPY app ./app
COPY data ./data
COPY database ./database
COPY debug ./debug
COPY scripts ./scripts
# 7) Expose a port (if you're running an API, e.g. FastAPI)
EXPOSE 8000

# 8) Set environment variables (non-sensitive defaults)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 9) Default command (adjust to your entrypoint)
# Example if you use FastAPI
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
#
# If you don't have FastAPI and just want to run a script, use something like:
CMD ["python", "app/main.py"]
