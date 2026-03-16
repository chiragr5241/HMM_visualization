FROM python:3.11-slim

WORKDIR /app

# git is required to pip-install simplexity from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY static/ static/

# HF Spaces expects port 7860
EXPOSE 7860

# Run with a single worker (JAX is already parallelised internally)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
