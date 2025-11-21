# Dockerfile for MIDI DNA API
# Modular Semantic Discovery System - Production Deployment
# Agent 10: Documentation & Deployment Manager

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY requirements-ml.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-ml.txt

# Copy application code
COPY midi_generator/ midi_generator/
COPY examples/ examples/
COPY setup.py .
COPY README.md .

# Install package
RUN pip install -e .

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/output

# Download pre-trained models (placeholder - actual download would go here)
# RUN python -m midi_generator.deployment.download_models

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run API server
CMD ["uvicorn", "midi_generator.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
