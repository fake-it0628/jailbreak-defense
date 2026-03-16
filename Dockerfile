# HiSCaM: Hidden State Causal Monitoring for LLM Jailbreak Defense
# Docker configuration for reproducible environments

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/splits \
    hidden_states checkpoints results figures

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command - run tests
CMD ["python", "-m", "pytest", "tests/", "-v"]

# Alternative entrypoints:
# Run demo: docker run -p 7860:7860 hiscam python demo/app.py
# Run training: docker run hiscam python scripts/train_safety_prober.py
# Run evaluation: docker run hiscam python scripts/evaluate_benchmark.py
