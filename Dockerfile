# Multi-stage Dockerfile for ML-TA System
# Stage 1: Base image with system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r mlta && useradd -r -g mlta mlta

# Set working directory
WORKDIR /app

# Stage 2: Dependencies
FROM base as dependencies

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 3: Application
FROM dependencies as application

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY tests/ ./tests/
COPY scripts/ ./scripts/

# Copy configuration files
COPY demo.py ./
COPY README.md ./

# Create necessary directories
RUN mkdir -p data/raw data/bronze data/silver data/gold \
    models logs artefacts cache monitoring

# Set ownership
RUN chown -R mlta:mlta /app

# Switch to non-root user
USER mlta

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 9090

# Default command
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
