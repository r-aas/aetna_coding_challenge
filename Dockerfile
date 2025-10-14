# Movie Recommendation System - Production Dockerfile
# Multi-stage build for optimal image size and security

# Build stage
FROM python:3.13-slim as builder

# Set build arguments
ARG BUILDPLATFORM
ARG TARGETPLATFORM

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Production stage
FROM python:3.13-slim as production

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Ensure venv is in PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY main.py ./
COPY pytest.ini ./

# Create necessary directories
RUN mkdir -p db models backups logs

# Copy default database files
COPY db/ ./db/

# Create example environment file
COPY .env.example ./

# Set proper permissions
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python main.py test-db || exit 1

# Default command
CMD ["python", "main.py", "--help"]

# Development stage (for docker-compose)
FROM production as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for development
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy development dependencies
COPY --from=builder /app /app

# Install development dependencies
WORKDIR /app
RUN uv sync --frozen

# Copy tests
COPY tests/ ./tests/

# Switch back to appuser
USER appuser

# Development command
CMD ["python", "main.py", "test-db"]

# MCP Server stage (for running MCP server)
FROM production as mcp-server

# Expose port for MCP server (if needed for future web interface)
EXPOSE 8000

# MCP server runs on stdio by default
CMD ["python", "main.py", "mcp-server"]

# Chat agent stage (for interactive chat)
FROM production as chat-agent

# Start interactive chat by default
CMD ["python", "main.py", "chat", "-i"]

# API server stage (for future API endpoints)
FROM production as api-server

# Install additional dependencies for API server
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

USER appuser

# Expose port for API
EXPOSE 8000

# Placeholder for future API server
CMD ["python", "main.py", "--help"]