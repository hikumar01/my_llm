# Multi-stage build to reduce final image size
# Stage 1: Build stage with all build dependencies
FROM python:3.12-slim AS builder

# Install build dependencies
# libclang-dev: Required for libclang Python package (symbol extraction)
# gcc/g++: May be needed if packages don't have pre-built wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libclang-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage with only runtime dependencies
FROM python:3.12-slim

# Install only runtime dependencies (libclang runtime, not dev)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libclang1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Copy application code
COPY frontend /app/frontend
COPY src /app/src

EXPOSE 8080
CMD ["python", "src/api_server.py"]
