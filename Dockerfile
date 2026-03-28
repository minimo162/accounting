# Stage 1: Build Svelte frontend
FROM node:22-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --ignore-scripts
COPY frontend/ ./
RUN npm run prepare && npm run build

# Stage 2: Python backend
FROM python:3.12-slim
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy Python project files
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ src/

# Create data directory (will be populated from GCS at runtime)
RUN mkdir -p data/index

# Copy built frontend
COPY --from=frontend-builder /app/frontend/build frontend/build

EXPOSE 8080

ENV USE_GCS=true
ENV GCS_BUCKET=jp-accounting-chat-data
ENV GCS_PREFIX=index

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
