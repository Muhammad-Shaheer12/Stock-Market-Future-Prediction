# Multi-stage build for FastAPI app
FROM python:3.11-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       curl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge
COPY src /app/src
COPY data /app/data
COPY models /app/models

# Add non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser \
    && chown -R appuser:appgroup /app
USER appuser
ENV PORT=8000
EXPOSE 8000

# Basic healthcheck hitting /health
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://localhost:${PORT}/health || exit 1
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
