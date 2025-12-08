# ML_Project

End-to-end ML system with FastAPI, Prefect, Docker, CI, and observability.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD
uvicorn src.api.app:app --reload
```

## Observability
- Metrics: `GET /metrics` (Prometheus format)
- JSON logs: method, path, status, duration_ms, X-Request-ID

## Prefect
- Flow: `src/prefect/flows.py:main_flow`
- Manual trigger: `POST /flow/run` (requires `ADMIN_API_KEY`)

### Scheduling (CLI)
```bash
prefect deployment build src/prefect/flows.py:main_flow -n mh-predictor-daily \
  --cron "0 2 * * *" -q default -o infra/prefect-deployment.yaml
prefect deployment apply infra/prefect-deployment.yaml
prefect agent start --work-queue default
```

### Scheduling (Python)
```bash
export PREFECT_CRON="0 2 * * *"
python scripts/schedule_prefect.py
prefect agent start --work-queue default
```

## CI
- GitHub Actions: runs black, flake8, mypy, pytest; builds Docker.
- Tags push image to GHCR.
# Multi-Horizon Stock Price Predictor

End-to-end ML system with FastAPI, Prefect, CI/CD, Docker, and a simple web UI.

## Quickstart (Local)
```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ALPHAVANTAGE_API_KEY=YOUR_KEY
uvicorn src.api.app:app --reload
# Open http://127.0.0.1:8000/
```

## Quickstart (Docker)
```bash
export ALPHAVANTAGE_API_KEY=YOUR_KEY
cd infra
docker compose up --build
# Open http://127.0.0.1:8000/
```

## Prefect Flow
```bash
python -m prefect version
python -m src.prefect.flows
```

## Project Layout
- `src/api/` FastAPI routes
- `src/data/` Alpha Vantage client & caching
- `src/features/` feature engineering & labels
- `src/models/` training, backtest, registry
- `src/prefect/` flows
- `src/web/` HTML/CSS/JS UI assets
- `tests/` pytest
- `infra/` Dockerfile & compose

## Notes
- Set `ALPHAVANTAGE_API_KEY` via env or `.env` in project root.
- Models saved under `models/` with metadata; production pointer in `models/production.txt`.
- CI builds & tests; mock or fixture API calls recommended for integration tests.
