import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
WEB_DIR = BASE_DIR / "src" / "web"

for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
DEFAULT_SYMBOL = os.getenv("DEFAULT_SYMBOL", "AAPL")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

# Tunables
HORIZONS = tuple(int(x) for x in os.getenv("HORIZONS", "1,3,7,30").split(","))
ALPHAVANTAGE_RATE_LIMIT_S = float(os.getenv("ALPHAVANTAGE_RATE_LIMIT_S", "12.0"))
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "30.0"))
PRICES_CACHE_TTL_S = int(os.getenv("PRICES_CACHE_TTL_S", "300"))
