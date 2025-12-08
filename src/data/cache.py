from __future__ import annotations
from pathlib import Path
from src.utils.config import RAW_DIR

RAW_DIR.mkdir(parents=True, exist_ok=True)


def cache_path(name: str) -> Path:
    return RAW_DIR / name
