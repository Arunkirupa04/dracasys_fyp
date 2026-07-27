"""Load shared integration/config.json — single source for demo + UI metadata."""
import json
from functools import lru_cache
from pathlib import Path

INTEGRATION_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = INTEGRATION_ROOT / "config.json"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def reload_config() -> dict:
    load_config.cache_clear()
    return load_config()


def demo_config() -> dict:
    return load_config().get("demo", {})
