from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
SCREENSHOTS_DIR = ROOT_DIR / "screenshots"
DEFAULT_SYNTHETIC_CSV = DATA_DIR / "synthetic_sales.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "sales_model.joblib"
DEFAULT_METRICS_PATH = OUTPUTS_DIR / "evaluation_metrics.json"
DEFAULT_FORECAST_PATH = OUTPUTS_DIR / "future_forecast_14d.csv"
DEFAULT_TEST_PREDICTIONS_PATH = OUTPUTS_DIR / "test_predictions.csv"

load_dotenv(ROOT_DIR / ".env", override=False)


def get_env_var(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) and value.strip() else default


def get_snowflake_config() -> dict[str, str]:
    return {
        "account": get_env_var("SNOWFLAKE_ACCOUNT") or "",
        "user": get_env_var("SNOWFLAKE_USER") or "",
        "password": get_env_var("SNOWFLAKE_PASSWORD") or "",
        "warehouse": get_env_var("SNOWFLAKE_WAREHOUSE") or "",
        "database": get_env_var("SNOWFLAKE_DATABASE") or "",
        "schema": get_env_var("SNOWFLAKE_SCHEMA") or "",
    }


def snowflake_config_available() -> bool:
    cfg = get_snowflake_config()
    return all(cfg.values())


def require_snowflake_config() -> dict[str, str]:
    cfg = get_snowflake_config()
    missing = [key for key, value in cfg.items() if not value]
    if missing:
        missing_display = ", ".join(missing)
        raise RuntimeError(
            "Snowflake credentials are missing or incomplete. "
            f"Set the following environment variables in a .env file: {missing_display}. "
            "Use the .env.example template as a guide."
        )
    return cfg
