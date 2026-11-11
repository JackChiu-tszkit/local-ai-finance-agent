"""Application settings.

Paths, currencies, categories, the timezone, and the Ollama connection live here.
Values saved from the settings page go to data/settings.json, which is not committed.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse


# Project root. pathlib keeps the path valid on Windows and macOS.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "finance.db"
SETTINGS_PATH = DATA_DIR / "settings.json"
EXPORT_DIR = DATA_DIR / "exports"

# Environment variables can point at a local Ollama on another machine.
DEFAULT_OLLAMA_HOST = os.environ.get("FINANCE_OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_TEXT_MODEL = os.environ.get("FINANCE_TEXT_MODEL", "qwen2.5:7b")
DEFAULT_VISION_MODEL = os.environ.get("FINANCE_VISION_MODEL", "qwen2.5vl:7b")
DEFAULT_CURRENCY = os.environ.get("FINANCE_DEFAULT_CURRENCY", "NOK")
DEFAULT_TIMEZONE = os.environ.get("FINANCE_TIMEZONE", "Europe/Oslo")
DEFAULT_WHISPER_MODEL = os.environ.get("FINANCE_WHISPER_MODEL", "small")
DEFAULT_WHISPER_DEVICE = os.environ.get("FINANCE_WHISPER_DEVICE", "cpu")
DEFAULT_WHISPER_COMPUTE = os.environ.get("FINANCE_WHISPER_COMPUTE", "int8")

# Large amounts get an extra warning. Every amount still requires confirmation.
DEFAULT_THRESHOLDS = {
    "CNY": "500",
    "NOK": "700",
    "USD": "70",
    "EUR": "70",
}

# Minor-unit exponent. Currencies not listed use 2 decimal places.
CURRENCY_EXPONENTS = {
    "NOK": 2,
    "CNY": 2,
    "USD": 2,
    "EUR": 2,
    "GBP": 2,
    "SEK": 2,
    "DKK": 2,
    "JPY": 0,
}

EXPENSE_CATEGORIES = [
    "Dining",
    "Groceries",
    "Transport",
    "Shopping",
    "Household",
    "Entertainment",
    "Health",
    "Housing",
    "Utilities",
    "Subscriptions",
    "Travel",
    "Other",
]

INCOME_CATEGORIES = [
    "Salary",
    "Investment",
    "Refund",
    "Gift",
    "Other income",
]

# Known merchants only suggest a category. A merchant that is not in the text is not saved.
MERCHANT_CATEGORIES = {
    "Kiwi": "Groceries",
    "Rema 1000": "Groceries",
    "Meny": "Groceries",
    "Coop": "Groceries",
    "Bunnpris": "Groceries",
    "Extra": "Groceries",
    "Netflix": "Subscriptions",
    "Spotify": "Subscriptions",
    "Uber": "Transport",
}

TRANSACTION_TYPE_LABELS = {
    "expense": "Expense",
    "income": "Income",
    "refund": "Refund",
}


def merchant_key(name: str) -> str:
    """Comparison key for merchant memory. Case and extra spaces are ignored."""
    return " ".join(name.casefold().split())


class ConfigError(ValueError):
    """Raised when a setting cannot be used."""


@dataclass
class Settings:
    """User settings stored as JSON."""

    ollama_host: str = DEFAULT_OLLAMA_HOST
    text_model: str = DEFAULT_TEXT_MODEL
    vision_model: str = DEFAULT_VISION_MODEL
    default_currency: str = DEFAULT_CURRENCY
    timezone: str = DEFAULT_TIMEZONE
    whisper_model: str = DEFAULT_WHISPER_MODEL
    whisper_device: str = DEFAULT_WHISPER_DEVICE
    whisper_compute_type: str = DEFAULT_WHISPER_COMPUTE
    whisper_language: str = "auto"
    confirm_all: bool = True
    thresholds: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))

    def validate(self) -> None:
        """Check the fields that affect the connection and how amounts are read."""
        validate_ollama_host(self.ollama_host)
        currency = self.default_currency.strip().upper()
        if len(currency) != 3 or not currency.isalpha():
            raise ConfigError("The default currency must be a 3-letter code, such as NOK.")
        self.default_currency = currency
        if not self.timezone.strip():
            raise ConfigError("The timezone cannot be empty.")
        if not self.text_model.strip() or not self.vision_model.strip():
            raise ConfigError("The text model and vision model names cannot be empty.")
        if self.whisper_device not in {"cpu", "cuda", "auto"}:
            raise ConfigError("The speech device must be cpu, cuda, or auto.")
        if self.whisper_language not in {"auto", "en", "no"}:
            self.whisper_language = "auto"
        cleaned: dict[str, str] = {}
        for code, raw_value in self.thresholds.items():
            normalized = str(code).strip().upper()
            if len(normalized) != 3 or not normalized.isalpha():
                raise ConfigError(f"Invalid threshold currency: {code}")
            cleaned[normalized] = str(raw_value).strip()
        self.thresholds = cleaned


def validate_ollama_host(host: str) -> str:
    """Allow only HTTP(S) so text and images are not sent to another scheme."""
    parsed = urlparse(host.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigError("The Ollama address must be an http or https URL, such as http://127.0.0.1:11434.")
    return host.strip().rstrip("/")


def load_settings(path: Path | None = None) -> Settings:
    """Load settings. A missing or broken file falls back to the defaults."""
    settings_path = path or SETTINGS_PATH
    if not settings_path.exists():
        settings = Settings()
        settings.validate()
        return settings
    try:
        raw = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        settings = Settings()
        settings.validate()
        return settings
    if not isinstance(raw, dict):
        settings = Settings()
        settings.validate()
        return settings
    known = {item.name for item in Settings.__dataclass_fields__.values()}
    payload = {key: value for key, value in raw.items() if key in known}
    settings = Settings(**payload)
    settings.validate()
    return settings


def save_settings(settings: Settings, path: Path | None = None) -> None:
    """Write settings atomically so a Streamlit rerun cannot leave a partial file."""
    settings.validate()
    settings_path = path or SETTINGS_PATH
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = settings_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(asdict(settings), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    temporary.replace(settings_path)
