"""应用配置。

路径、币种、分类、时区和 Ollama 连接信息都集中在这里。
用户在设置页保存的内容写入 data/settings.json，该文件不进入 Git。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse


# 项目根目录。使用 pathlib，避免写死 Windows 或 macOS 路径。
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "finance.db"
SETTINGS_PATH = DATA_DIR / "settings.json"
EXPORT_DIR = DATA_DIR / "exports"

# 环境变量可以覆盖默认值，方便在不同机器上指向本机 Ollama。
DEFAULT_OLLAMA_HOST = os.environ.get("FINANCE_OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_TEXT_MODEL = os.environ.get("FINANCE_TEXT_MODEL", "qwen2.5:7b")
DEFAULT_VISION_MODEL = os.environ.get("FINANCE_VISION_MODEL", "qwen2.5vl:7b")
DEFAULT_CURRENCY = os.environ.get("FINANCE_DEFAULT_CURRENCY", "NOK")
DEFAULT_TIMEZONE = os.environ.get("FINANCE_TIMEZONE", "Europe/Oslo")
DEFAULT_WHISPER_MODEL = os.environ.get("FINANCE_WHISPER_MODEL", "small")
DEFAULT_WHISPER_DEVICE = os.environ.get("FINANCE_WHISPER_DEVICE", "cpu")
DEFAULT_WHISPER_COMPUTE = os.environ.get("FINANCE_WHISPER_COMPUTE", "int8")

# 金额超过阈值时做醒目提示。MVP 阶段无论金额大小都必须确认，阈值不用于跳过确认。
DEFAULT_THRESHOLDS = {
    "CNY": "500",
    "NOK": "700",
    "USD": "70",
    "EUR": "70",
}

# 最小货币单位的小数位数。未列出的币种按 2 位处理。
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
    "餐饮",
    "食品杂货",
    "交通",
    "购物",
    "日用品",
    "娱乐",
    "医疗健康",
    "住房",
    "水电及公共事业费用",
    "订阅服务",
    "旅行",
    "其他",
]

INCOME_CATEGORIES = [
    "工资",
    "投资收益",
    "退款",
    "礼金",
    "其他收入",
]

# 已知商户只用于建议分类。文本里没有出现的商户不会被写入交易。
MERCHANT_CATEGORIES = {
    "Kiwi": "食品杂货",
    "Rema 1000": "食品杂货",
    "Meny": "食品杂货",
    "Coop": "食品杂货",
    "Bunnpris": "食品杂货",
    "Extra": "食品杂货",
    "Netflix": "订阅服务",
    "Spotify": "订阅服务",
    "Uber": "交通",
}

TRANSACTION_TYPE_LABELS = {
    "expense": "支出",
    "income": "收入",
    "refund": "退款",
}


class ConfigError(ValueError):
    """配置内容无法使用时抛出。"""


@dataclass
class Settings:
    """可持久化的用户设置。字段保持简单，便于 JSON 读写。"""

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
        """检查会直接影响连接和金额解释的字段。"""
        validate_ollama_host(self.ollama_host)
        currency = self.default_currency.strip().upper()
        if len(currency) != 3 or not currency.isalpha():
            raise ConfigError("默认币种必须是 3 位字母代码，例如 NOK 或 CNY。")
        self.default_currency = currency
        if not self.timezone.strip():
            raise ConfigError("时区不能为空。")
        if not self.text_model.strip() or not self.vision_model.strip():
            raise ConfigError("文本模型和视觉模型名称不能为空。")
        if self.whisper_device not in {"cpu", "cuda", "auto"}:
            raise ConfigError("语音推理设备只能是 cpu、cuda 或 auto。")
        if self.whisper_language not in {"auto", "zh", "en", "no"}:
            raise ConfigError("语音语言只能是自动、中文、英文或挪威语。")
        cleaned: dict[str, str] = {}
        for code, raw_value in self.thresholds.items():
            normalized = str(code).strip().upper()
            if len(normalized) != 3 or not normalized.isalpha():
                raise ConfigError(f"阈值币种无效：{code}")
            cleaned[normalized] = str(raw_value).strip()
        self.thresholds = cleaned


def validate_ollama_host(host: str) -> str:
    """只允许 HTTP(S) 地址，避免把图片或文本发到非预期协议。"""
    parsed = urlparse(host.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigError("Ollama 地址必须是 http 或 https URL，例如 http://127.0.0.1:11434。")
    return host.strip().rstrip("/")


def load_settings(path: Path | None = None) -> Settings:
    """读取设置。文件不存在或损坏时返回默认值，不中断应用启动。"""
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
    """原子写入设置，避免 Streamlit 重跑时写出半份 JSON。"""
    settings.validate()
    settings_path = path or SETTINGS_PATH
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = settings_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(asdict(settings), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(settings_path)
