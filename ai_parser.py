"""自然语言交易解析。

金额、日期和收支类型优先由确定性规则提取。本地 Ollama 只补充分类、
商户和描述建议，而且必须能在原文中找到依据。模型不可用时仍可解析
常见句式，并明确告诉用户语言模型没有参与。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Callable
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import (
    EXPENSE_CATEGORIES,
    INCOME_CATEGORIES,
    MERCHANT_CATEGORIES,
    Settings,
    validate_ollama_host,
)


class OllamaError(RuntimeError):
    """连接或调用本地 Ollama 失败。"""


class OllamaHTTPError(OllamaError):
    """Ollama 返回了 HTTP 错误。"""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class LLMTransactionPayload(BaseModel):
    """校验模型返回的 JSON。未知字段会被忽略，非法类型会被拒绝。"""

    model_config = ConfigDict(extra="ignore")

    transaction_date: str | None = None
    amount: str | None = None
    currency: str | None = None
    transaction_type: str | None = None
    category: str | None = None
    description: str | None = None
    merchant: str | None = None
    account: str | None = None
    needs_clarification: bool = False
    clarification_question: str | None = None

    @field_validator("amount", mode="before")
    @classmethod
    def coerce_amount(cls, value: Any) -> str | None:
        """把模型可能返回的数字转成字符串，再用 Decimal 检查格式。"""
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            raise ValueError("金额格式无效")
        if isinstance(value, float):
            text = format(Decimal(str(value)), "f")
        else:
            text = str(value).strip().replace(",", "")
        try:
            Decimal(text)
        except InvalidOperation as exc:
            raise ValueError("金额格式无效") from exc
        return text

    @field_validator("transaction_type")
    @classmethod
    def check_type(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        if value not in {"expense", "income", "refund"}:
            raise ValueError("交易类型无效")
        return value

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency(cls, value: Any) -> str | None:
        if value is None or value == "":
            return None
        text = str(value).strip().upper()
        if len(text) != 3 or not text.isalpha():
            return None
        return text


class TransactionDraft(BaseModel):
    """等待用户确认的交易草稿。确认前不会写入数据库。"""

    model_config = ConfigDict(extra="ignore")

    transaction_date: date | None = None
    date_assumed: bool = False
    date_explicit: bool = False
    amount: Decimal | None = None
    currency: str | None = None
    currency_assumed: bool = False
    transaction_type: str | None = None
    type_confident: bool = False
    category: str | None = None
    description: str | None = None
    merchant: str | None = None
    account: str | None = None
    needs_clarification: bool = False
    clarification_question: str | None = None
    warnings: list[str] = Field(default_factory=list)
    possible_amounts: list[Decimal] = Field(default_factory=list)
    search_start: date | None = None
    search_end: date | None = None
    items: list[str] = Field(default_factory=list)
    source: str = "text"
    parser: str = "rules"
    raw_text: str = ""


@dataclass
class AmountHit:
    """原文中的一处金额。currency 为空表示旁边没有币种词。"""

    amount: Decimal
    currency: str | None
    index: int


_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_CN_TRAILING_UNIT = {"万": 1000, "千": 100, "百": 10}

_CURRENCY_PATTERN = re.compile(
    r"挪威克朗|瑞典克朗|丹麦克朗|人民币|"
    r"\bRMB\b|\bCNY\b|\bUSD\b|\bEUR\b|\bGBP\b|\bSEK\b|\bDKK\b|\bNOK\b|"
    r"美元|欧元|英镑|克朗|\bkroner\b|块钱|元|块|[€£$]|\bkr\b",
    re.IGNORECASE,
)
_ARABIC_AT_END = re.compile(r"(\d+(?:\.\d{1,2})?)\s*$")
_CHINESE_AT_END = re.compile(r"([零〇一二两三四五六七八九十百千万]+)\s*$")
_ARABIC_AT_START = re.compile(r"^\s*(\d+(?:\.\d{1,2})?)")
_ANCHOR = re.compile(r"花了|花费|支付|付了|消费了|收到|退回了|退回|退了|退款|spent|paid")
_FULL_DATE = re.compile(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})")
_MONTH_DAY = re.compile(r"(\d{1,2})月(\d{1,2})[日号]")
_WEEKDAY = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}

_KNOWN_MERCHANTS = [
    (re.compile(r"rema\s*1000", re.IGNORECASE), "Rema 1000"),
    (re.compile(r"\bbunnpris\b", re.IGNORECASE), "Bunnpris"),
    (re.compile(r"\bnetflix\b", re.IGNORECASE), "Netflix"),
    (re.compile(r"\bspotify\b", re.IGNORECASE), "Spotify"),
    (re.compile(r"\bkiwi\b", re.IGNORECASE), "Kiwi"),
    (re.compile(r"\bmeny\b", re.IGNORECASE), "Meny"),
    (re.compile(r"\bcoop\b", re.IGNORECASE), "Coop"),
    (re.compile(r"\buber\b", re.IGNORECASE), "Uber"),
    (re.compile(r"\bextra\b", re.IGNORECASE), "Extra"),
]
_ACCOUNTS = ["微信支付", "支付宝", "微信", "银行卡", "信用卡", "Apple Pay", "Vipps", "Visa", "Mastercard"]
_REFUND_WORDS = ("退款", "退回", "退钱", "已退", "退了", "refund", "refusjon")
_INCOME_WORDS = ("工资", "薪水", "薪资", "收入", "进账", "收到", "salary", "lønn", "lonn")
_EXPENSE_WORDS = (
    "花了",
    "花费",
    "支付",
    "消费",
    "买",
    "付款",
    "paid",
    "spent",
    "购买",
    "brukte",
    "betalte",
    "kjøpte",
    "kjopte",
)
_EXPENSE_KEYWORDS = [
    ("食品杂货", ("买菜", "超市", "grocery", "groceries", "kiwi", "rema", "meny", "coop", "bunnpris")),
    ("餐饮", ("火锅", "吃饭", "午餐", "晚饭", "晚餐", "早餐", "午饭", "餐厅", "咖啡", "外卖", "奶茶", "午饭")),
    ("交通", ("公交", "地铁", "打车", "出租", "uber", "火车", "机票", "加油", "停车", "bus")),
    ("订阅服务", ("netflix", "spotify", "订阅", "会员")),
    ("住房", ("房租", "租金", "房贷")),
    ("水电及公共事业费用", ("电费", "水费", "网费", "燃气", "物业")),
    ("医疗健康", ("医院", "药店", "挂号", "药费")),
    ("娱乐", ("电影", "游戏", "ktv", "演出")),
    ("旅行", ("酒店", "旅行", "旅游", "机票")),
    ("日用品", ("日用", "纸巾", "洗衣液")),
    ("购物", ("购物", "衣服", "淘宝", "京东")),
]
_INCOME_KEYWORDS = [
    ("工资", ("工资", "薪水", "薪资", "salary", "lønn", "lonn")),
    ("投资收益", ("股息", "理财", "利息", "投资")),
    ("礼金", ("红包", "礼金")),
    ("退款", ("退款", "退回", "refund", "refusjon")),
]

_LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "transaction_date": {"type": ["string", "null"]},
        "amount": {"type": ["string", "null"]},
        "currency": {"type": ["string", "null"]},
        "transaction_type": {"type": ["string", "null"]},
        "category": {"type": ["string", "null"]},
        "description": {"type": ["string", "null"]},
        "merchant": {"type": ["string", "null"]},
        "account": {"type": ["string", "null"]},
        "needs_clarification": {"type": "boolean"},
        "clarification_question": {"type": ["string", "null"]},
    },
    "required": [
        "transaction_date",
        "amount",
        "currency",
        "transaction_type",
        "category",
        "description",
        "merchant",
        "account",
        "needs_clarification",
        "clarification_question",
    ],
}


def today_in_timezone(timezone_name: str, now: datetime | None = None) -> date:
    """按配置时区计算今天。now 只在测试中注入。"""
    current = now or datetime.now(ZoneInfo(_safe_timezone(timezone_name)))
    if current.tzinfo is None:
        current = current.replace(tzinfo=ZoneInfo(_safe_timezone(timezone_name)))
    return current.astimezone(ZoneInfo(_safe_timezone(timezone_name))).date()


def _safe_timezone(timezone_name: str) -> str:
    try:
        ZoneInfo(timezone_name)
    except Exception:
        return "UTC"
    return timezone_name


def normalize_text(text: str) -> str:
    """把全角数字转成半角，便于统一提取金额。"""
    translation = str.maketrans("０１２３４５６７８９．，", "0123456789.,")
    return text.translate(translation).strip()


def parse_chinese_integer(token: str) -> int | None:
    """解析常见中文整数。

    “三百五”“四万五”这类口语省略按下一档单位解释：
    三百五是 350，四万五是 45000。带“零”的写法仍按标准读法，三百零五是 305。
    """
    if not token:
        return None
    total = 0
    section = 0
    number = 0
    last_unit = ""
    for char in token:
        if char in _CN_DIGITS:
            number = _CN_DIGITS[char]
            continue
        if char in _CN_SMALL_UNITS:
            unit = _CN_SMALL_UNITS[char]
            if number == 0:
                number = 1
            section += number * unit
            number = 0
            last_unit = char
            continue
        if char == "万":
            section += number
            if section == 0:
                return None
            total += section * 10000
            section = 0
            number = 0
            last_unit = "万"
            continue
        return None
    if (
        last_unit in _CN_TRAILING_UNIT
        and number
        and len(token) >= 2
        and token[-1] in _CN_DIGITS
        and token[-2] == last_unit
    ):
        return total + section + number * _CN_TRAILING_UNIT[last_unit]
    return total + section + number


def _currency_code(token: str) -> str:
    folded = token.casefold()
    mapping = {
        "挪威克朗": "NOK",
        "克朗": "NOK",
        "kroner": "NOK",
        "kr": "NOK",
        "nok": "NOK",
        "瑞典克朗": "SEK",
        "sek": "SEK",
        "丹麦克朗": "DKK",
        "dkk": "DKK",
        "人民币": "CNY",
        "rmb": "CNY",
        "cny": "CNY",
        "元": "CNY",
        "块": "CNY",
        "块钱": "CNY",
        "美元": "USD",
        "usd": "USD",
        "$": "USD",
        "欧元": "EUR",
        "eur": "EUR",
        "€": "EUR",
        "英镑": "GBP",
        "gbp": "GBP",
        "£": "GBP",
    }
    return mapping[folded]


def _number_near_currency(text: str, start: int, end: int, token: str) -> Decimal | None:
    """从币种词前方或符号后方提取数字。找不到时返回空。"""
    if token in {"$", "€", "£"}:
        forward = _ARABIC_AT_START.search(text[end:end + 16])
        if forward:
            return Decimal(forward.group(1))
    window = text[max(0, start - 16):start]
    arabic = _ARABIC_AT_END.search(window)
    if arabic:
        return Decimal(arabic.group(1))
    chinese = _CHINESE_AT_END.search(window)
    if chinese:
        value = parse_chinese_integer(chinese.group(1))
        if value is not None and value > 0:
            return Decimal(value)
    return None


def extract_amounts(text: str) -> list[AmountHit]:
    """提取原文中带位置的金额。同一位置只保留一个结果。"""
    hits: list[AmountHit] = []
    occupied: set[int] = set()
    for match in _CURRENCY_PATTERN.finditer(text):
        amount = _number_near_currency(text, match.start(), match.end(), match.group())
        if amount is None or amount <= 0:
            continue
        if match.start() in occupied:
            continue
        occupied.add(match.start())
        hits.append(AmountHit(amount, _currency_code(match.group()), match.start()))
    if hits:
        return hits
    for match in re.finditer(r"\d+(?:\.\d{1,2})?", text):
        hits.append(AmountHit(Decimal(match.group()), None, match.start()))
    return hits


def choose_amount(text: str, hits: list[AmountHit]) -> tuple[AmountHit | None, list[Decimal]]:
    """多个金额时，优先选择紧跟“花了 / 收到 / 退回”的那一个。"""
    distinct = []
    for hit in hits:
        if hit.amount not in distinct:
            distinct.append(hit.amount)
    if not hits:
        return None, []
    if len(hits) == 1:
        return hits[0], distinct
    anchors = list(_ANCHOR.finditer(text))
    if anchors:
        anchor = anchors[-1].end()
        following = [hit for hit in hits if hit.index >= anchor]
        pool = following or hits
        chosen = min(pool, key=lambda hit: abs(hit.index - anchor))
        if len({hit.amount for hit in pool}) == 1:
            return chosen, distinct
        if following and len(following) == 1:
            return following[0], distinct
    if len(distinct) == 1:
        return hits[0], distinct
    return None, distinct


def _make_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def resolve_date(text: str, today: date) -> tuple[date | None, bool, bool, date | None, date | None]:
    """解析相对日期和绝对日期。

    返回值依次是：日期、是否为假定日期、是否在原文中明确出现、
    退款搜索起始日、退款搜索结束日。
    “今天 / 昨天 / 前天”只按传入的本地日期计算，不交给模型推算。
    """
    full = _FULL_DATE.search(text)
    if full:
        parsed = _make_date(int(full.group(1)), int(full.group(2)), int(full.group(3)))
        if parsed:
            return parsed, False, True, parsed, parsed
    month_day = _MONTH_DAY.search(text)
    if month_day:
        month = int(month_day.group(1))
        day = int(month_day.group(2))
        parsed = _make_date(today.year, month, day)
        if parsed and parsed > today + timedelta(days=1):
            parsed = _make_date(today.year - 1, month, day)
        if parsed:
            return parsed, False, True, parsed, parsed

    folded = text.casefold()
    if "前天" in text or "i forgårs" in folded or "i forgaars" in folded:
        target = today - timedelta(days=2)
        return target, False, True, target, target
    if "昨天" in text or "昨日" in text or "yesterday" in folded or re.search(r"\bi går\b", folded):
        target = today - timedelta(days=1)
        return target, False, True, target, target
    if "今天" in text or "今日" in text or "today" in folded or re.search(r"\bi dag\b", folded):
        return today, False, True, today, today

    week_match = re.search(r"(上上周|上周|上星期|本周|这周)([一二三四五六日天])", text)
    if week_match:
        prefix, weekday_token = week_match.group(1), week_match.group(2)
        weeks_back = 2 if prefix == "上上周" else 1 if prefix in {"上周", "上星期"} else 0
        target = _weekday_in_week(today, weeks_back, _WEEKDAY[weekday_token])
        return target, False, True, target, target
    lone_weekday = re.search(r"(?<![上本这])周([一二三四五六日天])", text)
    if lone_weekday:
        weekday = _WEEKDAY[lone_weekday.group(1)]
        target = _weekday_in_week(today, 0, weekday)
        if target > today:
            target = _weekday_in_week(today, 1, weekday)
        return target, False, True, target, target
    if "上周" in text or "上星期" in text or "last week" in folded:
        monday = today - timedelta(days=today.weekday())
        start = monday - timedelta(days=7)
        end = monday - timedelta(days=1)
        return None, False, False, start, end
    return today, True, False, None, None


def _weekday_in_week(today: date, weeks_back: int, weekday: int) -> date:
    monday = today - timedelta(days=today.weekday())
    return monday - timedelta(days=7 * weeks_back) + timedelta(days=weekday)


def detect_type(text: str) -> tuple[str | None, bool]:
    """判断收入、支出或退款。退款词优先于“收到”。"""
    folded = text.casefold()
    refund = any(word.casefold() in folded for word in _REFUND_WORDS)
    income = any(word.casefold() in folded for word in _INCOME_WORDS)
    expense = any(word.casefold() in folded for word in _EXPENSE_WORDS)
    if refund:
        return "refund", True
    if income and expense:
        return None, False
    if income:
        return "income", True
    if expense:
        return "expense", True
    return None, False


def extract_merchant(text: str) -> str | None:
    """只返回原文中真实出现的已知商户或“在 XXX 买”里的拉丁商户名。"""
    for pattern, canonical in _KNOWN_MERCHANTS:
        if pattern.search(text):
            return canonical
    match = re.search(r"在\s*([A-Za-z][A-Za-z0-9&.'’\- ]{1,30}?)\s*(?:买|花|消费|支付)", text)
    if match:
        return " ".join(match.group(1).split())
    return None


def extract_account(text: str) -> str | None:
    """提取原文里明确出现的支付账户。"""
    folded = text.casefold()
    for name in _ACCOUNTS:
        if name.casefold() in folded:
            return name
    return None


def infer_category(text: str, transaction_type: str | None, merchant: str | None) -> str:
    """根据商户和关键词给出分类。没有可靠依据时归入“其他”。"""
    if transaction_type == "refund":
        return "退款"
    if merchant in MERCHANT_CATEGORIES and transaction_type != "income":
        return MERCHANT_CATEGORIES[merchant]
    folded = text.casefold()
    if transaction_type == "income":
        for category, keywords in _INCOME_KEYWORDS:
            if any(keyword.casefold() in folded for keyword in keywords):
                return category
        return "其他收入"
    for category, keywords in _EXPENSE_KEYWORDS:
        if any(keyword.casefold() in folded for keyword in keywords):
            return category
    return "其他"


def parse_with_rules(text: str, *, today: date, default_currency: str) -> TransactionDraft:
    """用确定性规则解析一句话。缺少金额时不会编造。"""
    original = text.strip()
    cleaned = normalize_text(original)
    warnings: list[str] = []
    if not cleaned:
        return TransactionDraft(
            needs_clarification=True,
            clarification_question="请输入一笔交易。",
            raw_text=text,
            parser="rules",
        )
    transaction_type, type_confident = detect_type(cleaned)
    if transaction_type is None:
        transaction_type = "expense"
        warnings.append("没有判断出收入还是支出，已暂按支出填写，请在确认前修改。")
    hits = extract_amounts(cleaned)
    chosen, distinct = choose_amount(cleaned, hits)
    amount = chosen.amount if chosen else None
    currency = chosen.currency if chosen and chosen.currency else None
    currency_assumed = False
    possible = distinct if amount is None and len(distinct) > 1 else []
    if amount is None and len(distinct) > 1:
        warnings.append("识别到多个金额，请选择实际发生的那一个。")
    if amount is not None and currency is None:
        currency = default_currency.upper()
        currency_assumed = True
        warnings.append(f"没有识别到币种，已暂用默认币种 {currency}。")
    if amount is None and not possible:
        warnings.append("没有识别到金额，请补充后再保存。")

    resolved, assumed, explicit, search_start, search_end = resolve_date(cleaned, today)
    if transaction_type == "refund" and resolved is None and search_start and search_end:
        resolved = today
        assumed = True
        warnings.append("退款入账日期暂按今天填写，原交易会在上周范围内查找。")
    elif resolved is None:
        warnings.append("请补充具体交易日期。")
    elif assumed:
        warnings.append("没有识别到日期，已暂按今天填写，请确认。")

    merchant = extract_merchant(cleaned)
    category = infer_category(cleaned, transaction_type, merchant)
    draft = TransactionDraft(
        transaction_date=resolved,
        date_assumed=assumed,
        date_explicit=explicit,
        amount=amount,
        currency=currency,
        currency_assumed=currency_assumed,
        transaction_type=transaction_type,
        type_confident=type_confident,
        category=category,
        description=original[:500],
        merchant=merchant,
        account=extract_account(cleaned),
        warnings=warnings,
        possible_amounts=possible,
        search_start=search_start,
        search_end=search_end,
        source="text",
        parser="rules",
        raw_text=original,
    )
    return _mark_clarification(draft)


def _mark_clarification(draft: TransactionDraft) -> TransactionDraft:
    """根据仍缺失的关键信息决定要向用户提出的问题。"""
    if draft.amount is None:
        draft.needs_clarification = True
        draft.clarification_question = (
            "请选择正确的金额。" if draft.possible_amounts else "请补充金额。"
        )
        return draft
    if draft.transaction_date is None:
        draft.needs_clarification = True
        draft.clarification_question = "请补充具体交易日期。"
        return draft
    if draft.date_assumed or not draft.type_confident:
        draft.needs_clarification = True
        draft.clarification_question = (
            "请确认交易日期。" if draft.date_assumed else "请确认这是收入、支出还是退款。"
        )
        return draft
    draft.needs_clarification = False
    draft.clarification_question = None
    return draft


def extract_json_object(text: str) -> dict[str, Any]:
    """从模型输出中提取第一个 JSON 对象，容许 Markdown 代码块。"""
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    candidate = fence.group(1) if fence else stripped
    decoder = json.JSONDecoder()
    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise OllamaError("模型没有返回可解析的 JSON。")


def amount_grounded(amount: Decimal, text: str) -> bool:
    """金额必须能在原文中找到阿拉伯数字或对应中文数字。"""
    normalized = text.replace(",", "")
    if amount == amount.to_integral_value():
        integer_text = str(int(amount))
        if integer_text in normalized:
            return True
    plain = format(amount, "f")
    if plain in normalized:
        return True
    for token in re.findall(r"[零〇一二两三四五六七八九十百千万]+", text):
        value = parse_chinese_integer(token)
        if value is not None and Decimal(value) == amount:
            return True
    return False


def phrase_in_text(text: str, phrase: str | None) -> bool:
    """商户和账户必须是原文的子串，大小写不敏感。"""
    if not phrase:
        return False
    return phrase.casefold() in text.casefold()


def categories_for_type(transaction_type: str | None) -> list[str]:
    """返回某个交易类型允许使用的分类名称。"""
    if transaction_type == "expense":
        return list(EXPENSE_CATEGORIES)
    return list(INCOME_CATEGORIES)


def merge_llm_payload(
    rules: TransactionDraft,
    payload: LLMTransactionPayload,
    source_text: str,
) -> TransactionDraft:
    """合并规则结果和模型结果。规则已经确定的金额、日期和类型不被模型覆盖。"""
    merged = rules.model_copy(deep=True)
    merged.parser = "ollama"
    if merged.amount is None and payload.amount is not None:
        candidate = Decimal(payload.amount)
        if candidate > 0 and amount_grounded(candidate, source_text):
            merged.amount = candidate
            merged.possible_amounts = []
            merged.warnings = [
                item
                for item in merged.warnings
                if "没有识别到金额" not in item and "多个金额" not in item
            ]
        else:
            merged.warnings.append("模型给出的金额在原文中找不到，已忽略。")
    if payload.currency and (payload.currency in source_text.upper() or not merged.currency_assumed):
        if not merged.currency_assumed and merged.currency:
            pass
        elif payload.currency in source_text.upper():
            merged.currency = payload.currency
            merged.currency_assumed = False
    if payload.transaction_type and not merged.type_confident:
        merged.transaction_type = payload.transaction_type
        merged.type_confident = True
    allowed = categories_for_type(merged.transaction_type)
    generic = {"其他", "其他收入", "退款"}
    if payload.category in allowed and merged.category in generic:
        merged.category = payload.category
    if merged.transaction_type == "refund":
        merged.category = "退款"
    if merged.category not in categories_for_type(merged.transaction_type):
        merged.category = {
            "income": "其他收入",
            "refund": "退款",
        }.get(merged.transaction_type or "", "其他")
    if not merged.merchant and phrase_in_text(source_text, payload.merchant):
        merged.merchant = payload.merchant.strip() if payload.merchant else None
        if merged.category in generic and merged.merchant in MERCHANT_CATEGORIES:
            merged.category = MERCHANT_CATEGORIES[merged.merchant]
    elif payload.merchant and not phrase_in_text(source_text, payload.merchant):
        merged.warnings.append("模型补充的商户没有出现在原文中，已忽略。")
    if not merged.account and phrase_in_text(source_text, payload.account):
        merged.account = payload.account.strip() if payload.account else None
    return _mark_clarification(merged)


def build_llm_messages(text: str, today: date, default_currency: str) -> list[dict[str, str]]:
    """构造约束很强的提示词，要求缺失信息返回 null。"""
    system = (
        "你是本地记账提取器。只根据用户原文输出 JSON，不能编造金额、商户、账户或商品。"
        "原文没有的字段必须是 null。金额用字符串。日期格式为 YYYY-MM-DD。"
        f"今天是 {today.isoformat()}。默认币种是 {default_currency}，只有原文出现币种时才能填写 currency。"
        f"支出分类只能是：{'、'.join(EXPENSE_CATEGORIES)}。"
        f"收入分类只能是：{'、'.join(INCOME_CATEGORIES)}。"
        "退款的 transaction_type 使用 refund，category 使用退款。"
        "不确定时把 needs_clarification 设为 true，并在 clarification_question 写出需要问用户的问题。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


def request_ollama_chat(
    host: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    timeout: float = 120,
    schema: dict[str, Any] | None = _LLM_SCHEMA,
) -> dict[str, Any]:
    """调用本机 Ollama 聊天接口。连接失败时抛出中文错误。"""
    base = validate_ollama_host(host)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0},
    }
    if schema is not None:
        payload["format"] = schema
    try:
        response = httpx.post(f"{base}/api/chat", json=payload, timeout=timeout)
    except httpx.TimeoutException as exc:
        raise OllamaError("本地模型响应超时。请稍后重试，或换一个更小的模型。") from exc
    except httpx.HTTPError as exc:
        raise OllamaError(
            f"无法连接本地 Ollama（{base}）。请确认 Ollama 已安装并正在运行。"
        ) from exc
    if response.status_code >= 400:
        detail = response.text.strip()[:300]
        raise OllamaHTTPError(response.status_code, f"Ollama 返回错误 {response.status_code}：{detail}")
    try:
        body = response.json()
    except json.JSONDecodeError as exc:
        raise OllamaError("Ollama 返回的内容不是 JSON。") from exc
    message = body.get("message", {})
    content = message.get("content")
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        return extract_json_object(content)
    raise OllamaError("Ollama 没有返回消息内容。")


def list_installed_models(host: str, timeout: float = 3) -> list[str]:
    """读取本机已安装的 Ollama 模型名称。"""
    base = validate_ollama_host(host)
    try:
        response = httpx.get(f"{base}/api/tags", timeout=timeout)
    except httpx.HTTPError as exc:
        raise OllamaError(f"无法连接本地 Ollama（{base}）。请确认服务已启动。") from exc
    if response.status_code >= 400:
        raise OllamaHTTPError(response.status_code, f"读取模型列表失败：{response.status_code}")
    payload = response.json()
    names = []
    for item in payload.get("models", []):
        name = item.get("name")
        if isinstance(name, str):
            names.append(name)
    return names


def parse_transaction_text(
    text: str,
    *,
    today: date,
    default_currency: str,
    settings: Settings | None = None,
    use_ollama: bool = True,
    requester: Callable[..., dict[str, Any]] | None = None,
    timeout: float = 120,
) -> TransactionDraft:
    """解析自然语言。模型失败时保留规则结果，并写入可见警告。"""
    rules = parse_with_rules(text, today=today, default_currency=default_currency)
    if not use_ollama or not rules.raw_text:
        return rules
    active = settings or Settings()
    caller = requester or request_ollama_chat
    messages = build_llm_messages(normalize_text(rules.raw_text), today, default_currency)
    try:
        try:
            raw_payload = caller(
                active.ollama_host,
                active.text_model,
                messages,
                timeout=timeout,
                schema=_LLM_SCHEMA,
            )
        except OllamaHTTPError as exc:
            if exc.status != 400:
                raise
            raw_payload = caller(
                active.ollama_host,
                active.text_model,
                messages,
                timeout=timeout,
                schema=None,
            )
        payload = LLMTransactionPayload.model_validate(raw_payload)
    except OllamaError as exc:
        rules.warnings.insert(0, str(exc))
        return rules
    except ValueError:
        rules.warnings.insert(0, "模型输出没有通过校验，已改用规则解析。")
        return rules
    return merge_llm_payload(rules, payload, normalize_text(rules.raw_text))
