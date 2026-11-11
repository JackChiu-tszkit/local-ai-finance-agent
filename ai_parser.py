"""Natural-language transaction parsing.

Amount, date, and transaction type come from deterministic rules first.
A local Ollama model may only fill category, merchant, and description
when those values are grounded in the original text. Common English and
Norwegian sentences still parse when the model is unavailable.
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
    merchant_key,
    validate_ollama_host,
)


class OllamaError(RuntimeError):
    """The local Ollama service could not be reached or called."""


class OllamaHTTPError(OllamaError):
    """Ollama returned an HTTP error."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class LLMTransactionPayload(BaseModel):
    """Validated JSON from the model. Unknown fields are ignored."""

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
        """Turn a numeric model value into a string, then check it with Decimal."""
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            raise ValueError("Invalid amount")
        if isinstance(value, float):
            text = format(Decimal(str(value)), "f")
        else:
            text = str(value).strip().replace(",", "")
        try:
            Decimal(text)
        except InvalidOperation as exc:
            raise ValueError("Invalid amount") from exc
        return text

    @field_validator("transaction_type")
    @classmethod
    def check_type(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        if value not in {"expense", "income", "refund"}:
            raise ValueError("Invalid transaction type")
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
    """A draft waiting for confirmation. Nothing is written before that."""

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
    """One amount in the text. currency is empty when no currency word is beside it."""

    amount: Decimal
    currency: str | None
    index: int


_CURRENCY_PATTERN = re.compile(
    r"norske kroner|svenske kroner|danske kroner|"
    r"\bRMB\b|\bCNY\b|\bUSD\b|\bEUR\b|\bGBP\b|\bSEK\b|\bDKK\b|\bNOK\b|"
    r"\bkroner\b|\byuan\b|\bdollars?\b|\beuros?\b|\bpounds?\b|[€£$]|\bkr\b",
    re.IGNORECASE,
)
# The grouped form must come first, or "1,200.50" is read as 1.
# Commas are removed only from the matched token.
_ARABIC_NUMBER = r"(?:\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)"
_ARABIC_AT_END = re.compile(rf"({_ARABIC_NUMBER})\s*$")
_ARABIC_AT_START = re.compile(rf"^\s*({_ARABIC_NUMBER})")
# 12,345,678,901.25 is 17 characters, so a 16-character window would drop the start.
_AMOUNT_LOOKAROUND = 32
_ANCHOR = re.compile(
    r"\b(?:spent|paid|bought|received|refund(?:ed)?|cost|betalte|brukte|kj[øo]pte|mottok)\b",
    re.IGNORECASE,
)
_FULL_DATE = re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})")
_DMY_DATE = re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b")
_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
    "mandag": 0,
    "tirsdag": 1,
    "onsdag": 2,
    "torsdag": 3,
    "fredag": 4,
    "lordag": 5,
    "lørdag": 5,
    "sondag": 6,
    "søndag": 6,
}
_WEEKDAY_ALT = "|".join(sorted(_WEEKDAYS, key=len, reverse=True))
_MONTHS = {
    "january": 1,
    "januar": 1,
    "february": 2,
    "februar": 2,
    "march": 3,
    "mars": 3,
    "april": 4,
    "may": 5,
    "mai": 5,
    "june": 6,
    "juni": 6,
    "july": 7,
    "juli": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "oktober": 10,
    "november": 11,
    "december": 12,
    "desember": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))

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
_ACCOUNTS = ["Apple Pay", "Vipps", "Visa", "Mastercard", "bank card", "credit card"]
_REFUND_WORDS = ("refund", "refunded", "refusjon", "refundert")
_INCOME_WORDS = ("salary", "wage", "wages", "paycheck", "income", "received", "lønn", "lonn", "inntekt", "mottok")
_EXPENSE_WORDS = ("spent", "paid", "bought", "purchase", "cost", "brukte", "betalte", "kjøpte", "kjopte")
_EXPENSE_KEYWORDS = [
    ("Groceries", ("grocery", "groceries", "supermarket", "dagligvarer", "matbutikk", "kiwi", "rema", "meny", "coop", "bunnpris")),
    ("Dining", ("lunch", "dinner", "breakfast", "restaurant", "coffee", "cafe", "takeaway", "meal", "lunsj", "middag", "frokost", "kafe", "kafé")),
    ("Transport", ("bus", "metro", "subway", "taxi", "uber", "train", "flight", "fuel", "parking", "buss", "trikk", "drosje", "tog", "parkering")),
    ("Subscriptions", ("netflix", "spotify", "subscription", "membership", "abonnement")),
    ("Housing", ("rent", "mortgage", "husleie")),
    ("Utilities", ("electricity", "utilities", "internet", "strøm", "strom", "nett")),
    ("Health", ("hospital", "pharmacy", "medicine", "doctor", "apotek", "lege")),
    ("Entertainment", ("movie", "cinema", "game", "concert", "kino", "spill", "konsert")),
    ("Travel", ("hotel", "travel", "trip", "hotell", "reise")),
    ("Household", ("household", "detergent", "husholdning")),
    ("Shopping", ("shopping", "clothes", "klar", "klær")),
]
_INCOME_KEYWORDS = [
    ("Salary", ("salary", "wage", "wages", "paycheck", "lønn", "lonn")),
    ("Investment", ("dividend", "interest", "investment", "utbytte", "renter")),
    ("Gift", ("gift", "gave")),
    ("Refund", ("refund", "refunded", "refusjon", "refundert")),
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
    """Today in the configured timezone. Tests may inject now."""
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
    """Map fullwidth digits to ASCII digits so amounts can be read the same way."""
    translation = str.maketrans("０１２３４５６７８９．，", "0123456789.,")
    return text.translate(translation).strip()


def _currency_code(token: str) -> str:
    folded = token.casefold()
    mapping = {
        "norske kroner": "NOK",
        "kroner": "NOK",
        "kr": "NOK",
        "nok": "NOK",
        "svenske kroner": "SEK",
        "sek": "SEK",
        "danske kroner": "DKK",
        "dkk": "DKK",
        "yuan": "CNY",
        "rmb": "CNY",
        "cny": "CNY",
        "dollar": "USD",
        "dollars": "USD",
        "usd": "USD",
        "$": "USD",
        "euro": "EUR",
        "euros": "EUR",
        "eur": "EUR",
        "€": "EUR",
        "pound": "GBP",
        "pounds": "GBP",
        "gbp": "GBP",
        "£": "GBP",
    }
    return mapping[folded]


def _decimal_from_arabic(token: str) -> Decimal:
    """Convert one matched number. Thousands commas are removed only from this token."""
    return Decimal(token.replace(",", ""))


def _number_near_currency(text: str, start: int, end: int, token: str) -> Decimal | None:
    """Read the number before a currency word, or after a currency symbol."""
    if token in {"$", "€", "£"}:
        forward = _ARABIC_AT_START.search(text[end:end + _AMOUNT_LOOKAROUND])
        if forward:
            return _decimal_from_arabic(forward.group(1))
    window = text[max(0, start - _AMOUNT_LOOKAROUND):start]
    arabic = _ARABIC_AT_END.search(window)
    if arabic:
        return _decimal_from_arabic(arabic.group(1))
    return None


def extract_amounts(text: str) -> list[AmountHit]:
    """Amounts with their positions. One hit is kept for each currency match."""
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
    for match in re.finditer(_ARABIC_NUMBER, text):
        hits.append(AmountHit(_decimal_from_arabic(match.group()), None, match.start()))
    return hits


def choose_amount(text: str, hits: list[AmountHit]) -> tuple[AmountHit | None, list[Decimal]]:
    """When several amounts appear, prefer the one next to spent, paid, or received."""
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


def _dated(year: int, month: int, day: int, today: date) -> date | None:
    parsed = _make_date(year, month, day)
    if parsed and parsed > today + timedelta(days=1):
        parsed = _make_date(year - 1, month, day)
    return parsed


def resolve_date(text: str, today: date) -> tuple[date | None, bool, bool, date | None, date | None]:
    """Resolve a relative or absolute date.

    The tuple is the date, whether it was assumed, whether the text stated it,
    and the refund search start and end. Today, yesterday, and i dag are
    calculated from the supplied local date.
    """
    full = _FULL_DATE.search(text)
    if full:
        parsed = _make_date(int(full.group(1)), int(full.group(2)), int(full.group(3)))
        if parsed:
            return parsed, False, True, parsed, parsed
    dmy = _DMY_DATE.search(text)
    if dmy:
        parsed = _make_date(int(dmy.group(3)), int(dmy.group(2)), int(dmy.group(1)))
        if parsed:
            return parsed, False, True, parsed, parsed

    folded = text.casefold()
    day_first = re.search(rf"\b(\d{{1,2}})\.?\s+({_MONTH_ALT})\b(?:\s+(\d{{4}}))?", folded)
    month_first = re.search(rf"\b({_MONTH_ALT})\s+(\d{{1,2}})\b(?:\s+(\d{{4}}))?", folded)
    if day_first:
        year = int(day_first.group(3) or today.year)
        parsed = _dated(year, _MONTHS[day_first.group(2)], int(day_first.group(1)), today)
        if parsed:
            return parsed, False, True, parsed, parsed
    if month_first:
        year = int(month_first.group(3) or today.year)
        parsed = _dated(year, _MONTHS[month_first.group(1)], int(month_first.group(2)), today)
        if parsed:
            return parsed, False, True, parsed, parsed

    if (
        "the day before yesterday" in folded
        or re.search(r"\bi\s*forgårs\b|\bi\s*forgaars\b", folded)
    ):
        target = today - timedelta(days=2)
        return target, False, True, target, target
    if "yesterday" in folded or re.search(r"\bi\s*går\b|\bigår\b|\bi gar\b|\bigar\b", folded):
        target = today - timedelta(days=1)
        return target, False, True, target, target
    if re.search(r"\btoday\b|\bi\s*dag\b|\bidag\b", folded):
        return today, False, True, today, today

    last_weekday = re.search(rf"\b(?:last|forrige)\s+({_WEEKDAY_ALT})\b", folded)
    if last_weekday:
        target = _weekday_in_week(today, 1, _WEEKDAYS[last_weekday.group(1)])
        return target, False, True, target, target
    this_weekday = re.search(rf"\b(?:this|denne)\s+({_WEEKDAY_ALT})\b", folded)
    if this_weekday:
        target = _weekday_in_week(today, 0, _WEEKDAYS[this_weekday.group(1)])
        return target, False, True, target, target
    bare_weekday = re.search(rf"\b({_WEEKDAY_ALT})\b", folded)
    if bare_weekday:
        weekday = _WEEKDAYS[bare_weekday.group(1)]
        target = _weekday_in_week(today, 0, weekday)
        if target > today:
            target = _weekday_in_week(today, 1, weekday)
        return target, False, True, target, target
    if re.search(r"\blast week\b|\bforrige uke\b", folded):
        monday = today - timedelta(days=today.weekday())
        start = monday - timedelta(days=7)
        end = monday - timedelta(days=1)
        return None, False, False, start, end
    return today, True, False, None, None


def _weekday_in_week(today: date, weeks_back: int, weekday: int) -> date:
    monday = today - timedelta(days=today.weekday())
    return monday - timedelta(days=7 * weeks_back) + timedelta(days=weekday)


def _contains_keyword(folded: str, keyword: str) -> bool:
    return re.search(rf"\b{re.escape(keyword.casefold())}\b", folded) is not None


def detect_type(text: str) -> tuple[str | None, bool]:
    """Income, expense, or refund. Refund words win over received."""
    folded = text.casefold()
    refund = any(_contains_keyword(folded, word) for word in _REFUND_WORDS)
    income = any(_contains_keyword(folded, word) for word in _INCOME_WORDS)
    expense = any(_contains_keyword(folded, word) for word in _EXPENSE_WORDS)
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
    """A known merchant, or a Latin name after at, på, or hos."""
    for pattern, canonical in _KNOWN_MERCHANTS:
        if pattern.search(text):
            return canonical
    match = re.search(
        r"(?:at|på|paa|hos)\s+([A-Z][A-Za-z0-9&.'’\-]*(?:\s+[A-Z0-9][A-Za-z0-9&.'’\-]*)?)",
        text,
    )
    if match:
        return " ".join(match.group(1).split())
    return None


def extract_account(text: str) -> str | None:
    """A payment account that is written in the text."""
    folded = text.casefold()
    for name in _ACCOUNTS:
        if name.casefold() in folded:
            return name
    return None


def infer_category(text: str, transaction_type: str | None, merchant: str | None) -> str:
    """Suggest a category from the merchant and keywords. Otherwise use Other."""
    if transaction_type == "refund":
        return "Refund"
    if merchant in MERCHANT_CATEGORIES and transaction_type != "income":
        return MERCHANT_CATEGORIES[merchant]
    folded = text.casefold()
    if transaction_type == "income":
        for category, keywords in _INCOME_KEYWORDS:
            if any(_contains_keyword(folded, keyword) for keyword in keywords):
                return category
        return "Other income"
    for category, keywords in _EXPENSE_KEYWORDS:
        if any(_contains_keyword(folded, keyword) for keyword in keywords):
            return category
    return "Other"


def parse_with_rules(text: str, *, today: date, default_currency: str) -> TransactionDraft:
    """Parse one sentence with rules. A missing amount is not invented."""
    original = text.strip()
    cleaned = normalize_text(original)
    warnings: list[str] = []
    if not cleaned:
        return TransactionDraft(
            needs_clarification=True,
            clarification_question="Enter a transaction.",
            raw_text=text,
            parser="rules",
        )
    transaction_type, type_confident = detect_type(cleaned)
    if transaction_type is None:
        transaction_type = "expense"
        warnings.append("Could not tell income from expense. Filled in as an expense; please change it before saving.")
    hits = extract_amounts(cleaned)
    chosen, distinct = choose_amount(cleaned, hits)
    amount = chosen.amount if chosen else None
    currency = chosen.currency if chosen and chosen.currency else None
    currency_assumed = False
    possible = distinct if amount is None and len(distinct) > 1 else []
    if amount is None and len(distinct) > 1:
        warnings.append("Several amounts found. Choose the one that actually happened.")
    if amount is not None and currency is None:
        currency = default_currency.upper()
        currency_assumed = True
        warnings.append(f"No currency found. Using the default currency {currency} for now.")
    if amount is None and not possible:
        warnings.append("No amount found. Add one before saving.")

    resolved, assumed, explicit, search_start, search_end = resolve_date(cleaned, today)
    if transaction_type == "refund" and resolved is None and search_start and search_end:
        resolved = today
        assumed = True
        warnings.append("The refund is dated today for now. The original expense will be searched in last week.")
    elif resolved is None:
        warnings.append("Please add a specific date.")
    elif assumed:
        warnings.append("No date found. Using today for now; please confirm.")

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
    """Ask for whatever key field is still missing."""
    if draft.amount is None:
        draft.needs_clarification = True
        draft.clarification_question = (
            "Choose the correct amount." if draft.possible_amounts else "Add the amount."
        )
        return draft
    if draft.transaction_date is None:
        draft.needs_clarification = True
        draft.clarification_question = "Please add a specific date."
        return draft
    if draft.date_assumed or not draft.type_confident:
        draft.needs_clarification = True
        draft.clarification_question = (
            "Please confirm the date." if draft.date_assumed else "Please confirm whether this is income, an expense, or a refund."
        )
        return draft
    draft.needs_clarification = False
    draft.clarification_question = None
    return draft


def extract_json_object(text: str) -> dict[str, Any]:
    """Take the first JSON object from model output, including a Markdown fence."""
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
    raise OllamaError("The model did not return JSON that could be parsed.")


def amount_grounded(amount: Decimal, text: str) -> bool:
    """The amount must appear as digits in the original text."""
    normalized = text.replace(",", "")
    if amount == amount.to_integral_value():
        integer_text = str(int(amount))
        if integer_text in normalized:
            return True
    plain = format(amount, "f")
    return plain in normalized


def phrase_in_text(text: str, phrase: str | None) -> bool:
    """Merchant and account must be a case-insensitive substring of the text."""
    if not phrase:
        return False
    return phrase.casefold() in text.casefold()


def apply_merchant_memory(draft: TransactionDraft, memories: dict[str, str]) -> TransactionDraft:
    """Use a previously confirmed merchant category. Keep the suggestion when there is no memory."""
    if draft.transaction_type != "expense" or not draft.merchant:
        return draft
    learned = memories.get(merchant_key(draft.merchant))
    if learned not in EXPENSE_CATEGORIES:
        return draft
    if draft.category != learned:
        draft.category = learned
        note = "Using the category you confirmed earlier for this merchant."
        if note not in draft.warnings:
            draft.warnings.append(note)
    return draft


def categories_for_type(transaction_type: str | None) -> list[str]:
    """Category names allowed for a transaction type."""
    if transaction_type == "expense":
        return list(EXPENSE_CATEGORIES)
    return list(INCOME_CATEGORIES)


def merge_llm_payload(
    rules: TransactionDraft,
    payload: LLMTransactionPayload,
    source_text: str,
) -> TransactionDraft:
    """Merge the model into the rules. A rule amount, date, or type is not overwritten."""
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
                if "No amount found" not in item and "Several amounts" not in item
            ]
        else:
            merged.warnings.append("The amount from the model is not in the original text, so it was ignored.")
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
    generic = {"Other", "Other income", "Refund"}
    if payload.category in allowed and merged.category in generic:
        merged.category = payload.category
    if merged.transaction_type == "refund":
        merged.category = "Refund"
    if merged.category not in categories_for_type(merged.transaction_type):
        merged.category = {
            "income": "Other income",
            "refund": "Refund",
        }.get(merged.transaction_type or "", "Other")
    if not merged.merchant and phrase_in_text(source_text, payload.merchant):
        merged.merchant = payload.merchant.strip() if payload.merchant else None
        if merged.category in generic and merged.merchant in MERCHANT_CATEGORIES:
            merged.category = MERCHANT_CATEGORIES[merged.merchant]
    elif payload.merchant and not phrase_in_text(source_text, payload.merchant):
        merged.warnings.append("The merchant from the model is not in the original text, so it was ignored.")
    if not merged.account and phrase_in_text(source_text, payload.account):
        merged.account = payload.account.strip() if payload.account else None
    return _mark_clarification(merged)


def build_llm_messages(text: str, today: date, default_currency: str) -> list[dict[str, str]]:
    """A strict prompt. Missing fields must be null."""
    system = (
        "You extract one ledger entry from the user's text. Return JSON only. "
        "Do not invent an amount, merchant, account, or item. "
        "Fields that are not in the text must be null. Amounts are strings. Dates are YYYY-MM-DD. "
        f"Today is {today.isoformat()}. The default currency is {default_currency}; "
        "set currency only when the text names one. "
        f"Expense categories: {', '.join(EXPENSE_CATEGORIES)}. "
        f"Income categories: {', '.join(INCOME_CATEGORIES)}. "
        "A refund uses transaction_type refund and category Refund. "
        "If you are unsure, set needs_clarification to true and write the question in clarification_question."
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
    """Call the local Ollama chat API."""
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
        raise OllamaError("The local model timed out. Try again, or use a smaller model.") from exc
    except httpx.HTTPError as exc:
        raise OllamaError(
            f"Could not connect to local Ollama ({base}). Check that Ollama is installed and running."
        ) from exc
    if response.status_code >= 400:
        detail = response.text.strip()[:300]
        raise OllamaHTTPError(response.status_code, f"Ollama returned error {response.status_code}: {detail}")
    try:
        body = response.json()
    except json.JSONDecodeError as exc:
        raise OllamaError("Ollama did not return JSON.") from exc
    message = body.get("message", {})
    content = message.get("content")
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        return extract_json_object(content)
    raise OllamaError("Ollama returned no message content.")


def list_installed_models(host: str, timeout: float = 3) -> list[str]:
    """Names of the Ollama models installed on this computer."""
    base = validate_ollama_host(host)
    try:
        response = httpx.get(f"{base}/api/tags", timeout=timeout)
    except httpx.HTTPError as exc:
        raise OllamaError(f"Could not connect to local Ollama ({base}). Check that the service is running.") from exc
    if response.status_code >= 400:
        raise OllamaHTTPError(response.status_code, f"Could not read the model list: {response.status_code}")
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
    """Parse natural language. If the model fails, the rule result is kept and shown."""
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
        rules.warnings.insert(0, "The model output failed validation. Using the rule-based parse instead.")
        return rules
    return merge_llm_payload(rules, payload, normalize_text(rules.raw_text))
