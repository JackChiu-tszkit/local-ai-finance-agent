"""Receipt and screenshot parsing.

The image is sent only to the Ollama address in settings, which defaults to this computer.
The result becomes a draft and is saved only after confirmation.
"""

from __future__ import annotations

import base64
from datetime import date
from decimal import Decimal
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, field_validator

from ai_parser import TransactionDraft, _mark_clarification
from config import EXPENSE_CATEGORIES, Settings


class ReceiptError(ValueError):
    """The image could not be read, or the result cannot be used."""


class ReceiptPayload(BaseModel):
    """Receipt fields from the vision model. Missing values stay empty."""

    model_config = ConfigDict(extra="ignore")

    merchant: str | None = None
    transaction_date: str | None = None
    total_amount: str | None = None
    currency: str | None = None
    items: list[str] = []
    suggested_category: str | None = None
    possible_amounts: list[str] = []
    subtotal_amount: str | None = None
    notes: str | None = None
    needs_clarification: bool = False
    clarification_question: str | None = None

    @field_validator(
        "merchant",
        "transaction_date",
        "total_amount",
        "currency",
        "suggested_category",
        "subtotal_amount",
        "notes",
        "clarification_question",
        mode="before",
    )
    @classmethod
    def empty_to_none(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("items", "possible_amounts", mode="before")
    @classmethod
    def string_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


_MAX_IMAGE_BYTES = 8 * 1024 * 1024
_RECEIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "merchant": {"type": ["string", "null"]},
        "transaction_date": {"type": ["string", "null"]},
        "total_amount": {"type": ["string", "null"]},
        "currency": {"type": ["string", "null"]},
        "items": {"type": "array", "items": {"type": "string"}},
        "suggested_category": {"type": ["string", "null"]},
        "possible_amounts": {"type": "array", "items": {"type": "string"}},
        "subtotal_amount": {"type": ["string", "null"]},
        "notes": {"type": ["string", "null"]},
        "needs_clarification": {"type": "boolean"},
        "clarification_question": {"type": ["string", "null"]},
    },
    "required": [
        "merchant",
        "transaction_date",
        "total_amount",
        "currency",
        "items",
        "suggested_category",
        "possible_amounts",
        "subtotal_amount",
        "notes",
        "needs_clarification",
        "clarification_question",
    ],
}


def validate_image_bytes(data: bytes, filename: str) -> str:
    """Check the extension, file header, and size. Returns png or jpeg."""
    if not data:
        raise ReceiptError("The image file is empty.")
    if len(data) > _MAX_IMAGE_BYTES:
        raise ReceiptError("The image is larger than 8MB. Compress it and try again.")
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    is_png = data.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = data.startswith(b"\xff\xd8\xff")
    if is_png:
        if suffix != "png":
            raise ReceiptError("The file is a PNG, but the extension is not .png.")
        return "png"
    if is_jpeg:
        if suffix not in {"jpg", "jpeg"}:
            raise ReceiptError("The file is a JPEG, but the extension is not .jpg or .jpeg.")
        return "jpeg"
    raise ReceiptError("Only PNG, JPG, and JPEG images are supported.")


def _decimal_or_none(value: str | None) -> Decimal | None:
    if not value:
        return None
    try:
        amount = Decimal(value.replace(",", "").replace(" ", ""))
    except Exception:
        return None
    if amount <= 0:
        return None
    return amount


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def interpret_receipt(
    payload: ReceiptPayload,
    *,
    today: date,
    default_currency: str,
) -> TransactionDraft:
    """Turn receipt JSON into a draft.

    Several possible totals are not chosen automatically.
    A subtotal without an amount paid is not treated as the total.
    """
    warnings: list[str] = []
    total = _decimal_or_none(payload.total_amount)
    subtotal = _decimal_or_none(payload.subtotal_amount)
    possible: list[Decimal] = []
    for raw in payload.possible_amounts:
        amount = _decimal_or_none(raw)
        if amount is not None and amount not in possible:
            possible.append(amount)
    if total is not None and total not in possible:
        possible.insert(0, total)
    if subtotal is not None and total is not None and subtotal != total:
        warnings.append("Ignored the subtotal. The amount is the amount actually paid.")
    if total is None and subtotal is not None:
        warnings.append("Only a subtotal was visible, so it was not used as the amount paid.")

    currency = (payload.currency or "").upper()
    currency_assumed = False
    if len(currency) != 3 or not currency.isalpha():
        currency = default_currency.upper()
        currency_assumed = True
        warnings.append(f"The receipt has no clear currency. Using the default currency {currency} for now.")

    ambiguous = [item for item in possible if item != total] if total is not None else possible
    amount: Decimal | None = total
    possible_for_user: list[Decimal] = []
    if len(possible) > 1:
        amount = None
        possible_for_user = possible
        warnings.append("Several possible totals found. Choose the amount you actually paid.")
    elif amount is None:
        warnings.append("No total paid was found. Enter it manually.")

    tx_date = _parse_iso_date(payload.transaction_date)
    assumed = False
    explicit = tx_date is not None
    if tx_date is None:
        tx_date = today
        assumed = True
        warnings.append("No clear date on the receipt. Using today for now.")
    category = payload.suggested_category if payload.suggested_category in EXPENSE_CATEGORIES else "Other"
    merchant = payload.merchant.strip() if payload.merchant else None
    items = [item[:80] for item in payload.items[:30]]
    description = f"Receipt: {merchant}" if merchant else "Receipt entry"
    if payload.notes:
        warnings.append(f"Note: {payload.notes[:120]}")
    if payload.clarification_question:
        warnings.append(payload.clarification_question)
    draft = TransactionDraft(
        transaction_date=tx_date,
        date_assumed=assumed,
        date_explicit=explicit,
        amount=amount,
        currency=currency,
        currency_assumed=currency_assumed,
        transaction_type="expense",
        type_confident=True,
        category=category,
        description=description,
        merchant=merchant,
        warnings=warnings,
        possible_amounts=possible_for_user,
        items=items,
        source="receipt",
        parser="vision",
        raw_text=description,
    )
    if ambiguous and amount is None:
        draft.possible_amounts = possible_for_user
    return _mark_clarification(draft)


def parse_receipt_bytes(
    data: bytes,
    filename: str,
    *,
    today: date,
    default_currency: str,
    settings: Settings | None = None,
    requester: Callable[..., dict[str, Any]] | None = None,
    timeout: float = 180,
) -> TransactionDraft:
    """Read a receipt with the local vision model. If the model is down, this raises instead of guessing."""
    from ai_parser import request_ollama_chat

    validate_image_bytes(data, filename)
    active = settings or Settings()
    caller = requester or request_ollama_chat
    encoded = base64.b64encode(data).decode("ascii")
    prompt = (
        "Extract only what is visible in this receipt or payment screenshot. "
        "Do not invent a merchant, date, amount, or item. Use null for anything you cannot read. "
        "total_amount must be the final amount paid, not the subtotal. "
        "If several totals are possible, put all of them in possible_amounts and set needs_clarification to true. "
        "Amounts and dates are strings. Dates use YYYY-MM-DD. "
        f"suggested_category must be one of: {', '.join(EXPENSE_CATEGORIES)}."
    )
    messages = [{"role": "user", "content": prompt, "images": [encoded]}]
    raw = caller(
        active.ollama_host,
        active.vision_model,
        messages,
        timeout=timeout,
        schema=_RECEIPT_SCHEMA,
    )
    payload = ReceiptPayload.model_validate(raw)
    return interpret_receipt(payload, today=today, default_currency=default_currency)
