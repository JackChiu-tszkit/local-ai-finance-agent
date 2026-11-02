"""收据和截图解析。

图片只发送到用户配置的 Ollama 地址，默认是本机。
识别结果先变成交易草稿，必须经确认后才会入账。
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
    """图片无法读取，或识别结果不能使用。"""


class ReceiptPayload(BaseModel):
    """视觉模型返回的收据结构。缺失内容必须保持为空，不能补猜。"""

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
    """检查扩展名、文件头和大小。返回 png 或 jpeg。"""
    if not data:
        raise ReceiptError("图片是空文件。")
    if len(data) > _MAX_IMAGE_BYTES:
        raise ReceiptError("图片超过 8MB，请压缩后再试。")
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    is_png = data.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = data.startswith(b"\xff\xd8\xff")
    if is_png:
        if suffix != "png":
            raise ReceiptError("文件内容是 PNG，但扩展名不是 .png。")
        return "png"
    if is_jpeg:
        if suffix not in {"jpg", "jpeg"}:
            raise ReceiptError("文件内容是 JPEG，但扩展名不是 .jpg 或 .jpeg。")
        return "jpeg"
    raise ReceiptError("只支持 PNG、JPG 和 JPEG 图片。")


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
    """把收据 JSON 转成待确认草稿。

    有多个可能的总额时不自动挑选。只有小计、没有实付额时也不把小计当成总额。
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
        warnings.append("已忽略小计，金额以实付总额为准。")
    if total is None and subtotal is not None:
        warnings.append("只看到小计，没有把小计当作实付金额。")

    currency = (payload.currency or "").upper()
    currency_assumed = False
    if len(currency) != 3 or not currency.isalpha():
        currency = default_currency.upper()
        currency_assumed = True
        warnings.append(f"收据没有明确币种，已暂用默认币种 {currency}。")

    ambiguous = [item for item in possible if item != total] if total is not None else possible
    amount: Decimal | None = total
    possible_for_user: list[Decimal] = []
    if len(possible) > 1:
        amount = None
        possible_for_user = possible
        warnings.append("识别到多个可能的总额，请选择实际支付金额。")
    elif amount is None:
        warnings.append("没有识别到实付总额，请手动填写。")

    tx_date = _parse_iso_date(payload.transaction_date)
    assumed = False
    explicit = tx_date is not None
    if tx_date is None:
        tx_date = today
        assumed = True
        warnings.append("收据没有清晰日期，已暂按今天填写。")
    category = payload.suggested_category if payload.suggested_category in EXPENSE_CATEGORIES else "其他"
    merchant = payload.merchant.strip() if payload.merchant else None
    items = [item[:80] for item in payload.items[:30]]
    description = f"收据：{merchant}" if merchant else "收据记账"
    if payload.notes:
        warnings.append(f"识别备注：{payload.notes[:120]}")
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
    # 多个总额时 _mark_clarification 会要求用户选择；这里保留该行为。
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
    """调用本地视觉模型读取收据。模型不可用时直接报错，不编造内容。"""
    from ai_parser import request_ollama_chat

    validate_image_bytes(data, filename)
    active = settings or Settings()
    caller = requester or request_ollama_chat
    encoded = base64.b64encode(data).decode("ascii")
    prompt = (
        "请只根据图片中看得见的内容提取收据或交易截图。"
        "不要编造商户、日期、金额或商品。看不清的字段返回 null。"
        "total_amount 必须是最终实付金额，不要填写小计。"
        "如果有多个可能的总额，全部放入 possible_amounts，并把 needs_clarification 设为 true。"
        "金额和日期使用字符串，日期格式 YYYY-MM-DD。"
        f"支出分类只能从以下列表选择：{'、'.join(EXPENSE_CATEGORIES)}。"
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
