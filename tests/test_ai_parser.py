"""Parser tests for text, receipts, and speech. They do not call Ollama or download a speech model."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_parser import (
    LLMTransactionPayload,
    OllamaError,
    TransactionDraft,
    extract_json_object,
    parse_transaction_text,
    parse_with_rules,
    resolve_date,
)
from receipt_parser import ReceiptError, ReceiptPayload, interpret_receipt, validate_image_bytes
from voice_parser import VoiceError, transcribe_and_parse, transcribe_audio, validate_audio_name


TODAY = date(2026, 11, 2)


def test_ascii_thousands_separators_stay_one_amount() -> None:
    """An ASCII comma is a thousands separator, not a break in the number."""
    groceries = parse_with_rules("groceries cost 1,200 yuan", today=TODAY, default_currency="NOK")
    assert groceries.amount == Decimal("1200")
    assert groceries.currency == "CNY"

    paid = parse_with_rules("paid $1,000", today=TODAY, default_currency="NOK")
    assert paid.amount == Decimal("1000")
    assert paid.currency == "USD"

    cents = parse_with_rules("groceries cost 1,200.50 yuan", today=TODAY, default_currency="NOK")
    assert cents.amount == Decimal("1200.50")
    assert cents.currency == "CNY"

    long_amount = parse_with_rules("transferred 12,345,678,901.25 yuan", today=TODAY, default_currency="NOK")
    assert long_amount.amount == Decimal("12345678901.25")

    bare = parse_with_rules("spent 1,200", today=TODAY, default_currency="NOK")
    assert bare.amount == Decimal("1200")
    assert bare.currency == "NOK"

    not_thousands = parse_with_rules("spent 1,2 kroner", today=TODAY, default_currency="NOK")
    assert not_thousands.amount == Decimal("2")

    plain = parse_with_rules("spent 120 kroner", today=TODAY, default_currency="CNY")
    assert plain.amount == Decimal("120")
    assert plain.currency == "NOK"


def test_relative_dates_use_supplied_today() -> None:
    assert resolve_date("lunch today", TODAY)[0] == TODAY
    assert resolve_date("bus yesterday", TODAY)[0] == date(2026, 11, 1)
    assert resolve_date("groceries the day before yesterday", TODAY)[0] == date(2026, 10, 31)
    assert resolve_date("lunch last Wednesday", TODAY)[0] == date(2026, 10, 28)
    assert resolve_date("i dag lunsj", TODAY)[0] == TODAY
    assert resolve_date("i gar buss", TODAY)[0] == date(2026, 11, 1)
    start = resolve_date("last week 200 kroner", TODAY)[3]
    end = resolve_date("last week 200 kroner", TODAY)[4]
    assert start == date(2026, 10, 26)
    assert end == date(2026, 11, 1)


def test_examples_from_requirements() -> None:
    lunch = parse_with_rules("Lunch cost 120 yuan today.", today=TODAY, default_currency="NOK")
    assert lunch.amount == Decimal("120")
    assert lunch.currency == "CNY"
    assert lunch.category == "Dining"
    assert lunch.transaction_type == "expense"
    assert lunch.merchant is None
    assert lunch.transaction_date == TODAY
    assert lunch.needs_clarification is False

    kiwi = parse_with_rules("Spent 350 NOK at Kiwi today.", today=TODAY, default_currency="CNY")
    assert kiwi.amount == Decimal("350")
    assert kiwi.currency == "NOK"
    assert kiwi.merchant == "Kiwi"
    assert kiwi.category == "Groceries"
    assert kiwi.transaction_type == "expense"
    assert kiwi.transaction_date == TODAY
    assert kiwi.needs_clarification is False

    salary = parse_with_rules("Received salary 45000 NOK today.", today=TODAY, default_currency="CNY")
    assert salary.amount == Decimal("45000")
    assert salary.currency == "NOK"
    assert salary.transaction_type == "income"
    assert salary.category == "Salary"
    assert salary.merchant is None

    bus = parse_with_rules("Yesterday the bus cost 50 kroner.", today=TODAY, default_currency="CNY")
    assert bus.amount == Decimal("50")
    assert bus.currency == "NOK"
    assert bus.transaction_date == date(2026, 11, 1)
    assert bus.category == "Transport"
    assert bus.transaction_type == "expense"

    norwegian = parse_with_rules("I dag brukte jeg 80 kroner pa Kiwi.", today=TODAY, default_currency="CNY")
    assert norwegian.amount == Decimal("80")
    assert norwegian.currency == "NOK"
    assert norwegian.merchant == "Kiwi"
    assert norwegian.category == "Groceries"


def test_missing_amount_is_not_invented() -> None:
    missing = parse_with_rules("had lunch today", today=TODAY, default_currency="NOK")
    assert missing.amount is None
    assert missing.needs_clarification is True
    assert missing.merchant is None


def test_refund_phrases_do_not_invent_merchant() -> None:
    yesterday = parse_with_rules("Yesterday the 50 yuan purchase was refunded.", today=TODAY, default_currency="NOK")
    assert yesterday.transaction_type == "refund"
    assert yesterday.amount == Decimal("50")
    assert yesterday.currency == "CNY"
    assert yesterday.category == "Refund"
    assert yesterday.transaction_date == date(2026, 11, 1)
    assert yesterday.merchant is None

    netflix = parse_with_rules("Netflix refunded 99 NOK.", today=TODAY, default_currency="CNY")
    assert netflix.transaction_type == "refund"
    assert netflix.amount == Decimal("99")
    assert netflix.currency == "NOK"
    assert netflix.merchant == "Netflix"

    norwegian = parse_with_rules("i gar brukte jeg 80 kroner", today=TODAY, default_currency="CNY")
    assert norwegian.amount == Decimal("80")
    assert norwegian.currency == "NOK"
    assert norwegian.transaction_date == date(2026, 11, 1)
    assert norwegian.transaction_type == "expense"


def test_model_cannot_override_grounded_facts() -> None:
    def requester(*_args, **_kwargs):
        return {
            "transaction_date": "1999-01-01",
            "amount": "999",
            "currency": "USD",
            "transaction_type": "income",
            "category": "Salary",
            "description": "invented description",
            "merchant": "Invented Shop",
            "account": "Invented Account",
            "needs_clarification": False,
            "clarification_question": None,
        }

    draft = parse_transaction_text(
        "Lunch cost 120 yuan today.",
        today=TODAY,
        default_currency="NOK",
        requester=requester,
    )
    assert draft.amount == Decimal("120")
    assert draft.currency == "CNY"
    assert draft.transaction_type == "expense"
    assert draft.category == "Dining"
    assert draft.merchant is None
    assert draft.account is None
    assert draft.transaction_date == TODAY
    assert draft.description == "Lunch cost 120 yuan today."
    assert any("merchant" in warning.casefold() for warning in draft.warnings)


def test_ollama_failure_keeps_rule_parse() -> None:
    def requester(*_args, **_kwargs):
        raise OllamaError("Could not connect to local Ollama (http://127.0.0.1:9).")

    draft = parse_transaction_text(
        "Lunch cost 20 yuan today.",
        today=TODAY,
        default_currency="NOK",
        requester=requester,
    )
    assert draft.amount == Decimal("20")
    assert draft.currency == "CNY"
    assert any("Ollama" in warning for warning in draft.warnings)


def test_invalid_model_payload_is_rejected() -> None:
    with pytest.raises(ValidationError):
        LLMTransactionPayload.model_validate({"transaction_type": "transfer", "amount": "12"})

    def requester(*_args, **_kwargs):
        return {"transaction_type": "transfer", "amount": "not a number"}

    draft = parse_transaction_text(
        "Lunch cost 20 yuan today.",
        today=TODAY,
        default_currency="CNY",
        requester=requester,
    )
    assert draft.amount == Decimal("20")
    assert any("validation" in warning.casefold() for warning in draft.warnings)


def test_json_fence_and_draft_roundtrip() -> None:
    payload = extract_json_object('note\n```json\n{"amount": "12.50", "merchant": null}\n```')
    assert payload["amount"] == "12.50"
    draft = parse_with_rules("Lunch cost 20 yuan today.", today=TODAY, default_currency="CNY")
    restored = TransactionDraft.model_validate(draft.model_dump(mode="json"))
    assert restored.amount == draft.amount
    assert restored.transaction_date == draft.transaction_date
    assert restored.category == draft.category


def test_receipt_validation_and_amount_rules() -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"data"
    jpeg = b"\xff\xd8\xff" + b"data"
    assert validate_image_bytes(png, "receipt.png") == "png"
    assert validate_image_bytes(jpeg, "receipt.jpg") == "jpeg"
    with pytest.raises(ReceiptError):
        validate_image_bytes(png, "receipt.jpg")
    with pytest.raises(ReceiptError):
        validate_image_bytes(b"hello", "receipt.png")

    clear = interpret_receipt(
        ReceiptPayload(
            merchant="Kiwi",
            transaction_date="2026-11-02",
            total_amount="350.00",
            currency="NOK",
            items=["milk"],
            suggested_category="Groceries",
            subtotal_amount="300.00",
        ),
        today=TODAY,
        default_currency="CNY",
    )
    assert clear.amount == Decimal("350.00")
    assert clear.currency == "NOK"
    assert clear.merchant == "Kiwi"
    assert clear.category == "Groceries"
    assert any("subtotal" in warning.casefold() for warning in clear.warnings)
    assert clear.needs_clarification is False

    ambiguous = interpret_receipt(
        ReceiptPayload(
            total_amount="120",
            possible_amounts=["100", "120"],
            currency="CNY",
            transaction_date="2026-11-02",
        ),
        today=TODAY,
        default_currency="NOK",
    )
    assert ambiguous.amount is None
    assert ambiguous.needs_clarification is True
    assert Decimal("100") in ambiguous.possible_amounts
    assert Decimal("120") in ambiguous.possible_amounts

    subtotal_only = interpret_receipt(
        ReceiptPayload(subtotal_amount="80", currency="CNY", transaction_date="2026-11-02"),
        today=TODAY,
        default_currency="CNY",
    )
    assert subtotal_only.amount is None
    assert any("subtotal" in warning.casefold() for warning in subtotal_only.warnings)


def test_voice_pipeline_reuses_text_parser(tmp_path: Path) -> None:
    with pytest.raises(VoiceError):
        validate_audio_name("note.txt")

    audio_path = tmp_path / "lunch.wav"
    audio_path.write_bytes(b"RIFFfake")

    class Segment:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeModel:
        def transcribe(self, _path: str, language: str | None = None):
            return [Segment("Spent 300 kroner at the supermarket today.")], None

    def loader(_size: str, device: str, _compute: str):
        if device == "cuda":
            raise RuntimeError("No NVIDIA GPU is available")
        return FakeModel()

    transcript, draft = transcribe_and_parse(
        audio_path,
        today=TODAY,
        default_currency="CNY",
        model_loader=loader,
        use_ollama=False,
    )
    assert "300 kroner" in transcript
    assert draft.amount == Decimal("300")
    assert draft.currency == "NOK"
    assert draft.source == "voice"
    assert draft.category == "Groceries"

    fallback = transcribe_audio(audio_path, device="cuda", model_loader=loader)
    assert "300 kroner" in fallback
