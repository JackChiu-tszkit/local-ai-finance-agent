"""自然语言、收据和语音解析测试。这些测试不连接 Ollama，也不下载语音模型。"""

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
    parse_chinese_integer,
    parse_transaction_text,
    parse_with_rules,
    resolve_date,
)
from receipt_parser import ReceiptError, ReceiptPayload, interpret_receipt, validate_image_bytes
from voice_parser import VoiceError, transcribe_and_parse, transcribe_audio, validate_audio_name


TODAY = date(2026, 11, 2)


def test_ascii_thousands_separators_stay_one_amount() -> None:
    """英文逗号按千分位读取，不能只留下逗号一侧的数字。"""
    groceries = parse_with_rules("买菜花了1,200元", today=TODAY, default_currency="NOK")
    assert groceries.amount == Decimal("1200")
    assert groceries.currency == "CNY"

    fullwidth = parse_with_rules("买菜花了1，200元", today=TODAY, default_currency="NOK")
    assert fullwidth.amount == Decimal("1200")
    assert fullwidth.currency == "CNY"

    paid = parse_with_rules("paid $1,000", today=TODAY, default_currency="NOK")
    assert paid.amount == Decimal("1000")
    assert paid.currency == "USD"

    cents = parse_with_rules("买菜花了1,200.50元", today=TODAY, default_currency="NOK")
    assert cents.amount == Decimal("1200.50")
    assert cents.currency == "CNY"

    long_amount = parse_with_rules("转账12,345,678,901.25元", today=TODAY, default_currency="NOK")
    assert long_amount.amount == Decimal("12345678901.25")

    bare = parse_with_rules("花了1,200", today=TODAY, default_currency="NOK")
    assert bare.amount == Decimal("1200")

    not_thousands = parse_with_rules("花了1,2元", today=TODAY, default_currency="NOK")
    assert not_thousands.amount == Decimal("2")

    plain = parse_with_rules("花了120元", today=TODAY, default_currency="NOK")
    assert plain.amount == Decimal("120")


def test_chinese_integers() -> None:
    assert parse_chinese_integer("三百") == 300
    assert parse_chinese_integer("三百五十") == 350
    assert parse_chinese_integer("五十") == 50
    assert parse_chinese_integer("一百二十") == 120
    assert parse_chinese_integer("四万五") == 45000
    assert parse_chinese_integer("三百零五") == 305
    assert parse_chinese_integer("十二") == 12


def test_relative_dates_use_supplied_today() -> None:
    assert resolve_date("今天吃饭", TODAY)[0] == TODAY
    assert resolve_date("昨天坐公交", TODAY)[0] == date(2026, 11, 1)
    assert resolve_date("前天买菜", TODAY)[0] == date(2026, 10, 31)
    assert resolve_date("上周三吃饭", TODAY)[0] == date(2026, 10, 28)
    start = resolve_date("上周200克朗", TODAY)[3]
    end = resolve_date("上周200克朗", TODAY)[4]
    assert start == date(2026, 10, 26)
    assert end == date(2026, 11, 1)


def test_examples_from_requirements() -> None:
    hotpot = parse_with_rules("中午吃了火锅，花了120元。", today=TODAY, default_currency="NOK")
    assert hotpot.amount == Decimal("120")
    assert hotpot.currency == "CNY"
    assert hotpot.category == "餐饮"
    assert hotpot.transaction_type == "expense"
    assert hotpot.merchant is None
    assert hotpot.transaction_date == TODAY
    assert hotpot.needs_clarification is True

    kiwi = parse_with_rules("今天在 Kiwi 买菜花了350 NOK。", today=TODAY, default_currency="CNY")
    assert kiwi.amount == Decimal("350")
    assert kiwi.currency == "NOK"
    assert kiwi.merchant == "Kiwi"
    assert kiwi.category == "食品杂货"
    assert kiwi.transaction_type == "expense"
    assert kiwi.transaction_date == TODAY
    assert kiwi.needs_clarification is False

    salary = parse_with_rules("今天收到工资45000 NOK。", today=TODAY, default_currency="CNY")
    assert salary.amount == Decimal("45000")
    assert salary.currency == "NOK"
    assert salary.transaction_type == "income"
    assert salary.category == "工资"
    assert salary.merchant is None

    bus = parse_with_rules("昨天坐公交花了50克朗。", today=TODAY, default_currency="CNY")
    assert bus.amount == Decimal("50")
    assert bus.currency == "NOK"
    assert bus.transaction_date == date(2026, 11, 1)
    assert bus.category == "交通"
    assert bus.transaction_type == "expense"


def test_chinese_amount_and_missing_amount() -> None:
    spoken = parse_with_rules("今天去超市买东西花了三百克朗。", today=TODAY, default_currency="CNY")
    assert spoken.amount == Decimal("300")
    assert spoken.currency == "NOK"
    assert spoken.category == "食品杂货"
    missing = parse_with_rules("今天吃了午饭", today=TODAY, default_currency="NOK")
    assert missing.amount is None
    assert missing.needs_clarification is True
    assert missing.merchant is None


def test_refund_phrases_do_not_invent_merchant() -> None:
    yesterday = parse_with_rules("昨天那笔50元的消费退款了。", today=TODAY, default_currency="NOK")
    assert yesterday.transaction_type == "refund"
    assert yesterday.amount == Decimal("50")
    assert yesterday.currency == "CNY"
    assert yesterday.category == "退款"
    assert yesterday.transaction_date == date(2026, 11, 1)
    assert yesterday.merchant is None

    netflix = parse_with_rules("Netflix 退回了99 NOK。", today=TODAY, default_currency="CNY")
    assert netflix.transaction_type == "refund"
    assert netflix.amount == Decimal("99")
    assert netflix.currency == "NOK"
    assert netflix.merchant == "Netflix"

    norwegian = parse_with_rules("i går brukte jeg 80 kroner", today=TODAY, default_currency="CNY")
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
            "category": "工资",
            "description": "编造的描述",
            "merchant": "幻想商户",
            "account": "幻想账户",
            "needs_clarification": False,
            "clarification_question": None,
        }

    draft = parse_transaction_text(
        "中午吃了火锅，花了120元。",
        today=TODAY,
        default_currency="NOK",
        requester=requester,
    )
    assert draft.amount == Decimal("120")
    assert draft.currency == "CNY"
    assert draft.transaction_type == "expense"
    assert draft.category == "餐饮"
    assert draft.merchant is None
    assert draft.account is None
    assert draft.transaction_date == TODAY
    assert draft.description == "中午吃了火锅，花了120元。"
    assert any("商户" in warning for warning in draft.warnings)


def test_ollama_failure_keeps_rule_parse() -> None:
    def requester(*_args, **_kwargs):
        raise OllamaError("无法连接本地 Ollama（http://127.0.0.1:9）。请确认 Ollama 已安装并正在运行。")

    draft = parse_transaction_text(
        "今天午餐花了20元。",
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
        return {"transaction_type": "transfer", "amount": "不是数字"}

    draft = parse_transaction_text(
        "今天午餐花了20元。",
        today=TODAY,
        default_currency="CNY",
        requester=requester,
    )
    assert draft.amount == Decimal("20")
    assert any("校验" in warning for warning in draft.warnings)


def test_json_fence_and_draft_roundtrip() -> None:
    payload = extract_json_object('说明文字\n```json\n{"amount": "12.50", "merchant": null}\n```')
    assert payload["amount"] == "12.50"
    draft = parse_with_rules("今天午餐花了20元。", today=TODAY, default_currency="CNY")
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
            items=["牛奶"],
            suggested_category="食品杂货",
            subtotal_amount="300.00",
        ),
        today=TODAY,
        default_currency="CNY",
    )
    assert clear.amount == Decimal("350.00")
    assert clear.currency == "NOK"
    assert clear.merchant == "Kiwi"
    assert clear.category == "食品杂货"
    assert any("小计" in warning for warning in clear.warnings)
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
    assert any("小计" in warning for warning in subtotal_only.warnings)


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
            return [Segment("今天去超市买东西花了三百克朗。")], None

    def loader(_size: str, device: str, _compute: str):
        if device == "cuda":
            raise RuntimeError("没有可用的 NVIDIA GPU")
        return FakeModel()

    transcript, draft = transcribe_and_parse(
        audio_path,
        today=TODAY,
        default_currency="CNY",
        model_loader=loader,
        use_ollama=False,
    )
    assert "三百克朗" in transcript
    assert draft.amount == Decimal("300")
    assert draft.currency == "NOK"
    assert draft.source == "voice"
    assert draft.category == "食品杂货"

    fallback = transcribe_audio(audio_path, device="cuda", model_loader=loader)
    assert "三百克朗" in fallback
