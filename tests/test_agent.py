"""Assistant tests. Questions and category previews do not call Ollama or write the ledger."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

from agent import handle_user_message
from ai_parser import parse_with_rules
from finance_service import FinanceService


TODAY = date(2026, 11, 3)


def make_service(tmp_path: Path) -> FinanceService:
    return FinanceService(tmp_path / "finance.db")


def add(
    service: FinanceService,
    *,
    amount: str,
    currency: str = "CNY",
    category_name: str = "Dining",
    description: str = "lunch",
    merchant: str | None = None,
    transaction_date: date = date(2026, 11, 2),
    transaction_type: str = "expense",
    original_transaction_id: int | None = None,
) -> int:
    return service.create_transaction(
        transaction_date=transaction_date,
        amount=Decimal(amount),
        currency=currency,
        transaction_type=transaction_type,
        category_name=category_name,
        description=description,
        merchant=merchant,
        account_name=None,
        original_transaction_id=original_transaction_id,
        source="manual",
    )


def ask(service: FinanceService, text: str, pending=None):
    return handle_user_message(
        text,
        today=TODAY,
        default_currency="NOK",
        service=service,
        pending=pending,
        memories=service.merchant_memories(),
    )


def test_follow_up_updates_pending_draft_without_saving(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    pending = parse_with_rules("Lunch cost 80 yuan today.", today=TODAY, default_currency="NOK")
    amended = ask(service, "change it to last Wednesday", pending)
    assert amended.intent == "amend"
    assert amended.amended_draft is not None
    assert amended.amended_draft.transaction_date == date(2026, 10, 28)
    assert amended.amended_draft.amount == Decimal("80")
    assert not any("Using today" in item for item in amended.amended_draft.warnings)
    assert amended.amended_draft.raw_text == "Lunch cost 80 yuan today."
    assert service.list_transaction_rows() == []

    merchant = ask(service, "merchant is Kiwi", amended.amended_draft)
    assert merchant.amended_draft is not None
    assert merchant.amended_draft.merchant == "Kiwi"
    assert merchant.amended_draft.category == "Groceries"

    amount = ask(service, "change the amount to 120", merchant.amended_draft)
    assert amount.amended_draft is not None
    assert amount.amended_draft.amount == Decimal("120")
    assert amount.amended_draft.transaction_date == date(2026, 10, 28)
    assert service.list_transaction_rows() == []


def test_one_sentence_becomes_two_drafts(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    result = ask(service, "Lunch cost 80 yuan, the metro cost 30 yuan")
    assert result.intent == "draft"
    assert [item.amount for item in result.new_drafts] == [Decimal("80"), Decimal("30")]
    assert [item.category for item in result.new_drafts] == ["Dining", "Transport"]
    assert service.list_transaction_rows() == []


def test_category_question_uses_database_numbers(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    add(service, amount="80", currency="CNY", category_name="Dining", description="hotpot")
    before = len(service.list_transaction_rows())
    result = ask(service, "how much did Dining cost this month")
    assert result.intent == "question"
    assert result.new_drafts == []
    assert "80.00" in result.reply
    assert "CNY" in result.reply
    assert "Dining" in result.reply
    assert len(service.list_transaction_rows()) == before


def test_refund_remaining_comes_from_linked_rows(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    original = add(
        service,
        amount="99",
        currency="NOK",
        category_name="Subscriptions",
        description="membership",
        merchant="Netflix",
        transaction_date=date(2026, 11, 1),
    )
    add(
        service,
        amount="40",
        currency="NOK",
        category_name="Refund",
        description="partial refund",
        merchant="Netflix",
        transaction_type="refund",
        original_transaction_id=original,
        transaction_date=date(2026, 11, 2),
    )
    result = ask(service, "how much of Netflix can still be refunded")
    assert "59.00" in result.reply
    assert "NOK" in result.reply


def test_confirmed_merchant_category_is_reused(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.remember_merchant("Bakery", "Dining", "expense")
    service.remember_merchant("Kiwi", "Dining", "expense")
    bakery = ask(service, "spent 10 yuan at Bakery")
    assert bakery.new_drafts[0].merchant == "Bakery"
    assert bakery.new_drafts[0].category == "Dining"
    kiwi = ask(service, "spent 10 NOK at Kiwi today")
    assert kiwi.new_drafts[0].category == "Dining"


def test_batch_preview_does_not_write_until_confirmed(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    first = add(
        service,
        amount="30",
        currency="NOK",
        category_name="Other",
        description="taxi to work",
        transaction_date=date(2026, 10, 4),
    )
    second = add(
        service,
        amount="40",
        currency="NOK",
        category_name="Other",
        description="evening taxi",
        transaction_date=date(2026, 10, 18),
    )
    meal = add(
        service,
        amount="90",
        currency="NOK",
        category_name="Dining",
        description="dinner",
        transaction_date=date(2026, 10, 18),
    )
    result = ask(service, "move all last month taxi to Transport")
    assert result.proposal is not None
    assert result.proposal.transaction_ids == [second, first]
    categories = {row["id"]: row["category_name"] for row in service.list_transaction_rows()}
    assert categories[first] == "Other"
    assert categories[second] == "Other"
    changed = service.recategorize(result.proposal.transaction_ids, "Transport")
    assert changed == 2
    categories = {row["id"]: row["category_name"] for row in service.list_transaction_rows()}
    assert categories[first] == "Transport"
    assert categories[second] == "Transport"
    assert categories[meal] == "Dining"
    assert service.merchant_memories() == {}


def test_spending_insight_uses_personal_history(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    for month in (8, 9, 10):
        add(
            service,
            amount="100",
            currency="NOK",
            category_name="Dining",
            description="meal",
            transaction_date=date(2026, month, 10),
        )
    add(
        service,
        amount="200",
        currency="NOK",
        category_name="Dining",
        description="dinner out",
        transaction_date=date(2026, 11, 1),
    )
    notes = service.spending_insights(TODAY)
    assert any("Dining" in note and "above" in note for note in notes)

    quiet = make_service(tmp_path / "quiet")
    for month in (8, 9, 10):
        add(
            quiet,
            amount="100",
            currency="NOK",
            category_name="Dining",
            description="meal",
            transaction_date=date(2026, month, 10),
        )
    add(
        quiet,
        amount="120",
        currency="NOK",
        category_name="Dining",
        description="lunch",
        transaction_date=date(2026, 11, 1),
    )
    assert not any("above" in note for note in quiet.spending_insights(TODAY))


def test_recurring_merchant_and_other_bucket(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    for month in (8, 9, 11):
        add(
            service,
            amount="30",
            currency="NOK",
            category_name="Transport",
            description="taxi",
            merchant="Uber",
            transaction_date=date(2026, month, 6),
        )
    notes = " ".join(service.spending_insights(TODAY))
    assert "Uber" in notes
    assert "nothing is recorded automatically" in notes

    other = make_service(tmp_path / "other")
    for day in (1, 2, 3):
        add(
            other,
            amount="15",
            currency="NOK",
            category_name="Other",
            description=f"misc {day}",
            transaction_date=date(2026, 11, day),
        )
    assert any("Other" in note for note in other.spending_insights(TODAY))
