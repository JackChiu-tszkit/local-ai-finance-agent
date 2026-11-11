"""Tests for database constraints, transactions, and the audit log."""

from __future__ import annotations

import sqlite3
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from config import EXPENSE_CATEGORIES, INCOME_CATEGORIES, Settings, load_settings, save_settings
from database import (
    connect,
    get_category_by_name,
    get_connection,
    get_transaction,
    init_database,
    insert_transaction,
    list_audit_logs,
    list_transactions,
)
from finance_service import FinanceService


def test_schema_enforces_foreign_keys_and_indexes(tmp_path: Path) -> None:
    """Foreign keys are enabled on every connection, and common query columns are indexed."""
    db_path = tmp_path / "finance.db"
    init_database(db_path)
    connection = connect(db_path)
    try:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        indexes = {
            row[1]
            for row in connection.execute("PRAGMA index_list('transactions')").fetchall()
        }
        assert "idx_transactions_date" in indexes
        assert "idx_transactions_currency" in indexes
        assert "idx_transactions_original" in indexes
        categories = connection.execute("SELECT name, category_type FROM categories").fetchall()
    finally:
        connection.close()
    names = {(row["name"], row["category_type"]) for row in categories}
    for name in EXPENSE_CATEGORIES:
        assert (name, "expense") in names
    for name in INCOME_CATEGORIES:
        assert (name, "income") in names


def test_invalid_category_rolls_back(tmp_path: Path) -> None:
    """An invalid foreign key fails and does not leave a partial transaction."""
    db_path = tmp_path / "finance.db"
    init_database(db_path)
    with pytest.raises(sqlite3.IntegrityError):
        with get_connection(db_path) as connection:
            insert_transaction(
                connection,
                {
                    "transaction_date": "2026-11-02",
                    "amount_minor": 1000,
                    "currency": "NOK",
                    "transaction_type": "expense",
                    "category_id": 999999,
                    "description": "should not be saved",
                    "merchant": None,
                    "account_id": None,
                    "original_transaction_id": None,
                    "source": "manual",
                },
            )
    with get_connection(db_path) as connection:
        assert list_transactions(connection) == []


def test_transaction_rollback_keeps_previous_rows(tmp_path: Path) -> None:
    """If a transaction fails halfway, the whole write is rolled back."""
    db_path = tmp_path / "finance.db"
    init_database(db_path)
    with pytest.raises(RuntimeError):
        with get_connection(db_path) as connection:
            category = get_category_by_name(connection, "Dining", "expense")
            assert category is not None
            insert_transaction(
                connection,
                {
                    "transaction_date": "2026-11-02",
                    "amount_minor": 12050,
                    "currency": "CNY",
                    "transaction_type": "expense",
                    "category_id": category["id"],
                    "description": "hotpot",
                    "merchant": None,
                    "account_id": None,
                    "original_transaction_id": None,
                    "source": "manual",
                },
            )
            raise RuntimeError("simulated failure")
    with get_connection(db_path) as connection:
        assert list_transactions(connection) == []


def test_soft_delete_hides_row_but_keeps_it(tmp_path: Path) -> None:
    service = FinanceService(tmp_path / "finance.db")
    transaction_id = service.create_transaction(
        transaction_date=date(2026, 11, 2),
        amount=Decimal("88"),
        currency="CNY",
        transaction_type="expense",
        category_name="Transport",
        description="metro",
        merchant=None,
        account_name=None,
        source="manual",
    )
    service.delete_transaction(transaction_id, confirmed=True)
    assert service.list_transaction_rows() == []
    with get_connection(service.db_path) as connection:
        stored = get_transaction(connection, transaction_id)
    assert stored is not None
    assert stored["is_deleted"] == 1
    service.restore_transaction(transaction_id)
    restored = service.list_transaction_rows()
    assert len(restored) == 1
    assert restored[0]["description"] == "metro"


def test_keyword_search_escapes_like_wildcards(tmp_path: Path) -> None:
    """A percent sign and underscore in the query are searched literally and do not match every row."""
    service = FinanceService(tmp_path / "finance.db")
    service.create_transaction(
        transaction_date=date(2026, 11, 2),
        amount=Decimal("10"),
        currency="CNY",
        transaction_type="expense",
        category_name="Shopping",
        description="100%_off",
        merchant=None,
        account_name=None,
        source="manual",
    )
    service.create_transaction(
        transaction_date=date(2026, 11, 2),
        amount=Decimal("11"),
        currency="CNY",
        transaction_type="expense",
        category_name="Shopping",
        description="100Xoff",
        merchant=None,
        account_name=None,
        source="manual",
    )
    percent_hits = service.list_transaction_rows(keyword="%")
    underscore_hits = service.list_transaction_rows(keyword="_")
    assert [row["description"] for row in percent_hits] == ["100%_off"]
    assert [row["description"] for row in underscore_hits] == ["100%_off"]


def test_sql_injection_is_stored_as_text(tmp_path: Path) -> None:
    """A hostile string is stored only as a merchant name and cannot change the table structure."""
    service = FinanceService(tmp_path / "finance.db")
    payload = "'); DROP TABLE transactions; --"
    service.create_transaction(
        transaction_date=date(2026, 11, 2),
        amount=Decimal("15"),
        currency="NOK",
        transaction_type="expense",
        category_name="Other",
        description="ordinary description",
        merchant=payload,
        account_name=None,
        source="text",
    )
    rows = service.list_transaction_rows()
    assert rows[0]["merchant"] == payload
    assert service.list_categories()


def test_audit_log_does_not_copy_financial_values(tmp_path: Path) -> None:
    """The audit log stores field names and does not copy the amount or merchant."""
    service = FinanceService(tmp_path / "finance.db")
    service.create_transaction(
        transaction_date=date(2026, 11, 2),
        amount=Decimal("123.45"),
        currency="CNY",
        transaction_type="expense",
        category_name="Dining",
        description="secret lunch",
        merchant="SecretShop",
        account_name="Vipps",
        source="manual",
    )
    with get_connection(service.db_path) as connection:
        logs = list_audit_logs(connection)
    combined = " ".join(
        f"{row['action']} {row['changed_fields']}" for row in logs
    )
    assert "amount_minor" in combined
    assert "123.45" not in combined
    assert "SecretShop" not in combined
    assert "secret lunch" not in combined
    assert "Vipps" not in combined


def test_settings_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    settings = Settings(default_currency="CNY", timezone="Asia/Shanghai", text_model="qwen2.5:3b")
    save_settings(settings, path)
    loaded = load_settings(path)
    assert loaded.default_currency == "CNY"
    assert loaded.timezone == "Asia/Shanghai"
    assert loaded.text_model == "qwen2.5:3b"
    assert loaded.confirm_all is True
