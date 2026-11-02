"""数据库约束、事务和审计测试。"""

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
    """外键必须在每条连接上启用，常用查询字段要有索引。"""
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
    """非法外键会失败，并且不会留下半条交易。"""
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
                    "description": "不应该保存",
                    "merchant": None,
                    "account_id": None,
                    "original_transaction_id": None,
                    "source": "manual",
                },
            )
    with get_connection(db_path) as connection:
        assert list_transactions(connection) == []


def test_transaction_rollback_keeps_previous_rows(tmp_path: Path) -> None:
    """事务中途失败时，整次写入都回滚。"""
    db_path = tmp_path / "finance.db"
    init_database(db_path)
    with pytest.raises(RuntimeError):
        with get_connection(db_path) as connection:
            category = get_category_by_name(connection, "餐饮", "expense")
            assert category is not None
            insert_transaction(
                connection,
                {
                    "transaction_date": "2026-11-02",
                    "amount_minor": 12050,
                    "currency": "CNY",
                    "transaction_type": "expense",
                    "category_id": category["id"],
                    "description": "火锅",
                    "merchant": None,
                    "account_id": None,
                    "original_transaction_id": None,
                    "source": "manual",
                },
            )
            raise RuntimeError("模拟失败")
    with get_connection(db_path) as connection:
        assert list_transactions(connection) == []


def test_soft_delete_hides_row_but_keeps_it(tmp_path: Path) -> None:
    service = FinanceService(tmp_path / "finance.db")
    transaction_id = service.create_transaction(
        transaction_date=date(2026, 11, 2),
        amount=Decimal("88"),
        currency="CNY",
        transaction_type="expense",
        category_name="交通",
        description="地铁",
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
    assert restored[0]["description"] == "地铁"


def test_keyword_search_escapes_like_wildcards(tmp_path: Path) -> None:
    """用户输入的百分号和下划线按字面搜索，不能匹配所有记录。"""
    service = FinanceService(tmp_path / "finance.db")
    service.create_transaction(
        transaction_date=date(2026, 11, 2),
        amount=Decimal("10"),
        currency="CNY",
        transaction_type="expense",
        category_name="购物",
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
        category_name="购物",
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
    """恶意字符串只作为商户名称保存，不能破坏表结构。"""
    service = FinanceService(tmp_path / "finance.db")
    payload = "'); DROP TABLE transactions; --"
    service.create_transaction(
        transaction_date=date(2026, 11, 2),
        amount=Decimal("15"),
        currency="NOK",
        transaction_type="expense",
        category_name="其他",
        description="正常描述",
        merchant=payload,
        account_name=None,
        source="text",
    )
    rows = service.list_transaction_rows()
    assert rows[0]["merchant"] == payload
    assert service.list_categories()


def test_audit_log_does_not_copy_financial_values(tmp_path: Path) -> None:
    """审计日志只记录字段名，不复制金额和商户。"""
    service = FinanceService(tmp_path / "finance.db")
    service.create_transaction(
        transaction_date=date(2026, 11, 2),
        amount=Decimal("123.45"),
        currency="CNY",
        transaction_type="expense",
        category_name="餐饮",
        description="秘密午饭",
        merchant="SecretShop",
        account_name="支付宝",
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
    assert "秘密午饭" not in combined
    assert "支付宝" not in combined


def test_settings_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    settings = Settings(default_currency="CNY", timezone="Asia/Shanghai", text_model="qwen2.5:3b")
    save_settings(settings, path)
    loaded = load_settings(path)
    assert loaded.default_currency == "CNY"
    assert loaded.timezone == "Asia/Shanghai"
    assert loaded.text_model == "qwen2.5:3b"
    assert loaded.confirm_all is True
