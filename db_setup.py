"""创建 SQLite 数据库并写入预设分类。

可以单独执行：

    python db_setup.py
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from config import DB_PATH, EXPENSE_CATEGORIES, INCOME_CATEGORIES


# 金额使用整数最小货币单位 amount_minor，避免 REAL 浮点误差。
# 例如人民币和挪威克朗都是 2 位小数，120.50 保存为 12050。
SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category_type TEXT NOT NULL CHECK (category_type IN ('expense', 'income')),
    created_at TEXT NOT NULL,
    UNIQUE (name, category_type)
);

CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    currency TEXT CHECK (currency IS NULL OR length(currency) = 3),
    is_deleted INTEGER NOT NULL DEFAULT 0 CHECK (is_deleted IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY,
    transaction_date TEXT NOT NULL CHECK (length(transaction_date) = 10),
    amount_minor INTEGER NOT NULL CHECK (amount_minor > 0),
    currency TEXT NOT NULL CHECK (length(currency) = 3),
    transaction_type TEXT NOT NULL CHECK (transaction_type IN ('expense', 'income', 'refund')),
    category_id INTEGER NOT NULL,
    description TEXT,
    merchant TEXT,
    account_id INTEGER,
    original_transaction_id INTEGER,
    source TEXT NOT NULL DEFAULT 'manual'
        CHECK (source IN ('text', 'receipt', 'voice', 'manual', 'refund')),
    is_deleted INTEGER NOT NULL DEFAULT 0 CHECK (is_deleted IN (0, 1)),
    deleted_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories (id),
    FOREIGN KEY (account_id) REFERENCES accounts (id),
    FOREIGN KEY (original_transaction_id) REFERENCES transactions (id)
);

CREATE TABLE IF NOT EXISTS budgets (
    id INTEGER PRIMARY KEY,
    category_id INTEGER,
    currency TEXT NOT NULL CHECK (length(currency) = 3),
    monthly_limit_minor INTEGER NOT NULL CHECK (monthly_limit_minor >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories (id)
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_id INTEGER,
    action TEXT NOT NULL,
    changed_fields TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS merchant_memories (
    id INTEGER PRIMARY KEY,
    merchant_key TEXT NOT NULL UNIQUE,
    category_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_transactions_date
ON transactions (transaction_date);

CREATE INDEX IF NOT EXISTS idx_transactions_currency
ON transactions (currency);

CREATE INDEX IF NOT EXISTS idx_transactions_category
ON transactions (category_id);

CREATE INDEX IF NOT EXISTS idx_transactions_deleted
ON transactions (is_deleted);

CREATE INDEX IF NOT EXISTS idx_transactions_original
ON transactions (original_transaction_id);

CREATE INDEX IF NOT EXISTS idx_transactions_type
ON transactions (transaction_type);

CREATE UNIQUE INDEX IF NOT EXISTS idx_budgets_category_currency
ON budgets (category_id, currency)
WHERE category_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_budgets_total_currency
ON budgets (currency)
WHERE category_id IS NULL;
"""


def utc_now() -> str:
    """返回不带微秒的 UTC 时间戳，作为创建和更新时间。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def apply_schema(connection: sqlite3.Connection) -> None:
    """执行建表脚本。executescript 会在内部提交当前事务。"""
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)


def seed_categories(connection: sqlite3.Connection) -> None:
    """写入预设分类。重复执行不会产生重复行。"""
    timestamp = utc_now()
    rows = [(name, "expense", timestamp) for name in EXPENSE_CATEGORIES]
    rows.extend((name, "income", timestamp) for name in INCOME_CATEGORIES)
    connection.executemany(
        """
        INSERT OR IGNORE INTO categories (name, category_type, created_at)
        VALUES (?, ?, ?)
        """,
        rows,
    )


def initialize(connection: sqlite3.Connection) -> None:
    """在已打开的连接上创建表并填充分类。"""
    apply_schema(connection)
    seed_categories(connection)


def main() -> None:
    """命令行初始化默认数据库。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        initialize(connection)
        connection.commit()
    finally:
        connection.close()
    print(f"数据库已就绪：{DB_PATH}")


if __name__ == "__main__":
    main()
