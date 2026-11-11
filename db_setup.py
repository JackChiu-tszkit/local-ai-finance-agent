"""Create the SQLite database and seed the built-in categories.

Run it on its own with:

    python db_setup.py
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from config import DB_PATH, EXPENSE_CATEGORIES, INCOME_CATEGORIES


# Amounts are stored as integer minor units in amount_minor, not as REAL.
# NOK and CNY both use 2 decimal places, so 120.50 is stored as 12050.
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
    """UTC timestamp without microseconds, used for created and updated times."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def apply_schema(connection: sqlite3.Connection) -> None:
    """Create the tables. executescript commits the current transaction."""
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)


def seed_categories(connection: sqlite3.Connection) -> None:
    """Insert the built-in categories. Running this again does not duplicate rows."""
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
    """Create the tables and fill the categories on an open connection."""
    apply_schema(connection)
    seed_categories(connection)


def main() -> None:
    """Initialize the default database from the command line."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        initialize(connection)
        connection.commit()
    finally:
        connection.close()
    print(f"Database ready: {DB_PATH}")


if __name__ == "__main__":
    main()
