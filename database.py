"""SQLite access.

This layer only does parameterized reads and writes and transaction boundaries.
It does not call a language model or decide budget status.
Write functions receive an open connection. The caller commits or rolls back.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from db_setup import initialize, utc_now


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection and force foreign keys on. SQLite applies that switch per connection."""
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def get_connection(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Yield a connection that commits on success and rolls back on error."""
    connection = connect(db_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_database(db_path: Path) -> None:
    """Make sure the database file, tables, and built-in categories exist. Calling this again is safe."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as connection:
        initialize(connection)


def insert_audit(
    connection: sqlite3.Connection,
    *,
    entity_type: str,
    entity_id: int | None,
    action: str,
    changed_fields: list[str],
) -> None:
    """Record the action and the field names.

    The audit table does not store amounts, merchants, or descriptions.
    The original values stay on the transactions row and can still be traced after a soft delete.
    """
    fields = ",".join(sorted(set(changed_fields)))
    connection.execute(
        """
        INSERT INTO audit_logs (entity_type, entity_id, action, changed_fields, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (entity_type, entity_id, action, fields, utc_now()),
    )


def list_categories(connection: sqlite3.Connection, category_type: str | None = None) -> list[dict]:
    """Read categories of one type. An empty type returns all of them."""
    if category_type is None:
        rows = connection.execute(
            "SELECT id, name, category_type, created_at FROM categories ORDER BY id"
        ).fetchall()
    else:
        rows = connection.execute(
            """
            SELECT id, name, category_type, created_at
            FROM categories
            WHERE category_type = ?
            ORDER BY id
            """,
            (category_type,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_category_by_name(
    connection: sqlite3.Connection,
    name: str,
    category_type: str,
) -> dict | None:
    """Find one category by name and type."""
    row = connection.execute(
        """
        SELECT id, name, category_type, created_at
        FROM categories
        WHERE name = ? AND category_type = ?
        """,
        (name, category_type),
    ).fetchone()
    return dict(row) if row else None


def get_or_create_account(
    connection: sqlite3.Connection,
    name: str,
    currency: str | None,
) -> int:
    """Get or create an account by name. Surrounding whitespace is removed first."""
    cleaned = name.strip()
    existing = connection.execute(
        "SELECT id FROM accounts WHERE name = ?",
        (cleaned,),
    ).fetchone()
    if existing:
        return int(existing["id"])
    timestamp = utc_now()
    result = connection.execute(
        """
        INSERT INTO accounts (name, currency, is_deleted, created_at, updated_at)
        VALUES (?, ?, 0, ?, ?)
        """,
        (cleaned, currency, timestamp, timestamp),
    )
    account_id = int(result.lastrowid)
    insert_audit(
        connection,
        entity_type="account",
        entity_id=account_id,
        action="create",
        changed_fields=["name", "currency"],
    )
    return account_id


def insert_transaction(connection: sqlite3.Connection, record: dict) -> int:
    """Insert one transaction. amount_minor in the record must already be an integer minor unit."""
    timestamp = utc_now()
    result = connection.execute(
        """
        INSERT INTO transactions (
            transaction_date, amount_minor, currency, transaction_type, category_id,
            description, merchant, account_id, original_transaction_id, source,
            is_deleted, deleted_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?)
        """,
        (
            record["transaction_date"],
            record["amount_minor"],
            record["currency"],
            record["transaction_type"],
            record["category_id"],
            record.get("description"),
            record.get("merchant"),
            record.get("account_id"),
            record.get("original_transaction_id"),
            record.get("source", "manual"),
            timestamp,
            timestamp,
        ),
    )
    transaction_id = int(result.lastrowid)
    insert_audit(
        connection,
        entity_type="transaction",
        entity_id=transaction_id,
        action="create",
        changed_fields=[
            "transaction_date",
            "amount_minor",
            "currency",
            "transaction_type",
            "category_id",
            "description",
            "merchant",
            "account_id",
            "original_transaction_id",
            "source",
        ],
    )
    return transaction_id


def update_transaction(connection: sqlite3.Connection, transaction_id: int, fields: dict) -> None:
    """Update the editable fields and write only the field names to the audit log."""
    allowed = [
        "transaction_date",
        "amount_minor",
        "currency",
        "transaction_type",
        "category_id",
        "description",
        "merchant",
        "account_id",
        "original_transaction_id",
        "source",
    ]
    current = connection.execute(
        "SELECT * FROM transactions WHERE id = ? AND is_deleted = 0",
        (transaction_id,),
    ).fetchone()
    if current is None:
        raise KeyError(f"The transaction does not exist or is already deleted: {transaction_id}")
    assignments: list[str] = []
    params: list[object] = []
    changed: list[str] = []
    for column in allowed:
        if column not in fields:
            continue
        new_value = fields[column]
        if current[column] != new_value:
            changed.append(column)
        assignments.append(f"{column} = ?")
        params.append(new_value)
    if not assignments:
        return
    assignments.append("updated_at = ?")
    params.append(utc_now())
    params.append(transaction_id)
    connection.execute(
        f"UPDATE transactions SET {', '.join(assignments)} WHERE id = ?",
        params,
    )
    insert_audit(
        connection,
        entity_type="transaction",
        entity_id=transaction_id,
        action="update",
        changed_fields=changed,
    )


def soft_delete_transaction(connection: sqlite3.Connection, transaction_id: int) -> None:
    """Soft-delete a transaction. The row stays, and reports skip it by default."""
    timestamp = utc_now()
    result = connection.execute(
        """
        UPDATE transactions
        SET is_deleted = 1, deleted_at = ?, updated_at = ?
        WHERE id = ? AND is_deleted = 0
        """,
        (timestamp, timestamp, transaction_id),
    )
    if result.rowcount != 1:
        raise KeyError(f"The transaction does not exist or is already deleted: {transaction_id}")
    insert_audit(
        connection,
        entity_type="transaction",
        entity_id=transaction_id,
        action="delete",
        changed_fields=["is_deleted", "deleted_at"],
    )


def restore_transaction(connection: sqlite3.Connection, transaction_id: int) -> None:
    """Restore a soft-deleted transaction."""
    timestamp = utc_now()
    result = connection.execute(
        """
        UPDATE transactions
        SET is_deleted = 0, deleted_at = NULL, updated_at = ?
        WHERE id = ? AND is_deleted = 1
        """,
        (timestamp, transaction_id),
    )
    if result.rowcount != 1:
        raise KeyError(f"The transaction does not exist or is not deleted: {transaction_id}")
    insert_audit(
        connection,
        entity_type="transaction",
        entity_id=transaction_id,
        action="restore",
        changed_fields=["is_deleted", "deleted_at"],
    )


_TRANSACTION_SELECT = """
SELECT
    t.id,
    t.transaction_date,
    t.amount_minor,
    t.currency,
    t.transaction_type,
    t.category_id,
    c.name AS category_name,
    c.category_type AS category_kind,
    t.description,
    t.merchant,
    t.account_id,
    a.name AS account_name,
    t.original_transaction_id,
    ot.transaction_date AS original_date,
    ot.amount_minor AS original_amount_minor,
    ot.currency AS original_currency,
    ot.merchant AS original_merchant,
    oc.name AS original_category_name,
    t.source,
    t.is_deleted,
    t.deleted_at,
    t.created_at,
    t.updated_at
FROM transactions AS t
LEFT JOIN categories AS c ON c.id = t.category_id
LEFT JOIN accounts AS a ON a.id = t.account_id
LEFT JOIN transactions AS ot ON ot.id = t.original_transaction_id
LEFT JOIN categories AS oc ON oc.id = ot.category_id
"""


def get_transaction(connection: sqlite3.Connection, transaction_id: int) -> dict | None:
    """Read one transaction, including a soft-deleted row."""
    row = connection.execute(
        _TRANSACTION_SELECT + " WHERE t.id = ?",
        (transaction_id,),
    ).fetchone()
    return dict(row) if row else None


def escape_like(value: str) -> str:
    """Escape LIKE wildcards so a percent sign in the query is searched literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def list_transactions(
    connection: sqlite3.Connection,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    currency: str | None = None,
    category_id: int | None = None,
    transaction_type: str | None = None,
    keyword: str | None = None,
    include_deleted: bool = False,
) -> list[dict]:
    """Filter transactions. Every user value is passed as a bound parameter."""
    clauses = ["1 = 1"]
    params: list[object] = []
    if not include_deleted:
        clauses.append("t.is_deleted = 0")
    if start_date:
        clauses.append("t.transaction_date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("t.transaction_date <= ?")
        params.append(end_date)
    if currency:
        clauses.append("t.currency = ?")
        params.append(currency)
    if category_id is not None:
        clauses.append("t.category_id = ?")
        params.append(category_id)
    if transaction_type:
        clauses.append("t.transaction_type = ?")
        params.append(transaction_type)
    if keyword and keyword.strip():
        like = f"%{escape_like(keyword.strip())}%"
        clauses.append(
            """(
                IFNULL(t.description, '') LIKE ? ESCAPE '\\'
                OR IFNULL(t.merchant, '') LIKE ? ESCAPE '\\'
                OR IFNULL(a.name, '') LIKE ? ESCAPE '\\'
            )"""
        )
        params.extend([like, like, like])
    query = (
        _TRANSACTION_SELECT
        + " WHERE "
        + " AND ".join(clauses)
        + " ORDER BY t.transaction_date DESC, t.id DESC"
    )
    rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def sum_refunds_for(
    connection: sqlite3.Connection,
    original_transaction_id: int,
    *,
    exclude_id: int | None = None,
) -> int:
    """Sum the undeleted refunds for one expense, in minor units."""
    if exclude_id is None:
        row = connection.execute(
            """
            SELECT COALESCE(SUM(amount_minor), 0) AS total
            FROM transactions
            WHERE original_transaction_id = ?
              AND transaction_type = 'refund'
              AND is_deleted = 0
            """,
            (original_transaction_id,),
        ).fetchone()
    else:
        row = connection.execute(
            """
            SELECT COALESCE(SUM(amount_minor), 0) AS total
            FROM transactions
            WHERE original_transaction_id = ?
              AND transaction_type = 'refund'
              AND is_deleted = 0
              AND id != ?
            """,
            (original_transaction_id, exclude_id),
        ).fetchone()
    return int(row["total"])


def list_budgets(connection: sqlite3.Connection) -> list[dict]:
    """Read every budget with its category name. An overall budget has no category name."""
    rows = connection.execute(
        """
        SELECT
            b.id,
            b.category_id,
            c.name AS category_name,
            b.currency,
            b.monthly_limit_minor,
            b.created_at,
            b.updated_at
        FROM budgets AS b
        LEFT JOIN categories AS c ON c.id = b.category_id
        ORDER BY b.currency, b.category_id IS NOT NULL, c.name
        """
    ).fetchall()
    return [dict(row) for row in rows]


def get_budget(
    connection: sqlite3.Connection,
    *,
    category_id: int | None,
    currency: str,
) -> dict | None:
    """Find a budget by currency and category. An empty category is the monthly total for that currency."""
    if category_id is None:
        row = connection.execute(
            """
            SELECT id, category_id, currency, monthly_limit_minor, created_at, updated_at
            FROM budgets
            WHERE category_id IS NULL AND currency = ?
            """,
            (currency,),
        ).fetchone()
    else:
        row = connection.execute(
            """
            SELECT id, category_id, currency, monthly_limit_minor, created_at, updated_at
            FROM budgets
            WHERE category_id = ? AND currency = ?
            """,
            (category_id, currency),
        ).fetchone()
    return dict(row) if row else None


def upsert_budget(
    connection: sqlite3.Connection,
    *,
    category_id: int | None,
    currency: str,
    monthly_limit_minor: int,
) -> int:
    """Create or update a budget. Uniqueness comes from a partial index."""
    existing = get_budget(connection, category_id=category_id, currency=currency)
    timestamp = utc_now()
    if existing:
        connection.execute(
            """
            UPDATE budgets
            SET monthly_limit_minor = ?, updated_at = ?
            WHERE id = ?
            """,
            (monthly_limit_minor, timestamp, existing["id"]),
        )
        insert_audit(
            connection,
            entity_type="budget",
            entity_id=int(existing["id"]),
            action="update",
            changed_fields=["monthly_limit_minor"],
        )
        return int(existing["id"])
    result = connection.execute(
        """
        INSERT INTO budgets (category_id, currency, monthly_limit_minor, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (category_id, currency, monthly_limit_minor, timestamp, timestamp),
    )
    budget_id = int(result.lastrowid)
    insert_audit(
        connection,
        entity_type="budget",
        entity_id=budget_id,
        action="create",
        changed_fields=["category_id", "currency", "monthly_limit_minor"],
    )
    return budget_id


def delete_budget(connection: sqlite3.Connection, budget_id: int) -> None:
    """Delete one budget."""
    result = connection.execute("DELETE FROM budgets WHERE id = ?", (budget_id,))
    if result.rowcount != 1:
        raise KeyError(f"Budget not found: {budget_id}")
    insert_audit(
        connection,
        entity_type="budget",
        entity_id=budget_id,
        action="delete",
        changed_fields=["monthly_limit_minor"],
    )


def upsert_merchant_memory(
    connection: sqlite3.Connection,
    *,
    merchant_key: str,
    category_name: str,
) -> int:
    """Remember a merchant category the user confirmed. The audit log stores the field name, not the merchant text."""
    timestamp = utc_now()
    existing = connection.execute(
        "SELECT id FROM merchant_memories WHERE merchant_key = ?",
        (merchant_key,),
    ).fetchone()
    if existing:
        connection.execute(
            """
            UPDATE merchant_memories
            SET category_name = ?, updated_at = ?
            WHERE id = ?
            """,
            (category_name, timestamp, existing["id"]),
        )
        memory_id = int(existing["id"])
        action = "update"
    else:
        result = connection.execute(
            """
            INSERT INTO merchant_memories (merchant_key, category_name, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (merchant_key, category_name, timestamp, timestamp),
        )
        memory_id = int(result.lastrowid)
        action = "create"
    insert_audit(
        connection,
        entity_type="merchant_memory",
        entity_id=memory_id,
        action=action,
        changed_fields=["category_name"],
    )
    return memory_id


def list_merchant_memories(connection: sqlite3.Connection) -> list[dict]:
    """Read every remembered merchant category."""
    rows = connection.execute(
        """
        SELECT id, merchant_key, category_name, created_at, updated_at
        FROM merchant_memories
        ORDER BY id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def list_audit_logs(connection: sqlite3.Connection, limit: int = 50) -> list[dict]:
    """Read the most recent audit rows."""
    rows = connection.execute(
        """
        SELECT id, entity_type, entity_id, action, changed_fields, created_at
        FROM audit_logs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]
