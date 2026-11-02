"""SQLite 访问层。

这里只负责参数化读写和事务边界，不调用语言模型，也不计算预算结论。
所有写入函数都接收外部传入的连接，由调用方统一提交或回滚。
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from db_setup import initialize, utc_now


def connect(db_path: Path) -> sqlite3.Connection:
    """打开连接并强制启用外键。SQLite 的外键开关是按连接生效的。"""
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def get_connection(db_path: Path) -> Iterator[sqlite3.Connection]:
    """提供一个会自动提交或回滚的连接。"""
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
    """确保数据库文件、表和预设分类存在。重复调用是安全的。"""
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
    """记录发生过的操作和字段名。

    审计表不保存金额、商户和描述，避免在日志型数据里复制一份敏感内容。
    交易原值仍保留在 transactions 表中，软删除后也可以追溯。
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
    """按类型读取分类。类型为空时返回全部。"""
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
    """按名称和收支类型精确查找分类。"""
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
    """按账户名获取或创建账户。名称会先去掉首尾空白。"""
    cleaned = name.strip()
    existing = connection.execute(
        "SELECT id FROM accounts WHERE name = ?",
        (cleaned,),
    ).fetchone()
    if existing:
        return int(existing["id"])
    timestamp = utc_now()
    cursor = connection.execute(
        """
        INSERT INTO accounts (name, currency, is_deleted, created_at, updated_at)
        VALUES (?, ?, 0, ?, ?)
        """,
        (cleaned, currency, timestamp, timestamp),
    )
    account_id = int(cursor.lastrowid)
    insert_audit(
        connection,
        entity_type="account",
        entity_id=account_id,
        action="create",
        changed_fields=["name", "currency"],
    )
    return account_id


def insert_transaction(connection: sqlite3.Connection, record: dict) -> int:
    """插入一条交易。record 中的金额必须已经是整数最小单位。"""
    timestamp = utc_now()
    cursor = connection.execute(
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
    transaction_id = int(cursor.lastrowid)
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
    """更新允许修改的业务字段，并只把字段名写入审计表。"""
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
        raise KeyError(f"交易不存在或已删除：{transaction_id}")
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
    """软删除交易。历史行保留，报表默认不再统计。"""
    timestamp = utc_now()
    cursor = connection.execute(
        """
        UPDATE transactions
        SET is_deleted = 1, deleted_at = ?, updated_at = ?
        WHERE id = ? AND is_deleted = 0
        """,
        (timestamp, timestamp, transaction_id),
    )
    if cursor.rowcount != 1:
        raise KeyError(f"交易不存在或已删除：{transaction_id}")
    insert_audit(
        connection,
        entity_type="transaction",
        entity_id=transaction_id,
        action="delete",
        changed_fields=["is_deleted", "deleted_at"],
    )


def restore_transaction(connection: sqlite3.Connection, transaction_id: int) -> None:
    """恢复软删除的交易。"""
    timestamp = utc_now()
    cursor = connection.execute(
        """
        UPDATE transactions
        SET is_deleted = 0, deleted_at = NULL, updated_at = ?
        WHERE id = ? AND is_deleted = 1
        """,
        (timestamp, transaction_id),
    )
    if cursor.rowcount != 1:
        raise KeyError(f"交易不存在或并未删除：{transaction_id}")
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
    """读取单条交易，包含已软删除的记录。"""
    row = connection.execute(
        _TRANSACTION_SELECT + " WHERE t.id = ?",
        (transaction_id,),
    ).fetchone()
    return dict(row) if row else None


def escape_like(value: str) -> str:
    """转义 LIKE 通配符，使用户输入的百分号按字面意思搜索。"""
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
    """按条件筛选交易。所有用户输入都通过参数绑定传入。"""
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
    """汇总某笔支出尚未删除的退款金额，单位是最小货币单位。"""
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
    """读取全部预算，并带上分类名称。总预算的分类名为空。"""
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
    """按币种和分类查找预算。分类为空表示该币种的月度总支出预算。"""
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
    """创建或更新预算。唯一性由部分索引保证。"""
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
    cursor = connection.execute(
        """
        INSERT INTO budgets (category_id, currency, monthly_limit_minor, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (category_id, currency, monthly_limit_minor, timestamp, timestamp),
    )
    budget_id = int(cursor.lastrowid)
    insert_audit(
        connection,
        entity_type="budget",
        entity_id=budget_id,
        action="create",
        changed_fields=["category_id", "currency", "monthly_limit_minor"],
    )
    return budget_id


def delete_budget(connection: sqlite3.Connection, budget_id: int) -> None:
    """删除一条预算设置。"""
    cursor = connection.execute("DELETE FROM budgets WHERE id = ?", (budget_id,))
    if cursor.rowcount != 1:
        raise KeyError(f"预算不存在：{budget_id}")
    insert_audit(
        connection,
        entity_type="budget",
        entity_id=budget_id,
        action="delete",
        changed_fields=["monthly_limit_minor"],
    )


def list_audit_logs(connection: sqlite3.Connection, limit: int = 50) -> list[dict]:
    """读取最近的审计记录。"""
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
