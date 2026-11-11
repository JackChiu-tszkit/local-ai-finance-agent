"""Finance rules.

Money conversion, duplicates, refund limits, budgets, and totals are done here with Decimal and integers.
The language model does not calculate any total.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from io import BytesIO
from pathlib import Path

import pandas as pd

from config import CURRENCY_EXPONENTS, EXPENSE_CATEGORIES, TRANSACTION_TYPE_LABELS, merchant_key
from database import (
    delete_budget,
    get_category_by_name,
    get_connection,
    get_or_create_account,
    get_transaction,
    init_database,
    insert_transaction,
    list_budgets,
    list_categories,
    list_merchant_memories,
    list_transactions,
    soft_delete_transaction,
    sum_refunds_for,
    restore_transaction,
    update_transaction,
    upsert_budget,
    upsert_merchant_memory,
)
from difflib import SequenceMatcher


logger = logging.getLogger("ai_finance_agent")

_VALID_TYPES = {"expense", "income", "refund"}
_VALID_SOURCES = {"text", "receipt", "voice", "manual", "refund"}


class DuplicateTransactionError(ValueError):
    """A similar past transaction exists, and the caller did not explicitly allow the save."""

    def __init__(self, report: "DuplicateReport") -> None:
        super().__init__("A similar transaction was found. Confirm before saving.")
        self.report = report


@dataclass
class DuplicateReport:
    """Duplicate check. exact is identical. suspected is the same day and amount with a close merchant or description."""

    exact: list[dict] = field(default_factory=list)
    suspected: list[dict] = field(default_factory=list)

    @property
    def has_any(self) -> bool:
        return bool(self.exact or self.suspected)


@dataclass
class BudgetStatus:
    """How much of one currency's overall or category budget was used in a month."""

    budget_id: int
    currency: str
    category_name: str | None
    limit_minor: int
    spent_minor: int
    ratio: Decimal
    level: str


@dataclass
class CurrencySummary:
    """Totals for one currency. Different currencies are not added together."""

    currency: str
    total_income_minor: int
    total_expense_minor: int
    total_refund_minor: int
    net_expense_minor: int
    balance_minor: int
    category_net: list[tuple[str, int]]
    periods: list[str]
    period_income: list[int]
    period_net_expense: list[int]
    recent: list[dict]


def minor_exponent(currency: str) -> int:
    """Decimal places for a currency. An unknown currency uses 2."""
    return CURRENCY_EXPONENTS.get(currency.upper(), 2)


def to_minor(amount: Decimal, currency: str) -> int:
    """Convert a Decimal amount to minor units. A negative amount is rejected."""
    if not isinstance(amount, Decimal):
        amount = Decimal(str(amount))
    if amount < 0:
        raise ValueError("The amount cannot be negative.")
    exponent = minor_exponent(currency)
    quant = Decimal(1).scaleb(-exponent)
    quantized = amount.quantize(quant, rounding=ROUND_HALF_UP)
    factor = Decimal(10) ** exponent
    return int((quantized * factor).to_integral_value(rounding=ROUND_HALF_UP))


def from_minor(amount_minor: int, currency: str) -> Decimal:
    """Convert minor units back to a Decimal."""
    exponent = minor_exponent(currency)
    quant = Decimal(1).scaleb(-exponent)
    return (Decimal(amount_minor) / (Decimal(10) ** exponent)).quantize(quant)


def format_amount(amount_minor: int, currency: str) -> str:
    """Format an amount without the currency code."""
    value = from_minor(amount_minor, currency)
    exponent = minor_exponent(currency)
    if exponent == 0:
        return f"{int(value):,}"
    return f"{value:,.{exponent}f}"


def format_money(amount_minor: int, currency: str) -> str:
    """Format an amount with its currency code."""
    return f"{format_amount(amount_minor, currency)} {currency.upper()}"


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    text = " ".join(value.split())
    return text or None


def _normalize(value: str | None) -> str:
    return (value or "").casefold().strip()


def _similar(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _period_key(iso_date: str, frequency: str) -> str:
    current = date.fromisoformat(iso_date)
    if frequency == "day":
        return current.isoformat()
    if frequency == "week":
        iso = current.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    if frequency == "month":
        return f"{current.year}-{current.month:02d}"
    raise ValueError("The period must be day, week, or month.")


def shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    """Shift a calendar month. A negative delta moves backward."""
    index = year * 12 + (month - 1) + delta
    return index // 12, index % 12 + 1


def _month_bounds(year: int, month: int) -> tuple[str, str]:
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1)
    else:
        end = date(year, month + 1, 1)
    # The end date is the last day of the month because list queries are inclusive.
    return start.isoformat(), (end.fromordinal(end.toordinal() - 1)).isoformat()


def decorate(row: dict) -> dict:
    """Add display amount strings for the interface and export. Calculations still use amount_minor."""
    item = dict(row)
    item["amount"] = format_amount(int(row["amount_minor"]), row["currency"])
    item["type_label"] = TRANSACTION_TYPE_LABELS.get(row["transaction_type"], row["transaction_type"])
    return item


def _excel_safe(value: object) -> object:
    """Stop a spreadsheet from treating a description that starts with = as a formula."""
    if isinstance(value, str) and value[:1] in {"=", "+", "-", "@"}:
        return "'" + value
    return value


def _has_currency_expense(rows: list[dict], currency: str) -> bool:
    return any(
        row["currency"] == currency and row["transaction_type"] == "expense" and not row.get("is_deleted")
        for row in rows
    )


def _spent_by_category(rows: list[dict], currency: str) -> dict[str, int]:
    """Net expense by category for one currency. A refund reduces the original category."""
    names: list[str] = []
    for row in rows:
        if row["currency"] != currency:
            continue
        if row["transaction_type"] == "expense" and row.get("category_name"):
            name = str(row["category_name"])
        elif row["transaction_type"] == "refund" and row.get("original_category_name"):
            name = str(row["original_category_name"])
        else:
            continue
        if name not in names:
            names.append(name)
    return {name: net_spent_minor(rows, currency, name) for name in names}


def _merchant_display(rows: list[dict], currency: str) -> dict[str, str]:
    """Return a merchant comparison key and the display name from the latest row."""
    found: dict[str, str] = {}
    for row in rows:
        if row["currency"] != currency or row["transaction_type"] != "expense":
            continue
        merchant = (row.get("merchant") or "").strip()
        if not merchant:
            continue
        found[merchant_key(merchant)] = merchant
    return found


def net_spent_minor(rows: list[dict], currency: str, category_name: str | None = None) -> int:
    """Net expense. A refund reduces the original category and is not treated as salary or investment income."""
    total = 0
    for row in rows:
        if row["currency"] != currency or row.get("is_deleted"):
            continue
        if row["transaction_type"] == "expense":
            if category_name is None or row.get("category_name") == category_name:
                total += int(row["amount_minor"])
        elif row["transaction_type"] == "refund":
            attributed = row.get("original_category_name")
            if category_name is None or attributed == category_name:
                total -= int(row["amount_minor"])
    return total


class FinanceService:
    """Finance operations used by the interface. Each write runs in one database transaction."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        init_database(self.db_path)

    def merchant_memories(self) -> dict[str, str]:
        """Map a merchant comparison key to a category name. Built-in merchants are not included."""
        with get_connection(self.db_path) as connection:
            rows = list_merchant_memories(connection)
        return {row["merchant_key"]: row["category_name"] for row in rows}

    def remember_merchant(
        self,
        merchant: str | None,
        category_name: str,
        transaction_type: str,
    ) -> None:
        """Remember a merchant category after the user confirms an expense. Skip rows with no merchant, and skip non-expenses."""
        if transaction_type != "expense" or not merchant or not merchant.strip():
            return
        if category_name not in EXPENSE_CATEGORIES:
            return
        key = merchant_key(merchant)
        if not key:
            return
        with get_connection(self.db_path) as connection:
            upsert_merchant_memory(connection, merchant_key=key, category_name=category_name)
        logger.info("merchant memory saved category=%s", category_name)

    def recategorize(self, transaction_ids: list[int], category_name: str) -> int:
        """Move the previewed expenses to a new category. The user must already have confirmed."""
        if category_name not in EXPENSE_CATEGORIES:
            raise ValueError("An expense can only be moved to an existing expense category.")
        updated = 0
        for transaction_id in transaction_ids:
            with get_connection(self.db_path) as connection:
                row = get_transaction(connection, transaction_id)
            if row is None or row["is_deleted"] or row["transaction_type"] != "expense":
                continue
            if row.get("category_name") == category_name:
                continue
            self.update_transaction(
                transaction_id,
                transaction_date=date.fromisoformat(row["transaction_date"]),
                amount=from_minor(int(row["amount_minor"]), row["currency"]),
                currency=row["currency"],
                transaction_type="expense",
                category_name=category_name,
                description=row.get("description"),
                merchant=row.get("merchant"),
                account_name=row.get("account_name"),
                original_transaction_id=None,
                allow_duplicate=True,
            )
            self.remember_merchant(row.get("merchant"), category_name, "expense")
            updated += 1
        return updated

    def spending_insights(self, today: date) -> list[str]:
        """Reminders from this person's own history. The comparison and thresholds are calculated here, not by a model."""
        notes: list[str] = []
        current_rows = self._month_rows(today.year, today.month)
        previous: list[list[dict]] = []
        year, month = today.year, today.month
        for _step in range(3):
            year, month = shift_month(year, month, -1)
            previous.append(self._month_rows(year, month))
        currencies: list[str] = []
        for row in current_rows:
            if row["currency"] not in currencies:
                currencies.append(row["currency"])
        for currency in currencies:
            current_spent = _spent_by_category(current_rows, currency)
            previous_spent = [_spent_by_category(rows, currency) for rows in previous]
            active_months = sum(1 for rows in previous if _has_currency_expense(rows, currency))
            if active_months >= 2:
                floor = to_minor(Decimal("100"), currency)
                for name, spent in current_spent.items():
                    if spent <= 0:
                        continue
                    average = sum((item.get(name, 0) for item in previous_spent), Decimal(0)) / Decimal(3)
                    if average <= 0:
                        continue
                    if Decimal(spent) >= average * Decimal("1.5") and Decimal(spent) - average >= Decimal(floor):
                        notes.append(
                            f"This month {name} net expense is {format_money(spent, currency)}, "
                            f"above the previous three-month average of {format_money(int(average), currency)}. "
                            "This reminder is calculated from your own history."
                        )
            other_count = sum(
                1
                for row in current_rows
                if row["currency"] == currency
                and row["transaction_type"] == "expense"
                and row.get("category_name") == "Other"
            )
            if other_count >= 3:
                notes.append(f"This month has {other_count} {currency} expenses in Other. The assistant can help recategorize them.")
            current_merchants = _merchant_display(current_rows, currency)
            for key, display in current_merchants.items():
                hits = sum(1 for rows in previous if key in _merchant_display(rows, currency))
                if hits >= 2:
                    notes.append(
                        f"{display} appears this month and in at least two of the previous three months. "
                        "This is only a reminder; nothing is recorded automatically."
                    )
        return notes[:8]

    def _month_rows(self, year: int, month: int) -> list[dict]:
        start, end = _month_bounds(year, month)
        return self.list_transaction_rows(
            start_date=date.fromisoformat(start),
            end_date=date.fromisoformat(end),
        )

    def list_categories(self, category_type: str | None = None) -> list[dict]:
        with get_connection(self.db_path) as connection:
            return list_categories(connection, category_type)

    def list_transaction_rows(
        self,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        currency: str | None = None,
        category_id: int | None = None,
        transaction_type: str | None = None,
        keyword: str | None = None,
        include_deleted: bool = False,
    ) -> list[dict]:
        """Read transactions for the filters and attach a display amount."""
        with get_connection(self.db_path) as connection:
            rows = list_transactions(
                connection,
                start_date=start_date.isoformat() if start_date else None,
                end_date=end_date.isoformat() if end_date else None,
                currency=currency.upper() if currency else None,
                category_id=category_id,
                transaction_type=transaction_type,
                keyword=keyword,
                include_deleted=include_deleted,
            )
        return [decorate(row) for row in rows]

    def create_transaction(
        self,
        *,
        transaction_date: date,
        amount: Decimal,
        currency: str,
        transaction_type: str,
        category_name: str,
        description: str | None,
        merchant: str | None,
        account_name: str | None,
        original_transaction_id: int | None = None,
        source: str = "manual",
        allow_duplicate: bool = False,
    ) -> int:
        """Write a confirmed transaction. A similar existing row requires an explicit allow."""
        prepared = self._prepare(
            transaction_date=transaction_date,
            amount=amount,
            currency=currency,
            transaction_type=transaction_type,
            category_name=category_name,
            description=description,
            merchant=merchant,
            account_name=account_name,
            original_transaction_id=original_transaction_id,
            source=source,
        )
        with get_connection(self.db_path) as connection:
            report = self._duplicates_on(connection, prepared, exclude_id=None)
            if report.has_any and not allow_duplicate:
                raise DuplicateTransactionError(report)
            category_id, account_id = self._resolve_links(connection, prepared)
            if prepared["transaction_type"] == "refund":
                self._assert_refund_capacity(
                    connection,
                    int(prepared["original_transaction_id"]),
                    prepared["amount_minor"],
                    prepared["currency"],
                )
            elif prepared["original_transaction_id"] is not None:
                raise ValueError("Only a refund can be linked to an original transaction.")
            transaction_id = insert_transaction(
                connection,
                {
                    "transaction_date": prepared["transaction_date"],
                    "amount_minor": prepared["amount_minor"],
                    "currency": prepared["currency"],
                    "transaction_type": prepared["transaction_type"],
                    "category_id": category_id,
                    "description": prepared["description"],
                    "merchant": prepared["merchant"],
                    "account_id": account_id,
                    "original_transaction_id": prepared["original_transaction_id"],
                    "source": "refund" if prepared["transaction_type"] == "refund" else prepared["source"],
                },
            )
        logger.info(
            "transaction created id=%s type=%s source=%s",
            transaction_id,
            prepared["transaction_type"],
            prepared["source"],
        )
        return transaction_id

    def update_transaction(
        self,
        transaction_id: int,
        *,
        transaction_date: date,
        amount: Decimal,
        currency: str,
        transaction_type: str,
        category_name: str,
        description: str | None,
        merchant: str | None,
        account_name: str | None,
        original_transaction_id: int | None = None,
        allow_duplicate: bool = False,
    ) -> None:
        """Edit a transaction that is not deleted, and check the refund limit and duplicates again."""
        prepared = self._prepare(
            transaction_date=transaction_date,
            amount=amount,
            currency=currency,
            transaction_type=transaction_type,
            category_name=category_name,
            description=description,
            merchant=merchant,
            account_name=account_name,
            original_transaction_id=original_transaction_id,
            source="manual",
        )
        with get_connection(self.db_path) as connection:
            current = get_transaction(connection, transaction_id)
            if current is None or current["is_deleted"]:
                raise KeyError("The transaction does not exist or is already deleted.")
            self._assert_existing_refunds_still_fit(connection, current, prepared)
            report = self._duplicates_on(connection, prepared, exclude_id=transaction_id)
            if report.has_any and not allow_duplicate:
                raise DuplicateTransactionError(report)
            category_id, account_id = self._resolve_links(connection, prepared)
            if prepared["transaction_type"] == "refund":
                self._assert_refund_capacity(
                    connection,
                    int(prepared["original_transaction_id"]),
                    prepared["amount_minor"],
                    prepared["currency"],
                    exclude_id=transaction_id,
                )
            update_transaction(
                connection,
                transaction_id,
                {
                    "transaction_date": prepared["transaction_date"],
                    "amount_minor": prepared["amount_minor"],
                    "currency": prepared["currency"],
                    "transaction_type": prepared["transaction_type"],
                    "category_id": category_id,
                    "description": prepared["description"],
                    "merchant": prepared["merchant"],
                    "account_id": account_id,
                    "original_transaction_id": prepared["original_transaction_id"],
                },
            )
        logger.info("transaction updated id=%s", transaction_id)

    def delete_transaction(self, transaction_id: int, *, confirmed: bool) -> None:
        """Soft-delete. An expense that still has an active refund cannot be deleted, and old refunds are not removed automatically."""
        if not confirmed:
            raise ValueError("Confirm before deleting.")
        with get_connection(self.db_path) as connection:
            current = get_transaction(connection, transaction_id)
            if current is None or current["is_deleted"]:
                raise KeyError("The transaction does not exist or is already deleted.")
            if current["transaction_type"] == "expense" and sum_refunds_for(connection, transaction_id) > 0:
                raise ValueError("This expense still has a refund, so it cannot be deleted.")
            soft_delete_transaction(connection, transaction_id)
        logger.info("transaction soft-deleted id=%s", transaction_id)

    def restore_transaction(self, transaction_id: int) -> None:
        """Restore a soft-deleted row. Restoring a refund checks the remaining balance on the original expense again."""
        with get_connection(self.db_path) as connection:
            current = get_transaction(connection, transaction_id)
            if current is None or not current["is_deleted"]:
                raise KeyError("The transaction does not exist or is not deleted.")
            if current["transaction_type"] == "refund":
                self._assert_refund_capacity(
                    connection,
                    int(current["original_transaction_id"]),
                    int(current["amount_minor"]),
                    current["currency"],
                    exclude_id=transaction_id,
                )
            restore_transaction(connection, transaction_id)
        logger.info("transaction restored id=%s", transaction_id)

    def find_duplicates(
        self,
        *,
        transaction_date: date,
        amount: Decimal,
        currency: str,
        transaction_type: str,
        description: str | None,
        merchant: str | None,
        exclude_id: int | None = None,
    ) -> DuplicateReport:
        """Show similar transactions on the confirmation form. The write checks again."""
        code = currency.strip().upper()
        if transaction_type not in _VALID_TYPES:
            raise ValueError("Invalid transaction type.")
        amount_minor = to_minor(Decimal(amount), code)
        if amount_minor <= 0:
            raise ValueError("Enter an amount greater than 0.")
        prepared = {
            "transaction_date": transaction_date.isoformat(),
            "amount_minor": amount_minor,
            "currency": code,
            "transaction_type": transaction_type,
            "description": _blank_to_none(description),
            "merchant": _blank_to_none(merchant),
        }
        with get_connection(self.db_path) as connection:
            return self._duplicates_on(connection, prepared, exclude_id=exclude_id)

    def find_refund_candidates(
        self,
        *,
        amount: Decimal,
        currency: str,
        merchant: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        exclude_refund_id: int | None = None,
    ) -> tuple[list[dict], str]:
        """Find expenses that still have a refundable balance. If a strict match fails, widen the merchant or date and say why."""
        currency = currency.upper()
        amount_minor = to_minor(amount, currency)
        if amount_minor <= 0:
            raise ValueError("Enter an amount greater than 0.")
        with get_connection(self.db_path) as connection:
            rows = list_transactions(
                connection,
                currency=currency,
                transaction_type="expense",
            )
            candidates = []
            for row in rows:
                refunded = sum_refunds_for(
                    connection,
                    int(row["id"]),
                    exclude_id=exclude_refund_id,
                )
                remaining = int(row["amount_minor"]) - refunded
                if remaining < amount_minor:
                    continue
                item = decorate(row)
                item["refunded_minor"] = refunded
                item["remaining_minor"] = remaining
                item["remaining"] = format_amount(remaining, currency)
                candidates.append(item)
        note_parts: list[str] = []
        pool = candidates
        if merchant and merchant.strip():
            matched = [item for item in pool if _normalize(item.get("merchant")) == _normalize(merchant)]
            if matched:
                pool = matched
            else:
                note_parts.append("No expense with the same merchant was found, so the list uses amount and currency.")
        if start_date and end_date:
            start_text = start_date.isoformat()
            end_text = end_date.isoformat()
            ranged = [
                item for item in pool if start_text <= item["transaction_date"] <= end_text
            ]
            if ranged:
                pool = ranged
            else:
                note_parts.append("Nothing matched that date range, so the date filter was widened.")
        return pool, " ".join(note_parts)

    def set_budget(
        self,
        *,
        category_name: str | None,
        currency: str,
        amount: Decimal,
    ) -> int:
        """Set a monthly total budget or a budget for one expense category."""
        currency = currency.upper()
        limit_minor = to_minor(amount, currency)
        with get_connection(self.db_path) as connection:
            category_id = None
            if category_name:
                category = get_category_by_name(connection, category_name, "expense")
                if category is None:
                    raise ValueError("Budgets can only be set for an expense category.")
                category_id = int(category["id"])
            return upsert_budget(
                connection,
                category_id=category_id,
                currency=currency,
                monthly_limit_minor=limit_minor,
            )

    def delete_budget(self, budget_id: int) -> None:
        with get_connection(self.db_path) as connection:
            delete_budget(connection, budget_id)

    def list_budgets(self) -> list[dict]:
        with get_connection(self.db_path) as connection:
            rows = list_budgets(connection)
        for row in rows:
            row["limit"] = format_amount(int(row["monthly_limit_minor"]), row["currency"])
        return rows

    def budget_status(self, *, year: int, month: int) -> list[BudgetStatus]:
        """Net expense for the month of the transaction date, with 80% and 100% status."""
        start, end = _month_bounds(year, month)
        with get_connection(self.db_path) as connection:
            budgets = list_budgets(connection)
            rows = list_transactions(connection, start_date=start, end_date=end)
        statuses: list[BudgetStatus] = []
        for budget in budgets:
            currency = budget["currency"]
            category_name = budget.get("category_name")
            spent = net_spent_minor(rows, currency, category_name)
            limit_minor = int(budget["monthly_limit_minor"])
            if limit_minor == 0:
                ratio = Decimal("0") if spent <= 0 else Decimal("999")
            else:
                ratio = (Decimal(spent) / Decimal(limit_minor)).quantize(Decimal("0.0001"))
            if spent <= 0:
                level = "ok"
            elif limit_minor == 0 or ratio >= 1:
                level = "over"
            elif ratio >= Decimal("0.8"):
                level = "warn"
            else:
                level = "ok"
            statuses.append(
                BudgetStatus(
                    budget_id=int(budget["id"]),
                    currency=currency,
                    category_name=category_name,
                    limit_minor=limit_minor,
                    spent_minor=spent,
                    ratio=ratio,
                    level=level,
                )
            )
        return statuses

    def summarize(
        self,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        currency: str | None = None,
        category_id: int | None = None,
        transaction_type: str | None = None,
        keyword: str | None = None,
        frequency: str = "month",
    ) -> list[CurrencySummary]:
        """Summarize the rows in the database. Each currency in the filter is returned separately."""
        rows = self.list_transaction_rows(
            start_date=start_date,
            end_date=end_date,
            currency=currency,
            category_id=category_id,
            transaction_type=transaction_type,
            keyword=keyword,
        )
        currencies = []
        for row in rows:
            if row["currency"] not in currencies:
                currencies.append(row["currency"])
        return [self._summarize_currency(rows, code, frequency) for code in currencies]

    def export_frame(
        self,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        currency: str | None = None,
        category_id: int | None = None,
        transaction_type: str | None = None,
        keyword: str | None = None,
        include_deleted: bool = False,
    ) -> pd.DataFrame:
        """Build the export table. Dates are YYYY-MM-DD and amounts are fixed-point decimal strings."""
        rows = self.list_transaction_rows(
            start_date=start_date,
            end_date=end_date,
            currency=currency,
            category_id=category_id,
            transaction_type=transaction_type,
            keyword=keyword,
            include_deleted=include_deleted,
        )
        exported = []
        for row in rows:
            exported.append(
                {
                    "id": row["id"],
                    "date": row["transaction_date"],
                    "amount": format_amount(int(row["amount_minor"]), row["currency"]).replace(",", ""),
                    "currency": row["currency"],
                    "transaction_type": row["transaction_type"],
                    "type_label": row["type_label"],
                    "category": _excel_safe(row.get("category_name") or ""),
                    "description": _excel_safe(row.get("description") or ""),
                    "merchant": _excel_safe(row.get("merchant") or ""),
                    "account": _excel_safe(row.get("account_name") or ""),
                    "original_id": row.get("original_transaction_id") or "",
                    "source": row.get("source") or "",
                    "created_at": row.get("created_at") or "",
                    "updated_at": row.get("updated_at") or "",
                }
            )
        columns = [
            "id",
            "date",
            "amount",
            "currency",
            "transaction_type",
            "type_label",
            "category",
            "description",
            "merchant",
            "account",
            "original_id",
            "source",
            "created_at",
            "updated_at",
        ]
        frame = pd.DataFrame(exported, columns=columns)
        # Keep amount and date as text so pandas does not turn 120.50 into the float 120.5.
        for column in columns:
            frame[column] = frame[column].map(lambda value: "" if value is None else str(value))
        return frame

    def export_csv(self, **filters: object) -> bytes:
        """Export a UTF-8 BOM CSV so Excel opens it directly."""
        frame = self.export_frame(**filters)
        return frame.to_csv(index=False).encode("utf-8-sig")

    def export_excel(self, **filters: object) -> bytes:
        """Export xlsx. openpyxl writes the workbook."""
        frame = self.export_frame(**filters)
        buffer = BytesIO()
        frame.to_excel(buffer, index=False, engine="openpyxl")
        return buffer.getvalue()

    def _summarize_currency(self, rows: list[dict], currency: str, frequency: str) -> CurrencySummary:
        """Group one currency with pandas. The amount column stays in integer minor units."""
        selected = [row for row in rows if row["currency"] == currency]
        empty_buckets = {"income": 0, "expense": 0, "refund": 0}
        if not selected:
            return CurrencySummary(
                currency=currency,
                total_income_minor=0,
                total_expense_minor=0,
                total_refund_minor=0,
                net_expense_minor=0,
                balance_minor=0,
                category_net=[],
                periods=[],
                period_income=[],
                period_net_expense=[],
                recent=[],
            )
        frame = pd.DataFrame(selected)
        frame["amount_minor"] = frame["amount_minor"].astype("int64")
        frame["period"] = frame["transaction_date"].map(lambda value: _period_key(str(value), frequency))
        totals = frame.groupby("transaction_type")["amount_minor"].sum().to_dict()
        income = int(totals.get("income", 0))
        expense = int(totals.get("expense", 0))
        refund = int(totals.get("refund", 0))
        grouped = frame.groupby(["period", "transaction_type"])["amount_minor"].sum()
        ordered_periods = sorted(frame["period"].unique())
        period_income = []
        period_net = []
        for period in ordered_periods:
            values = empty_buckets.copy()
            if period in grouped.index.get_level_values(0):
                for tx_type in ("income", "expense", "refund"):
                    key = (period, tx_type)
                    if key in grouped.index:
                        values[tx_type] = int(grouped.loc[key])
            period_income.append(values["income"])
            period_net.append(values["expense"] - values["refund"])
        expense_rows = frame[frame["transaction_type"] == "expense"]
        refund_rows = frame[frame["transaction_type"] == "refund"].copy()
        if not refund_rows.empty:
            refund_rows["category_name"] = refund_rows["original_category_name"].fillna("Other")
        categories: dict[str, int] = {}
        if not expense_rows.empty:
            for name, value in expense_rows.groupby("category_name")["amount_minor"].sum().items():
                categories[str(name)] = categories.get(str(name), 0) + int(value)
        if not refund_rows.empty:
            for name, value in refund_rows.groupby("category_name")["amount_minor"].sum().items():
                categories[str(name)] = categories.get(str(name), 0) - int(value)
        category_net = sorted(categories.items(), key=lambda item: item[1], reverse=True)
        return CurrencySummary(
            currency=currency,
            total_income_minor=income,
            total_expense_minor=expense,
            total_refund_minor=refund,
            net_expense_minor=expense - refund,
            balance_minor=income - (expense - refund),
            category_net=category_net,
            periods=ordered_periods,
            period_income=period_income,
            period_net_expense=period_net,
            recent=selected[:8],
        )

    def _prepare(
        self,
        *,
        transaction_date: date,
        amount: Decimal,
        currency: str,
        transaction_type: str,
        category_name: str,
        description: str | None,
        merchant: str | None,
        account_name: str | None,
        original_transaction_id: int | None,
        source: str,
    ) -> dict:
        """Validate the input and build the fields the database needs."""
        if not isinstance(transaction_date, date):
            raise ValueError("The date is invalid.")
        if transaction_date < date(2000, 1, 1) or transaction_date > date(2100, 12, 31):
            raise ValueError("The date is outside the supported range.")
        code = currency.strip().upper()
        if len(code) != 3 or not code.isalpha():
            raise ValueError("The currency must be 3 letters.")
        if transaction_type not in _VALID_TYPES:
            raise ValueError("Invalid transaction type.")
        if source not in _VALID_SOURCES:
            raise ValueError("Invalid transaction source.")
        amount_minor = to_minor(Decimal(amount), code)
        if amount_minor <= 0:
            raise ValueError("Enter an amount greater than 0.")
        category = "Refund" if transaction_type == "refund" else category_name.strip()
        if transaction_type == "refund" and original_transaction_id is None:
            raise ValueError("A refund must point at the original expense.")
        if transaction_type != "refund":
            original_transaction_id = None
        return {
            "transaction_date": transaction_date.isoformat(),
            "amount_minor": amount_minor,
            "currency": code,
            "transaction_type": transaction_type,
            "category_name": category,
            "category_kind": "income" if transaction_type in {"income", "refund"} else "expense",
            "description": _blank_to_none(description),
            "merchant": _blank_to_none(merchant),
            "account_name": _blank_to_none(account_name),
            "original_transaction_id": original_transaction_id,
            "source": source,
        }

    def _resolve_links(self, connection, prepared: dict) -> tuple[int, int | None]:
        category = get_category_by_name(
            connection,
            prepared["category_name"],
            prepared["category_kind"],
        )
        if category is None:
            raise ValueError("That category does not apply to this transaction type.")
        account_id = None
        if prepared["account_name"]:
            account_id = get_or_create_account(
                connection,
                prepared["account_name"],
                prepared["currency"],
            )
        return int(category["id"]), account_id

    def _assert_refund_capacity(
        self,
        connection,
        original_id: int,
        amount_minor: int,
        currency: str,
        exclude_id: int | None = None,
    ) -> None:
        """Refunds cannot add up to more than the part of the original expense that is still unrefunded."""
        original = get_transaction(connection, original_id)
        if original is None or original["is_deleted"]:
            raise ValueError("The original transaction does not exist.")
        if original["transaction_type"] != "expense":
            raise ValueError("A refund can only be applied to an expense.")
        if original["currency"] != currency:
            raise ValueError("The refund currency must match the original transaction.")
        refunded = sum_refunds_for(connection, original_id, exclude_id=exclude_id)
        if amount_minor + refunded > int(original["amount_minor"]):
            raise ValueError("The refund is larger than the amount still available on the original expense.")

    def _assert_existing_refunds_still_fit(self, connection, current: dict, prepared: dict) -> None:
        """An expense that already has a refund cannot be made smaller than the refunded amount, or switched to another currency."""
        if current["transaction_type"] != "expense":
            return
        refunded = sum_refunds_for(connection, int(current["id"]))
        if refunded <= 0:
            return
        if (
            prepared["transaction_type"] != "expense"
            or prepared["currency"] != current["currency"]
            or prepared["amount_minor"] < refunded
        ):
            raise ValueError("This expense already has a refund. The edited amount must still cover it.")

    def _duplicates_on(self, connection, prepared: dict, exclude_id: int | None) -> DuplicateReport:
        """Compare date, amount, currency, merchant, and description with deterministic rules."""
        rows = list_transactions(
            connection,
            currency=prepared["currency"],
            transaction_type=prepared["transaction_type"],
        )
        exact: list[dict] = []
        suspected: list[dict] = []
        candidate_date = date.fromisoformat(prepared["transaction_date"])
        merchant = _normalize(prepared["merchant"])
        description = _normalize(prepared["description"])
        for row in rows:
            if exclude_id is not None and int(row["id"]) == exclude_id:
                continue
            if int(row["amount_minor"]) != prepared["amount_minor"]:
                continue
            row_merchant = _normalize(row.get("merchant"))
            row_description = _normalize(row.get("description"))
            same_day = row["transaction_date"] == prepared["transaction_date"]
            same_merchant = merchant == row_merchant
            same_description = description == row_description
            if same_day and same_merchant and same_description:
                exact.append(decorate(row))
                continue
            days_apart = abs((date.fromisoformat(row["transaction_date"]) - candidate_date).days)
            merchant_match = bool(merchant) and same_merchant
            description_close = same_day and _similar(description, row_description) >= 0.75
            near_day_same_merchant = merchant_match and days_apart <= 1
            if same_day and (merchant_match or description_close) or near_day_same_merchant:
                suspected.append(decorate(row))
        return DuplicateReport(exact=exact, suspected=suspected)
