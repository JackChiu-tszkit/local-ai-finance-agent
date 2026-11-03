"""财务业务规则。

金额换算、重复判断、退款上限、预算和统计都在这里用 Decimal 与整数完成。
语言模型不参与任何汇总计算。
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
    """存在相似历史交易，且调用方没有明确允许继续保存。"""

    def __init__(self, report: "DuplicateReport") -> None:
        super().__init__("发现相似交易，请确认后保存。")
        self.report = report


@dataclass
class DuplicateReport:
    """重复检查结果。exact 是完全相同，suspected 是同日同额且商户或描述接近。"""

    exact: list[dict] = field(default_factory=list)
    suspected: list[dict] = field(default_factory=list)

    @property
    def has_any(self) -> bool:
        return bool(self.exact or self.suspected)


@dataclass
class BudgetStatus:
    """某币种总预算或分类预算在指定月份的使用情况。"""

    budget_id: int
    currency: str
    category_name: str | None
    limit_minor: int
    spent_minor: int
    ratio: Decimal
    level: str


@dataclass
class CurrencySummary:
    """单一币种的统计结果。不同币种不会加总。"""

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
    """返回币种的小数位数。未知币种按分或欧分处理。"""
    return CURRENCY_EXPONENTS.get(currency.upper(), 2)


def to_minor(amount: Decimal, currency: str) -> int:
    """把 Decimal 金额转换为最小货币单位。负数会被拒绝。"""
    if not isinstance(amount, Decimal):
        amount = Decimal(str(amount))
    if amount < 0:
        raise ValueError("金额不能为负数。")
    exponent = minor_exponent(currency)
    quant = Decimal(1).scaleb(-exponent)
    quantized = amount.quantize(quant, rounding=ROUND_HALF_UP)
    factor = Decimal(10) ** exponent
    return int((quantized * factor).to_integral_value(rounding=ROUND_HALF_UP))


def from_minor(amount_minor: int, currency: str) -> Decimal:
    """把最小货币单位还原为 Decimal。"""
    exponent = minor_exponent(currency)
    quant = Decimal(1).scaleb(-exponent)
    return (Decimal(amount_minor) / (Decimal(10) ** exponent)).quantize(quant)


def format_amount(amount_minor: int, currency: str) -> str:
    """格式化金额，不附带币种代码。"""
    value = from_minor(amount_minor, currency)
    exponent = minor_exponent(currency)
    if exponent == 0:
        return f"{int(value):,}"
    return f"{value:,.{exponent}f}"


def format_money(amount_minor: int, currency: str) -> str:
    """格式化金额并附带币种代码。"""
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
    raise ValueError("统计周期只支持 day、week 或 month。")


def shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    """按公历月份平移。delta 为负表示往前。"""
    index = year * 12 + (month - 1) + delta
    return index // 12, index % 12 + 1


def _month_bounds(year: int, month: int) -> tuple[str, str]:
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1)
    else:
        end = date(year, month + 1, 1)
    # 结束日用月末，因为列表查询是闭区间。
    return start.isoformat(), (end.fromordinal(end.toordinal() - 1)).isoformat()


def decorate(row: dict) -> dict:
    """补充界面和导出使用的金额字符串。计算仍以 amount_minor 为准。"""
    item = dict(row)
    item["amount"] = format_amount(int(row["amount_minor"]), row["currency"])
    item["type_label"] = TRANSACTION_TYPE_LABELS.get(row["transaction_type"], row["transaction_type"])
    return item


def _excel_safe(value: object) -> object:
    """防止表格软件把以等号开头的描述当成公式。"""
    if isinstance(value, str) and value[:1] in {"=", "+", "-", "@"}:
        return "'" + value
    return value


def _has_currency_expense(rows: list[dict], currency: str) -> bool:
    return any(
        row["currency"] == currency and row["transaction_type"] == "expense" and not row.get("is_deleted")
        for row in rows
    )


def _spent_by_category(rows: list[dict], currency: str) -> dict[str, int]:
    """同一币种下各分类的净支出。退款冲减原交易分类。"""
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
    """返回商户比较键和最近一笔的展示名称。"""
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
    """计算净支出。退款冲减原交易分类，不计入工资或投资收益。"""
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
    """面向界面的财务操作入口。每次写入都在同一个数据库事务中完成。"""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        init_database(self.db_path)

    def merchant_memories(self) -> dict[str, str]:
        """返回商户比较键到分类名的映射。内置商户表不在这里。"""
        with get_connection(self.db_path) as connection:
            rows = list_merchant_memories(connection)
        return {row["merchant_key"]: row["category_name"] for row in rows}

    def remember_merchant(
        self,
        merchant: str | None,
        category_name: str,
        transaction_type: str,
    ) -> None:
        """用户确认支出后记住商户分类。没有商户、或不是支出时不写。"""
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
        """把预览中的支出改到新分类。调用前必须已经让用户确认。"""
        if category_name not in EXPENSE_CATEGORIES:
            raise ValueError("只能把支出改到已有的支出分类。")
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
        """根据本人历史给出提醒。比较和阈值都在这里完成，不交给模型。"""
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
                            f"本月{name}净支出 {format_money(spent, currency)}，"
                            f"高于前三个月平均 {format_money(int(average), currency)}。"
                            "这是根据你自己的历史记录计算的提醒。"
                        )
            other_count = sum(
                1
                for row in current_rows
                if row["currency"] == currency
                and row["transaction_type"] == "expense"
                and row.get("category_name") == "其他"
            )
            if other_count >= 3:
                notes.append(f"本月有 {other_count} 笔 {currency} 支出归在「其他」，可以让记账助手帮你归类。")
            current_merchants = _merchant_display(current_rows, currency)
            for key, display in current_merchants.items():
                hits = sum(1 for rows in previous if key in _merchant_display(rows, currency))
                if hits >= 2:
                    notes.append(
                        f"{display} 在本月以及之前三个月里的至少两个月都有支出。"
                        "这只是重复出现的提醒，不会自动记账。"
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
        """按筛选条件读取交易，并附上可显示的金额。"""
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
        """确认后的交易写入数据库。存在相似记录时必须显式允许。"""
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
                raise ValueError("只有退款可以关联原交易。")
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
        """修改未删除的交易，并重新检查退款上限和重复。"""
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
                raise KeyError("交易不存在或已删除。")
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
        """软删除。仍有有效退款的支出不能删除，历史退款也不会被自动清掉。"""
        if not confirmed:
            raise ValueError("删除前需要确认。")
        with get_connection(self.db_path) as connection:
            current = get_transaction(connection, transaction_id)
            if current is None or current["is_deleted"]:
                raise KeyError("交易不存在或已删除。")
            if current["transaction_type"] == "expense" and sum_refunds_for(connection, transaction_id) > 0:
                raise ValueError("这笔支出仍有关联退款，不能删除。")
            soft_delete_transaction(connection, transaction_id)
        logger.info("transaction soft-deleted id=%s", transaction_id)

    def restore_transaction(self, transaction_id: int) -> None:
        """恢复软删除记录。恢复退款时重新检查是否超过原交易余额。"""
        with get_connection(self.db_path) as connection:
            current = get_transaction(connection, transaction_id)
            if current is None or not current["is_deleted"]:
                raise KeyError("交易不存在或并未删除。")
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
        """供确认界面提前展示相似交易。真正写入时会再检查一次。"""
        code = currency.strip().upper()
        if transaction_type not in _VALID_TYPES:
            raise ValueError("交易类型无效。")
        amount_minor = to_minor(Decimal(amount), code)
        if amount_minor <= 0:
            raise ValueError("请填写大于 0 的金额。")
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
        """查找仍有可退余额的支出。没有严格匹配时放宽商户或日期，并说明原因。"""
        currency = currency.upper()
        amount_minor = to_minor(amount, currency)
        if amount_minor <= 0:
            raise ValueError("请填写大于 0 的金额。")
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
                note_parts.append("没有商户一致的支出，已改为按金额和币种列出。")
        if start_date and end_date:
            start_text = start_date.isoformat()
            end_text = end_date.isoformat()
            ranged = [
                item for item in pool if start_text <= item["transaction_date"] <= end_text
            ]
            if ranged:
                pool = ranged
            else:
                note_parts.append("指定日期范围内没有匹配，已放宽日期。")
        return pool, " ".join(note_parts)

    def set_budget(
        self,
        *,
        category_name: str | None,
        currency: str,
        amount: Decimal,
    ) -> int:
        """设置月度总支出预算或某个支出分类的预算。"""
        currency = currency.upper()
        limit_minor = to_minor(amount, currency)
        with get_connection(self.db_path) as connection:
            category_id = None
            if category_name:
                category = get_category_by_name(connection, category_name, "expense")
                if category is None:
                    raise ValueError("只能为支出分类设置预算。")
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
        """按入账日期计算当月净支出，并给出 80% 和 100% 两档状态。"""
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
        """用数据库中的真实记录汇总。筛选到多种币种时分别返回。"""
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
        """生成导出表。日期使用 YYYY-MM-DD，金额使用定点十进制字符串。"""
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
                    "交易编号": row["id"],
                    "日期": row["transaction_date"],
                    "金额": format_amount(int(row["amount_minor"]), row["currency"]).replace(",", ""),
                    "币种": row["currency"],
                    "交易类型": row["transaction_type"],
                    "类型名称": row["type_label"],
                    "分类": _excel_safe(row.get("category_name") or ""),
                    "描述": _excel_safe(row.get("description") or ""),
                    "商户": _excel_safe(row.get("merchant") or ""),
                    "账户": _excel_safe(row.get("account_name") or ""),
                    "原交易编号": row.get("original_transaction_id") or "",
                    "来源": row.get("source") or "",
                    "创建时间": row.get("created_at") or "",
                    "更新时间": row.get("updated_at") or "",
                }
            )
        columns = [
            "交易编号",
            "日期",
            "金额",
            "币种",
            "交易类型",
            "类型名称",
            "分类",
            "描述",
            "商户",
            "账户",
            "原交易编号",
            "来源",
            "创建时间",
            "更新时间",
        ]
        frame = pd.DataFrame(exported, columns=columns)
        # 金额和日期保持文本，避免 pandas 把 120.50 收成浮点 120.5。
        for column in columns:
            frame[column] = frame[column].map(lambda value: "" if value is None else str(value))
        return frame

    def export_csv(self, **filters: object) -> bytes:
        """导出 UTF-8 BOM CSV，方便 Excel 直接打开中文。"""
        frame = self.export_frame(**filters)
        return frame.to_csv(index=False).encode("utf-8-sig")

    def export_excel(self, **filters: object) -> bytes:
        """导出 xlsx。openpyxl 负责写入工作簿。"""
        frame = self.export_frame(**filters)
        buffer = BytesIO()
        frame.to_excel(buffer, index=False, engine="openpyxl")
        return buffer.getvalue()

    def _summarize_currency(self, rows: list[dict], currency: str, frequency: str) -> CurrencySummary:
        """用 pandas 对同一币种做分组汇总，金额列保持整数最小单位。"""
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
            refund_rows["category_name"] = refund_rows["original_category_name"].fillna("其他")
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
        """统一校验并生成数据库需要的字段。"""
        if not isinstance(transaction_date, date):
            raise ValueError("日期无效。")
        if transaction_date < date(2000, 1, 1) or transaction_date > date(2100, 12, 31):
            raise ValueError("日期超出可记录范围。")
        code = currency.strip().upper()
        if len(code) != 3 or not code.isalpha():
            raise ValueError("币种必须是 3 位字母。")
        if transaction_type not in _VALID_TYPES:
            raise ValueError("交易类型无效。")
        if source not in _VALID_SOURCES:
            raise ValueError("交易来源无效。")
        amount_minor = to_minor(Decimal(amount), code)
        if amount_minor <= 0:
            raise ValueError("请填写大于 0 的金额。")
        category = "退款" if transaction_type == "refund" else category_name.strip()
        if transaction_type == "refund" and original_transaction_id is None:
            raise ValueError("退款必须选择原始支出。")
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
            raise ValueError("分类不适用于该交易类型。")
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
        """退款累计不能超过原支出尚未退还的部分。"""
        original = get_transaction(connection, original_id)
        if original is None or original["is_deleted"]:
            raise ValueError("原交易不存在。")
        if original["transaction_type"] != "expense":
            raise ValueError("只能对支出办理退款。")
        if original["currency"] != currency:
            raise ValueError("退款币种必须与原交易相同。")
        refunded = sum_refunds_for(connection, original_id, exclude_id=exclude_id)
        if amount_minor + refunded > int(original["amount_minor"]):
            raise ValueError("退款金额超过原交易尚未退还的金额。")

    def _assert_existing_refunds_still_fit(self, connection, current: dict, prepared: dict) -> None:
        """已有退款的支出不能被改得比已退金额更小，也不能改成别的币种。"""
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
            raise ValueError("这笔支出已有退款，修改后必须仍能覆盖已退金额。")

    def _duplicates_on(self, connection, prepared: dict, exclude_id: int | None) -> DuplicateReport:
        """用日期、金额、币种、商户和描述做确定性比对。"""
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
