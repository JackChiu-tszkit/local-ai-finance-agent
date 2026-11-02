"""金额、去重、退款、预算、统计和导出测试。"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from io import BytesIO
from pathlib import Path

import pytest
from openpyxl import load_workbook

from finance_service import (
    DuplicateTransactionError,
    FinanceService,
    from_minor,
    to_minor,
)


def make_service(tmp_path: Path) -> FinanceService:
    return FinanceService(tmp_path / "finance.db")


def add(
    service: FinanceService,
    *,
    amount: str = "100",
    currency: str = "CNY",
    transaction_type: str = "expense",
    category_name: str = "餐饮",
    description: str = "火锅",
    merchant: str | None = None,
    transaction_date: date = date(2026, 11, 2),
    original_transaction_id: int | None = None,
    allow_duplicate: bool = False,
    source: str = "manual",
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
        source=source,
        allow_duplicate=allow_duplicate,
    )


def test_money_rounding_avoids_binary_float() -> None:
    assert to_minor(Decimal("10.005"), "CNY") == 1001
    assert to_minor(Decimal("10.004"), "CNY") == 1000
    assert to_minor(Decimal("120.50"), "NOK") == 12050
    assert from_minor(12050, "NOK") == Decimal("120.50")
    assert to_minor(Decimal("10.6"), "JPY") == 11
    with pytest.raises(ValueError):
        to_minor(Decimal("-1"), "CNY")


def test_exact_and_suspected_duplicates(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    add(service, amount="350", currency="NOK", category_name="食品杂货", description="买菜", merchant="Kiwi")
    with pytest.raises(DuplicateTransactionError) as caught:
        add(service, amount="350", currency="NOK", category_name="食品杂货", description="买菜", merchant="Kiwi")
    assert caught.value.report.exact
    add(
        service,
        amount="350",
        currency="NOK",
        category_name="食品杂货",
        description="买菜",
        merchant="Kiwi",
        allow_duplicate=True,
    )
    assert len(service.list_transaction_rows()) == 2

    with pytest.raises(DuplicateTransactionError) as suspected:
        add(
            service,
            amount="350",
            currency="NOK",
            category_name="食品杂货",
            description="买水果",
            merchant="Kiwi",
        )
    assert suspected.value.report.suspected
    assert not suspected.value.report.exact

    add(service, amount="80", description="火锅午餐", merchant=None)
    add(service, amount="80", description="地铁车票", merchant=None)
    assert len(service.list_transaction_rows(keyword="火锅")) == 1


def test_near_date_same_merchant_is_suspected(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    add(
        service,
        amount="40",
        currency="NOK",
        transaction_date=date(2026, 11, 1),
        merchant="Kiwi",
        description="苹果",
        category_name="食品杂货",
    )
    with pytest.raises(DuplicateTransactionError) as caught:
        add(
            service,
            amount="40",
            currency="NOK",
            transaction_date=date(2026, 11, 2),
            merchant="Kiwi",
            description="香蕉",
            category_name="食品杂货",
        )
    assert caught.value.report.suspected
    assert not caught.value.report.exact


def test_deleted_transaction_does_not_block_new_entry(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    transaction_id = add(service, description="一次性午餐")
    service.delete_transaction(transaction_id, confirmed=True)
    add(service, description="一次性午餐")
    assert len(service.list_transaction_rows()) == 1


def test_refund_reduces_net_expense_and_cannot_exceed_original(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    original_id = add(service, amount="100", description="火锅", transaction_date=date(2026, 11, 1))
    candidates, _note = service.find_refund_candidates(
        amount=Decimal("40"),
        currency="CNY",
        start_date=date(2026, 11, 1),
        end_date=date(2026, 11, 1),
    )
    assert [item["id"] for item in candidates] == [original_id]
    add(
        service,
        amount="40",
        transaction_type="refund",
        category_name="退款",
        description="火锅退款",
        original_transaction_id=original_id,
        transaction_date=date(2026, 11, 2),
    )
    add(
        service,
        amount="60",
        transaction_type="refund",
        category_name="工资",
        description="剩余退款",
        original_transaction_id=original_id,
        transaction_date=date(2026, 11, 2),
    )
    with pytest.raises(ValueError, match="尚未退还"):
        add(
            service,
            amount="1",
            transaction_type="refund",
            category_name="退款",
            description="超额退款",
            original_transaction_id=original_id,
        )
    summary = service.summarize(
        start_date=date(2026, 11, 1),
        end_date=date(2026, 11, 2),
        currency="CNY",
        frequency="month",
    )[0]
    assert summary.total_expense_minor == 10000
    assert summary.total_refund_minor == 10000
    assert summary.net_expense_minor == 0
    assert summary.total_income_minor == 0
    assert summary.category_net == [("餐饮", 0)]


def test_refund_is_not_income_and_currencies_stay_separate(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    original_id = add(service, amount="50", currency="CNY")
    add(
        service,
        amount="45000",
        currency="NOK",
        transaction_type="income",
        category_name="工资",
        description="工资",
    )
    add(
        service,
        amount="20",
        currency="CNY",
        transaction_type="refund",
        category_name="退款",
        description="部分退款",
        original_transaction_id=original_id,
    )
    summaries = {
        item.currency: item
        for item in service.summarize(start_date=date(2026, 11, 1), end_date=date(2026, 11, 30), frequency="month")
    }
    assert summaries["CNY"].total_income_minor == 0
    assert summaries["CNY"].net_expense_minor == 3000
    assert summaries["CNY"].balance_minor == -3000
    assert summaries["NOK"].total_income_minor == 4500000
    assert summaries["NOK"].total_expense_minor == 0
    income_id = add(service, amount="10", transaction_type="income", category_name="其他收入", description="红包")
    with pytest.raises(ValueError, match="只能对支出"):
        add(
            service,
            amount="10",
            transaction_type="refund",
            category_name="退款",
            description="不能退收入",
            original_transaction_id=income_id,
        )


def test_refund_blocks_delete_and_shrinking_original(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    original_id = add(service, amount="50", description="订单")
    refund_id = add(
        service,
        amount="20",
        transaction_type="refund",
        category_name="退款",
        description="退回一部分",
        original_transaction_id=original_id,
    )
    with pytest.raises(ValueError, match="关联退款"):
        service.delete_transaction(original_id, confirmed=True)
    with pytest.raises(ValueError, match="确认"):
        service.delete_transaction(refund_id, confirmed=False)
    with pytest.raises(ValueError, match="已有退款"):
        service.update_transaction(
            original_id,
            transaction_date=date(2026, 11, 2),
            amount=Decimal("10"),
            currency="CNY",
            transaction_type="expense",
            category_name="餐饮",
            description="订单",
            merchant=None,
            account_name=None,
        )
    service.delete_transaction(refund_id, confirmed=True)
    service.delete_transaction(original_id, confirmed=True)
    assert service.list_transaction_rows() == []


def test_budget_thresholds_and_refund_relief(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.set_budget(category_name=None, currency="CNY", amount=Decimal("100"))
    service.set_budget(category_name="餐饮", currency="CNY", amount=Decimal("100"))
    service.set_budget(category_name=None, currency="NOK", amount=Decimal("50"))
    add(service, amount="79")
    levels = { (item.currency, item.category_name): item.level for item in service.budget_status(year=2026, month=11) }
    assert levels[("CNY", None)] == "ok"
    add(service, amount="1", description="再花一点")
    levels = { (item.currency, item.category_name): item.level for item in service.budget_status(year=2026, month=11) }
    assert levels[("CNY", None)] == "warn"
    assert levels[("CNY", "餐饮")] == "warn"
    add(service, amount="20", description="继续吃", category_name="餐饮")
    levels = { (item.currency, item.category_name): item.level for item in service.budget_status(year=2026, month=11) }
    assert levels[("CNY", None)] == "over"
    original_id = service.list_transaction_rows()[0]["id"]
    add(
        service,
        amount="20",
        transaction_type="refund",
        category_name="退款",
        description="退款缓解预算",
        original_transaction_id=int(original_id),
    )
    levels = { (item.currency, item.category_name): item.level for item in service.budget_status(year=2026, month=11) }
    assert levels[("CNY", None)] == "warn"
    add(service, amount="30", currency="NOK", category_name="食品杂货", description="Kiwi", merchant="Kiwi")
    levels = { (item.currency, item.category_name): item.level for item in service.budget_status(year=2026, month=11) }
    assert levels[("NOK", None)] == "ok"
    with pytest.raises(ValueError, match="支出分类"):
        service.set_budget(category_name="工资", currency="CNY", amount=Decimal("10"))


def test_october_expense_is_not_added_to_november_budget(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.set_budget(category_name=None, currency="CNY", amount=Decimal("100"))
    original_id = add(service, amount="100", transaction_date=date(2026, 10, 31), description="上月晚餐")
    add(
        service,
        amount="20",
        transaction_type="refund",
        category_name="退款",
        description="本月退款",
        original_transaction_id=original_id,
        transaction_date=date(2026, 11, 2),
    )
    november = service.budget_status(year=2026, month=11)[0]
    october = service.budget_status(year=2026, month=10)[0]
    assert november.spent_minor == -2000
    assert november.level == "ok"
    assert october.spent_minor == 10000
    assert october.level == "over"


def test_csv_and_excel_export(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    add(service, amount="120.50", description="=不要执行", merchant="Kiwi", currency="NOK", category_name="食品杂货")
    frame = service.export_frame()
    assert list(frame.columns)[:4] == ["交易编号", "日期", "金额", "币种"]
    assert frame.loc[0, "日期"] == "2026-11-02"
    assert frame.loc[0, "金额"] == "120.50"
    assert str(frame.loc[0, "描述"]).startswith("'")
    csv_text = service.export_csv().decode("utf-8-sig")
    assert "2026-11-02" in csv_text
    assert "120.50" in csv_text
    workbook = load_workbook(BytesIO(service.export_excel()))
    sheet = workbook.active
    headers = [cell.value for cell in next(sheet.iter_rows(max_row=1))]
    assert "日期" in headers
    assert "金额" in headers
    values = [cell.value for row in sheet.iter_rows(min_row=2, max_row=2) for cell in row]
    assert "2026-11-02" in values
    assert "120.50" in values
