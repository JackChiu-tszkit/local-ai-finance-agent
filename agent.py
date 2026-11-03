"""本地记账助手。

助手可以连续修改待确认草稿、把一句话拆成多笔，并用数据库回答账本问题。
金额、笔数、预算和退款余额都由财务服务计算。这里不会直接写入交易。
批量改分类只生成预览，必须由用户再确认一次。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from ai_parser import (
    TransactionDraft,
    _mark_clarification,
    apply_merchant_memory,
    extract_amounts,
    parse_with_rules,
    resolve_date,
)
from config import EXPENSE_CATEGORIES, INCOME_CATEGORIES, MERCHANT_CATEGORIES, TRANSACTION_TYPE_LABELS, merchant_key
from finance_service import FinanceService, format_money, from_minor, net_spent_minor, shift_month, to_minor


_QUESTION = re.compile(r"多少|几笔|预算|还能退|相比|对比|结余|比上个月|和上个月")
_AMEND = re.compile(r"改成|改为|换成|商户是|商家是|商户改|商家改|分类改|金额改|日期改|币种改|描述改|这是收入|这是支出|这是退款")
_BATCH = re.compile(r"把(?P<body>.+?)(?:都归到|都改成|全部改成|都算作|都归为)(?P<category>.+)$")
_DATE_HINT = re.compile(
    r"今天|昨天|昨日|前天|上周|上星期|上上周|本周|这周|周[一二三四五六日天]|\d{4}[-/.年]|\d{1,2}月\d{1,2}"
)
_BOOKING = re.compile(r"花了|花费|支付|消费|买|收到|工资|退款|退回|记一笔|spent|paid|brukte|betalte")
_CURRENCIES = (
    ("挪威克朗", "NOK"),
    ("人民币", "CNY"),
    ("克朗", "NOK"),
    ("kroner", "NOK"),
    ("美元", "USD"),
    ("欧元", "EUR"),
    ("英镑", "GBP"),
    ("日元", "JPY"),
    ("NOK", "NOK"),
    ("CNY", "CNY"),
    ("USD", "USD"),
    ("EUR", "EUR"),
    ("GBP", "GBP"),
    ("JPY", "JPY"),
    ("元", "CNY"),
    ("块", "CNY"),
)


@dataclass
class CategoryProposal:
    """尚未执行的分类修改。transaction_ids 来自当前数据库查询。"""

    category_name: str
    transaction_ids: list[int]
    preview: list[dict]
    summary: str


@dataclass
class AgentResult:
    """一轮对话的结果。new_drafts 和 amended_draft 都还不是账本记录。"""

    reply: str
    intent: str
    new_drafts: list[TransactionDraft] = field(default_factory=list)
    amended_draft: TransactionDraft | None = None
    proposal: CategoryProposal | None = None


def handle_user_message(
    text: str,
    *,
    today: date,
    default_currency: str,
    service: FinanceService,
    pending: TransactionDraft | None,
    memories: dict[str, str],
) -> AgentResult:
    """判断这句话是在改草稿、提问、预览改分类，还是新的一笔账。"""
    cleaned = text.strip()
    if not cleaned:
        return AgentResult("请说一笔交易，或问这个月花了多少。", "empty")

    batch = _BATCH.search(cleaned)
    if batch:
        return _propose_recategory(cleaned, batch, today, service)

    if pending is not None and (_AMEND.search(cleaned) or _looks_like_field_update(cleaned, today)):
        amended = amend_draft(pending, cleaned, today=today, default_currency=default_currency, memories=memories)
        if _same_draft(pending, amended):
            pending_date = next((item for item in amended.warnings if "日期还没改" in item), None)
            if pending_date:
                return AgentResult(pending_date, "amend")
            return AgentResult(
                "没有识别出要修改的字段。可以说「金额改成120元」「商户是 Kiwi」或「改成上周三」。",
                "amend",
            )
        return AgentResult(f"已修改当前草稿：{describe_draft(amended)}", "amend", amended_draft=amended)

    if _QUESTION.search(cleaned) and not _BOOKING.search(cleaned):
        return AgentResult(answer_ledger_question(service, cleaned, today), "question")
    if _QUESTION.search(cleaned) and "多少" in cleaned and not extract_amounts(cleaned):
        return AgentResult(answer_ledger_question(service, cleaned, today), "question")

    if _BOOKING.search(cleaned) or extract_amounts(cleaned) or "记" in cleaned:
        drafts = draft_transactions(cleaned, today=today, default_currency=default_currency, memories=memories)
        if len(drafts) == 1:
            reply = f"已生成待确认草稿：{describe_draft(drafts[0])}"
        else:
            reply = f"已拆成 {len(drafts)} 笔待确认草稿。请先确认第一笔：{describe_draft(drafts[0])}"
        return AgentResult(reply, "draft", new_drafts=drafts)

    if pending is not None:
        return AgentResult(
            "还没理解这句修改。可以说「金额改成120元」「商户是 Kiwi」或「改成上周三」。",
            "unknown",
        )
    return AgentResult(
        "我可以帮你记账、修改待确认草稿，或根据本机账本回答支出、预算和退款余额。例如：「午饭花了80元，地铁花了30元」。",
        "help",
    )


def draft_transactions(
    text: str,
    *,
    today: date,
    default_currency: str,
    memories: dict[str, str],
) -> list[TransactionDraft]:
    """把一句话拆成多笔草稿。只有每一段都有金额时才拆开。"""
    parts = [part.strip() for part in re.split(r"[，,；;。]|然后|还有|另外", text) if part.strip()]
    if len(parts) >= 2:
        parsed = [
            _finish_draft(part, today=today, default_currency=default_currency, memories=memories)
            for part in parts
        ]
        if sum(item.amount is not None for item in parsed) >= 2:
            return parsed
    return [_finish_draft(text, today=today, default_currency=default_currency, memories=memories)]


def amend_draft(
    draft: TransactionDraft,
    text: str,
    *,
    today: date,
    default_currency: str,
    memories: dict[str, str],
) -> TransactionDraft:
    """按用户补充修改当前草稿。没提到的字段保持原值。"""
    updated = draft.model_copy(deep=True)
    category_explicit = False
    merchant_explicit = False

    merchant_match = re.search(r"(?:商户|商家|店名)(?:是|为|改成|改为|换成)\s*([^\s，,。；;]{1,40})", text)
    if merchant_match:
        updated.merchant = " ".join(merchant_match.group(1).split())
        merchant_explicit = True

    category_name = _category_in_text(text)
    if category_name and re.search(r"分类|改成|改为|换成", text):
        updated.category = category_name
        category_explicit = True
        if category_name in INCOME_CATEGORIES and category_name not in EXPENSE_CATEGORIES:
            updated.transaction_type = "refund" if category_name == "退款" else "income"
            updated.type_confident = True

    type_match = re.search(r"(?:这是|改成|改为|换成)(收入|支出|退款)", text)
    if type_match:
        mapping = {"收入": "income", "支出": "expense", "退款": "refund"}
        updated.transaction_type = mapping[type_match.group(1)]
        updated.type_confident = True
        if updated.transaction_type == "refund":
            updated.category = "退款"
            category_explicit = True
        elif updated.transaction_type == "income" and updated.category not in INCOME_CATEGORIES:
            updated.category = "其他收入"
        elif updated.transaction_type == "expense" and updated.category not in EXPENSE_CATEGORIES:
            updated.category = "其他"

    description_match = re.search(r"描述(?:改成|改为|换成|是)\s*(.+)$", text)
    if description_match:
        updated.description = description_match.group(1).strip()[:500]

    if re.search(r"金额|改成|改为|换成", text) or _amount_only(text):
        hits = extract_amounts(text)
        if len(hits) == 1:
            updated.amount = hits[0].amount
            updated.possible_amounts = []
            if hits[0].currency:
                updated.currency = hits[0].currency
                updated.currency_assumed = False
            elif updated.currency is None:
                updated.currency = default_currency.upper()
                updated.currency_assumed = True
        elif len(hits) > 1:
            updated.amount = None
            updated.possible_amounts = []
            for hit in hits:
                if hit.amount not in updated.possible_amounts:
                    updated.possible_amounts.append(hit.amount)

    if _DATE_HINT.search(text):
        resolved, _assumed, explicit, _start, _end = resolve_date(text, today)
        if explicit and resolved is not None:
            updated.transaction_date = resolved
            updated.date_assumed = False
            updated.date_explicit = True
        elif "上周" in text or "上星期" in text:
            updated.warnings.append("上周需要具体到某一天，日期还没改。")

    currency_code = _currency_in_text(text)
    if currency_code and re.search(r"币种|货币|改成|改为|换成", text):
        updated.currency = currency_code
        updated.currency_assumed = False

    if merchant_explicit and not category_explicit and updated.transaction_type == "expense" and updated.merchant:
        learned = memories.get(merchant_key(updated.merchant))
        builtin = MERCHANT_CATEGORIES.get(updated.merchant)
        if learned in EXPENSE_CATEGORIES:
            updated.category = learned
        elif builtin:
            updated.category = builtin

    updated.parser = "agent"
    _drop_stale_warnings(updated)
    updated = _mark_clarification(updated)
    if not category_explicit:
        updated = apply_merchant_memory(updated, memories)
    return updated


def answer_ledger_question(service: FinanceService, text: str, today: date) -> str:
    """用数据库里的数字回答。找不到记录时直接说明，不估算。"""
    year, month = mentioned_month(text, today)
    currency = _currency_in_text(text)
    if "预算" in text:
        return _answer_budget(service, text, year, month)
    if "还能退" in text or ("退" in text and "多少" in text):
        return _answer_refund_remaining(service, text, currency)
    if re.search(r"相比|对比|比上个月|和上个月", text):
        return _answer_comparison(service, text, year, month, currency)
    return _answer_summary(service, text, year, month, currency)


def describe_draft(draft: TransactionDraft) -> str:
    """把草稿说成一句可以核对的中文。"""
    when = draft.transaction_date.isoformat() if draft.transaction_date else "日期待补充"
    if draft.amount is not None and draft.currency:
        money = format_money(to_minor(draft.amount, draft.currency), draft.currency)
    else:
        money = "金额待补充"
    label = TRANSACTION_TYPE_LABELS.get(draft.transaction_type or "", "支出")
    category = draft.category or "未分类"
    merchant = f"，商户 {draft.merchant}" if draft.merchant else ""
    question = f"。{draft.clarification_question}" if draft.clarification_question else ""
    return f"{when} {label} {money} {category}{merchant}{question}"


def mentioned_month(text: str, today: date) -> tuple[int, int]:
    """从问题里确定统计月份。没写月份时用今天所在的月。"""
    if "上个月" in text or "上月" in text:
        return shift_month(today.year, today.month, -1)
    if "这个月" not in text and "本月" not in text:
        match = re.search(r"(?<!\d)(\d{1,2})月", text)
        if match:
            month = int(match.group(1))
            if 1 <= month <= 12:
                year = today.year if month <= today.month else today.year - 1
                return year, month
    return today.year, today.month


def _finish_draft(
    text: str,
    *,
    today: date,
    default_currency: str,
    memories: dict[str, str],
) -> TransactionDraft:
    draft = parse_with_rules(text, today=today, default_currency=default_currency)
    draft.parser = "agent"
    return apply_merchant_memory(draft, memories)


def _propose_recategory(text: str, match: re.Match[str], today: date, service: FinanceService) -> AgentResult:
    category_name = _category_in_text(match.group("category"))
    if category_name not in EXPENSE_CATEGORIES:
        return AgentResult("请使用已有的支出分类，例如交通、餐饮或食品杂货。这次没有修改任何记录。", "batch")
    body = match.group("body")
    if not re.search(r"上个月|上月|这个月|本月|(?<!\d)\d{1,2}月", body):
        return AgentResult("请说明要改哪一个月，例如「把上个月打车都归到交通」。这次没有修改任何记录。", "batch")
    year, month = mentioned_month(body, today)
    keyword = re.sub(r"上个月|上月|这个月|本月|\d{1,2}月", "", body).strip(" 的")
    if not keyword:
        return AgentResult("请加上商户或关键词，例如打车或 Uber。这次没有修改任何记录。", "batch")
    start, end = _month_bounds_dates(year, month)
    rows = service.list_transaction_rows(
        start_date=start,
        end_date=end,
        transaction_type="expense",
        keyword=keyword,
    )
    selected = [row for row in rows if row.get("category_name") != category_name]
    if not selected:
        return AgentResult(
            f"{year}年{month}月没有找到仍需改成{category_name}的「{keyword}」支出。",
            "batch",
        )
    preview = [
        {
            "编号": row["id"],
            "日期": row["transaction_date"],
            "金额": row["amount"],
            "币种": row["currency"],
            "商户": row.get("merchant") or "",
            "原分类": row.get("category_name") or "",
        }
        for row in selected[:20]
    ]
    proposal = CategoryProposal(
        category_name=category_name,
        transaction_ids=[int(row["id"]) for row in selected],
        preview=preview,
        summary=(
            f"准备把 {year}年{month}月 {len(selected)} 笔「{keyword}」支出改成{category_name}。"
            "确认前不会写入数据库。"
        ),
    )
    return AgentResult(proposal.summary, "batch", proposal=proposal)


def _answer_budget(service: FinanceService, text: str, year: int, month: int) -> str:
    statuses = service.budget_status(year=year, month=month)
    category_name = _category_in_text(text)
    if category_name:
        statuses = [item for item in statuses if item.category_name == category_name]
    elif "总" not in text:
        pass
    if not statuses:
        return f"{year}年{month}月还没有对应的预算。"
    lines = [f"{year}年{month}月预算："]
    for status in statuses:
        name = status.category_name or "总支出"
        spent = format_money(status.spent_minor, status.currency)
        limit = format_money(status.limit_minor, status.currency)
        if status.limit_minor <= 0:
            lines.append(f"{name}（{status.currency}）已支出 {spent}，预算为 {limit}。")
            continue
        percent = (status.ratio * Decimal("100")).quantize(Decimal("0.1"))
        lines.append(f"{name}（{status.currency}）已用 {percent}%，支出 {spent} / 预算 {limit}。")
    return "\n".join(lines)


def _answer_refund_remaining(service: FinanceService, text: str, currency: str | None) -> str:
    keyword = _question_keyword(text)
    expenses = service.list_transaction_rows(transaction_type="expense", currency=currency)
    refunds = service.list_transaction_rows(transaction_type="refund", currency=currency)
    refunded: dict[int, int] = {}
    for row in refunds:
        original_id = row.get("original_transaction_id")
        if original_id:
            refunded[int(original_id)] = refunded.get(int(original_id), 0) + int(row["amount_minor"])
    lines = []
    for row in expenses:
        haystack = f"{row.get('merchant') or ''} {row.get('description') or ''}"
        if keyword and keyword.casefold() not in haystack.casefold():
            continue
        remaining = int(row["amount_minor"]) - refunded.get(int(row["id"]), 0)
        if remaining <= 0:
            continue
        label = row.get("merchant") or row.get("description") or f"#{row['id']}"
        lines.append(
            f"#{row['id']} {row['transaction_date']} {label} 还可退 "
            f"{format_money(remaining, row['currency'])}。"
        )
        if len(lines) >= 8:
            break
    if not lines:
        target = f"「{keyword}」" if keyword else "当前"
        return f"没有找到{target}仍可退款的支出。"
    return "\n".join(lines)


def _answer_comparison(
    service: FinanceService,
    text: str,
    year: int,
    month: int,
    currency: str | None,
) -> str:
    category_name = _category_in_text(text)
    previous_year, previous_month = shift_month(year, month, -1)
    current_rows = service.list_transaction_rows(
        start_date=_month_bounds_dates(year, month)[0],
        end_date=_month_bounds_dates(year, month)[1],
        currency=currency,
    )
    previous_rows = service.list_transaction_rows(
        start_date=_month_bounds_dates(previous_year, previous_month)[0],
        end_date=_month_bounds_dates(previous_year, previous_month)[1],
        currency=currency,
    )
    currencies = _currencies_in(current_rows + previous_rows, currency)
    if not currencies:
        return f"{year}年{month}月和{previous_year}年{previous_month}月都没有可比较的交易。"
    subject = category_name or "净支出"
    lines = []
    for code in currencies:
        current_minor = net_spent_minor(current_rows, code, category_name)
        previous_minor = net_spent_minor(previous_rows, code, category_name)
        delta = from_minor(current_minor, code) - from_minor(previous_minor, code)
        lines.append(
            f"{year}年{month}月{subject} {format_money(current_minor, code)}，"
            f"{previous_year}年{previous_month}月为 {format_money(previous_minor, code)}，"
            f"相差 {delta} {code}。"
        )
    return "\n".join(lines)


def _answer_summary(service: FinanceService, text: str, year: int, month: int, currency: str | None) -> str:
    start, end = _month_bounds_dates(year, month)
    rows = service.list_transaction_rows(start_date=start, end_date=end, currency=currency)
    if "几笔" in text:
        return f"{year}年{month}月共有 {len(rows)} 笔交易。"
    category_name = _category_in_text(text)
    currencies = _currencies_in(rows, currency)
    if category_name:
        if not currencies:
            return f"{year}年{month}月没有{category_name}支出。"
        lines = []
        for code in currencies:
            spent = net_spent_minor(rows, code, category_name)
            if spent == 0 and not any(row["currency"] == code and row.get("category_name") == category_name for row in rows):
                continue
            lines.append(f"{year}年{month}月{category_name}净支出 {format_money(spent, code)}。")
        return "\n".join(lines) if lines else f"{year}年{month}月没有{category_name}支出。"
    if not currencies:
        return f"{year}年{month}月还没有交易。"
    lines = []
    for code in currencies:
        income = _sum_type(rows, code, "income")
        expense = _sum_type(rows, code, "expense")
        refund = _sum_type(rows, code, "refund")
        net = expense - refund
        balance = income - net
        lines.append(
            f"{year}年{month}月（{code}）总收入 {format_money(income, code)}，"
            f"总支出 {format_money(expense, code)}，退款 {format_money(refund, code)}，"
            f"净支出 {format_money(net, code)}，结余 {format_money(balance, code)}。"
        )
    return "\n".join(lines)


def _sum_type(rows: list[dict], currency: str, transaction_type: str) -> int:
    return sum(
        int(row["amount_minor"])
        for row in rows
        if row["currency"] == currency and row["transaction_type"] == transaction_type
    )


def _currencies_in(rows: list[dict], currency: str | None) -> list[str]:
    if currency:
        return [currency] if any(row["currency"] == currency for row in rows) else []
    found: list[str] = []
    for row in rows:
        if row["currency"] not in found:
            found.append(row["currency"])
    return found


def _month_bounds_dates(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    next_year, next_month = shift_month(year, month, 1)
    end = date.fromordinal(date(next_year, next_month, 1).toordinal() - 1)
    return start, end


def _category_in_text(text: str) -> str | None:
    for name in sorted(EXPENSE_CATEGORIES + INCOME_CATEGORIES, key=len, reverse=True):
        if name in text:
            return name
    return None


def _currency_in_text(text: str) -> str | None:
    folded = text.casefold()
    for token, code in _CURRENCIES:
        if token.casefold() in folded:
            return code
    return None


def _question_keyword(text: str) -> str:
    cleaned = re.sub(
        r"这个月|上个月|本月|上月|多少|还能|可以|退款|退回|退钱|钱|呢|吗|花了|什么|查询|一下|的|了|还|退",
        " ",
        text,
    )
    cleaned = re.sub(r"\d+(?:\.\d+)?", " ", cleaned)
    for token, _code in _CURRENCIES:
        cleaned = re.sub(re.escape(token), " ", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.replace("？", " ").replace("?", " ").split())


def _looks_like_field_update(text: str, today: date) -> bool:
    if _amount_only(text):
        return True
    if extract_amounts(text) or _BOOKING.search(text) or _QUESTION.search(text):
        return False
    _resolved, _assumed, explicit, _start, _end = resolve_date(text, today)
    return explicit and len(text) <= 16


def _amount_only(text: str) -> bool:
    if not extract_amounts(text):
        return False
    stripped = re.sub(r"\d+(?:\.\d{1,2})?", "", text)
    for token, _code in _CURRENCIES:
        stripped = re.sub(re.escape(token), "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"[金额改为成换成是\s，,。元块]", "", stripped)
    return stripped == ""


def _drop_stale_warnings(draft: TransactionDraft) -> None:
    """字段已经补上之后，去掉原先提示缺失的警告。"""
    stale: list[str] = []
    if draft.transaction_date is not None and not draft.date_assumed:
        stale.extend(["没有识别到日期", "暂按今天"])
    if draft.amount is not None:
        stale.extend(["没有识别到金额", "多个金额"])
    if draft.type_confident:
        stale.append("没有判断出收入还是支出")
    if stale:
        draft.warnings = [item for item in draft.warnings if not any(token in item for token in stale)]


def _same_draft(left: TransactionDraft, right: TransactionDraft) -> bool:
    return (
        left.transaction_date == right.transaction_date
        and left.amount == right.amount
        and left.currency == right.currency
        and left.transaction_type == right.transaction_type
        and left.category == right.category
        and (left.merchant or None) == (right.merchant or None)
        and (left.description or "") == (right.description or "")
    )
