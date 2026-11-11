"""Local bookkeeping assistant.

It can edit a pending draft, split one sentence into several drafts, and
answer questions from the database. Amounts, counts, budgets, and refund
balances are calculated by the finance service. Nothing is written here.
A batch category change is only a preview until the user confirms it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from ai_parser import (
    TransactionDraft,
    _WEEKDAY_ALT,
    _mark_clarification,
    apply_merchant_memory,
    extract_amounts,
    parse_with_rules,
    resolve_date,
)
from config import EXPENSE_CATEGORIES, INCOME_CATEGORIES, MERCHANT_CATEGORIES, TRANSACTION_TYPE_LABELS, merchant_key
from finance_service import FinanceService, format_money, from_minor, net_spent_minor, shift_month, to_minor


_QUESTION = re.compile(
    r"\bhow much\b|\bhow many\b|\bbudget\b|\brefund(?:ed)?\b|\bcompared?\b|\bcompare\b|\bbalance\b|"
    r"\bhvor mye\b|\bhvor mange\b|\bbudsjett\b|\brefusjon\b|\bsammenlignet\b|\bsaldo\b",
    re.IGNORECASE,
)
_AMEND = re.compile(
    r"\bchange\b|\bmerchant is\b|\bstore is\b|\bcategory\b|\bamount\b|\bdescription\b|"
    r"\bthis is\b|\bendr(?:e|et)\b|\bbel[øo]p\b|\bbutikk\b",
    re.IGNORECASE,
)
_BATCH = re.compile(
    r"(?:move|change)\s+all\s+(?P<body>.+?)\s+to\s+(?P<category>[A-Za-z][A-Za-z ]+)$"
    r"|(?:flytt|endre)\s+alle\s+(?P<body_no>.+?)\s+til\s+(?P<category_no>[A-Za-z][A-Za-z ]+)$",
    re.IGNORECASE,
)
_DATE_HINT = re.compile(
    rf"\b(?:today|yesterday|the day before yesterday|last week|this week|forrige uke|"
    rf"i\s*dag|idag|i\s*går|igår|i\s*gar|igar|{_WEEKDAY_ALT})\b|\d{{4}}-\d{{2}}-\d{{2}}",
    re.IGNORECASE,
)
_BOOKING = re.compile(
    r"\b(?:spent|paid|bought|received|salary|cost|refund|brukte|betalte|kj[øo]pte|mottok|lønn|lonn)\b",
    re.IGNORECASE,
)
_CURRENCIES = (
    ("norske kroner", "NOK"),
    ("kroner", "NOK"),
    ("yuan", "CNY"),
    ("dollars", "USD"),
    ("dollar", "USD"),
    ("euros", "EUR"),
    ("euro", "EUR"),
    ("pounds", "GBP"),
    ("pound", "GBP"),
    ("NOK", "NOK"),
    ("CNY", "CNY"),
    ("USD", "USD"),
    ("EUR", "EUR"),
    ("GBP", "GBP"),
    ("JPY", "JPY"),
    ("SEK", "SEK"),
    ("DKK", "DKK"),
    ("kr", "NOK"),
)
_MONTH_WORDS = (
    "last month",
    "previous month",
    "this month",
    "forrige måned",
    "forrige maned",
    "denne måneden",
    "denne maneden",
    "denne måned",
)


@dataclass
class CategoryProposal:
    """A category change that has not been applied. Ids come from the current query."""

    category_name: str
    transaction_ids: list[int]
    preview: list[dict]
    summary: str


@dataclass
class AgentResult:
    """One turn. Drafts are not ledger rows yet."""

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
    """Decide whether this turn edits a draft, asks a question, previews a recategorization, or starts a new entry."""
    cleaned = text.strip()
    if not cleaned:
        return AgentResult("Describe a transaction, or ask how much you spent this month.", "empty")

    batch = _BATCH.search(cleaned)
    if batch:
        return _propose_recategory(cleaned, batch, today, service)

    if pending is not None and (_AMEND.search(cleaned) or _looks_like_field_update(cleaned, today)):
        amended = amend_draft(pending, cleaned, today=today, default_currency=default_currency, memories=memories)
        if _same_draft(pending, amended):
            pending_date = next((item for item in amended.warnings if "date was not changed" in item), None)
            if pending_date:
                return AgentResult(pending_date, "amend")
            return AgentResult(
                'No field to change was recognized. Try "change the amount to 120 NOK", "merchant is Kiwi", or "change it to last Wednesday".',
                "amend",
            )
        return AgentResult(f"Updated the current draft: {describe_draft(amended)}", "amend", amended_draft=amended)

    if _QUESTION.search(cleaned) and not _BOOKING.search(cleaned):
        return AgentResult(answer_ledger_question(service, cleaned, today), "question")
    if _QUESTION.search(cleaned) and re.search(r"\bhow much\b|\bhvor mye\b", cleaned, re.IGNORECASE) and not extract_amounts(cleaned):
        return AgentResult(answer_ledger_question(service, cleaned, today), "question")

    if _BOOKING.search(cleaned) or extract_amounts(cleaned):
        drafts = draft_transactions(cleaned, today=today, default_currency=default_currency, memories=memories)
        if len(drafts) == 1:
            reply = f"Draft ready to confirm: {describe_draft(drafts[0])}"
        else:
            reply = f"Split into {len(drafts)} drafts. Confirm the first one: {describe_draft(drafts[0])}"
        return AgentResult(reply, "draft", new_drafts=drafts)

    if pending is not None:
        return AgentResult(
            'I did not understand that change. Try "change the amount to 120 NOK", "merchant is Kiwi", or "change it to last Wednesday".',
            "unknown",
        )
    return AgentResult(
        'I can record a transaction, edit a draft, or answer from the ledger on this computer. For example: "Lunch cost 80 NOK, then the metro cost 30 NOK."',
        "help",
    )


def draft_transactions(
    text: str,
    *,
    today: date,
    default_currency: str,
    memories: dict[str, str],
) -> list[TransactionDraft]:
    """Split one sentence into drafts only when at least two parts have amounts."""
    parts = [
        part.strip()
        for part in re.split(r"[,;]| then | and then | also | også | ogsa ", text, flags=re.IGNORECASE)
        if part.strip()
    ]
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
    """Apply the fields the user mentioned. Unmentioned fields stay as they are."""
    updated = draft.model_copy(deep=True)
    category_explicit = False
    merchant_explicit = False

    merchant_match = re.search(
        r"\b(?:merchant|store|shop|butikk)\s+(?:is|to)\s+([A-Za-z0-9&.'’\- ]{1,40})",
        text,
        re.IGNORECASE,
    )
    if merchant_match:
        updated.merchant = " ".join(merchant_match.group(1).split())
        merchant_explicit = True

    category_name = _category_in_text(text)
    if category_name and re.search(r"\bcategory\b|\bchange\b|\bto\b|\btil\b", text, re.IGNORECASE):
        updated.category = category_name
        category_explicit = True
        if category_name in INCOME_CATEGORIES and category_name not in EXPENSE_CATEGORIES:
            updated.transaction_type = "refund" if category_name == "Refund" else "income"
            updated.type_confident = True

    type_match = re.search(r"\bthis is (?:an |a )?(income|expense|refund)\b", text, re.IGNORECASE)
    if type_match:
        updated.transaction_type = type_match.group(1).lower()
        updated.type_confident = True
        if updated.transaction_type == "refund":
            updated.category = "Refund"
            category_explicit = True
        elif updated.transaction_type == "income" and updated.category not in INCOME_CATEGORIES:
            updated.category = "Other income"
        elif updated.transaction_type == "expense" and updated.category not in EXPENSE_CATEGORIES:
            updated.category = "Other"

    description_match = re.search(r"\bdescription\s+(?:to|is)\s+(.+)$", text, re.IGNORECASE)
    if description_match:
        updated.description = description_match.group(1).strip()[:500]

    if re.search(r"\bamount\b|\bchange\b|\bbel[øo]p\b", text, re.IGNORECASE) or _amount_only(text):
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
        elif re.search(r"\blast week\b|\bforrige uke\b", text, re.IGNORECASE):
            updated.warnings.append("Last week needs a specific day. The date was not changed.")

    currency_code = _currency_in_text(text)
    if currency_code and re.search(r"\bcurrency\b|\bchange\b|\bto\b|\bvaluta\b", text, re.IGNORECASE):
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
    """Answer with numbers from the database. Say when nothing was found."""
    year, month = mentioned_month(text, today)
    currency = _currency_in_text(text)
    folded = text.casefold()
    if "budget" in folded or "budsjett" in folded:
        return _answer_budget(service, text, year, month)
    if re.search(r"\brefund\b|\brefusjon\b", folded) and re.search(r"\bhow much\b|\bhvor mye\b|\bstill\b", folded):
        return _answer_refund_remaining(service, text, currency)
    if re.search(r"\bcompare|\bcompared\b|\bsammenlignet\b|\blast month\b|\bforrige måned\b|\bforrige maned\b", folded):
        if re.search(r"\bcompare|\bcompared\b|\bsammenlignet\b", folded):
            return _answer_comparison(service, text, year, month, currency)
    return _answer_summary(service, text, year, month, currency)


def describe_draft(draft: TransactionDraft) -> str:
    """One sentence the user can check before saving."""
    when = draft.transaction_date.isoformat() if draft.transaction_date else "date missing"
    if draft.amount is not None and draft.currency:
        money = format_money(to_minor(draft.amount, draft.currency), draft.currency)
    else:
        money = "amount missing"
    label = TRANSACTION_TYPE_LABELS.get(draft.transaction_type or "", "Expense")
    category = draft.category or "Uncategorized"
    merchant = f", merchant {draft.merchant}" if draft.merchant else ""
    question = f". {draft.clarification_question}" if draft.clarification_question else ""
    return f"{when} {label} {money} {category}{merchant}{question}"


def mentioned_month(text: str, today: date) -> tuple[int, int]:
    """The month a question refers to. Defaults to the month of today."""
    folded = text.casefold()
    if "last month" in folded or "previous month" in folded or "forrige måned" in folded or "forrige maned" in folded:
        return shift_month(today.year, today.month, -1)
    if "this month" not in folded and "denne måned" not in folded and "denne maned" not in folded:
        for name, number in (
            ("january", 1), ("januar", 1), ("february", 2), ("februar", 2),
            ("march", 3), ("mars", 3), ("april", 4), ("may", 5), ("mai", 5),
            ("june", 6), ("juni", 6), ("july", 7), ("juli", 7), ("august", 8),
            ("september", 9), ("october", 10), ("oktober", 10), ("november", 11),
            ("december", 12), ("desember", 12),
        ):
            if re.search(rf"\b{name}\b", folded):
                year = today.year if number <= today.month else today.year - 1
                return year, number
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


def _batch_groups(match: re.Match[str]) -> tuple[str, str]:
    if match.group("body") is not None:
        return match.group("body"), match.group("category")
    return match.group("body_no"), match.group("category_no")


def _propose_recategory(text: str, match: re.Match[str], today: date, service: FinanceService) -> AgentResult:
    body, category_text = _batch_groups(match)
    category_name = _category_in_text(category_text)
    if category_name not in EXPENSE_CATEGORIES:
        return AgentResult("Use an existing expense category, such as Transport, Dining, or Groceries. Nothing was changed.", "batch")
    if not re.search(r"last month|previous month|this month|forrige måned|forrige maned|denne måned|denne maned", body, re.IGNORECASE):
        return AgentResult('Say which month to change, for example "move all last month taxi to Transport". Nothing was changed.', "batch")
    year, month = mentioned_month(body, today)
    keyword = body
    for phrase in _MONTH_WORDS:
        keyword = re.sub(re.escape(phrase), " ", keyword, flags=re.IGNORECASE)
    keyword = " ".join(keyword.replace("'s", " ").split())
    if not keyword:
        return AgentResult("Add a merchant or keyword, such as taxi or Uber. Nothing was changed.", "batch")
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
            f"No {keyword} expenses in {year}-{month:02d} still need to be changed to {category_name}.",
            "batch",
        )
    preview = [
        {
            "id": row["id"],
            "date": row["transaction_date"],
            "amount": row["amount"],
            "currency": row["currency"],
            "merchant": row.get("merchant") or "",
            "previous category": row.get("category_name") or "",
        }
        for row in selected[:20]
    ]
    proposal = CategoryProposal(
        category_name=category_name,
        transaction_ids=[int(row["id"]) for row in selected],
        preview=preview,
        summary=(
            f"Ready to change {len(selected)} \"{keyword}\" expenses from {year}-{month:02d} to {category_name}. "
            "Nothing is saved until you confirm."
        ),
    )
    return AgentResult(proposal.summary, "batch", proposal=proposal)


def _answer_budget(service: FinanceService, text: str, year: int, month: int) -> str:
    statuses = service.budget_status(year=year, month=month)
    category_name = _category_in_text(text)
    if category_name:
        statuses = [item for item in statuses if item.category_name == category_name]
    if not statuses:
        return f"No matching budget for {year}-{month:02d}."
    lines = [f"Budget for {year}-{month:02d}:"]
    for status in statuses:
        name = status.category_name or "Total spending"
        spent = format_money(status.spent_minor, status.currency)
        limit = format_money(status.limit_minor, status.currency)
        if status.limit_minor <= 0:
            lines.append(f"{name} ({status.currency}): spent {spent}, budget {limit}.")
            continue
        percent = (status.ratio * Decimal("100")).quantize(Decimal("0.1"))
        lines.append(f"{name} ({status.currency}): {percent}% used, spent {spent} / budget {limit}.")
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
            f"#{row['id']} {row['transaction_date']} {label} can still be refunded "
            f"{format_money(remaining, row['currency'])}."
        )
        if len(lines) >= 8:
            break
    if not lines:
        target = f" for \"{keyword}\"" if keyword else ""
        return f"No expense{target} can still be refunded."
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
        return f"There is nothing to compare between {year}-{month:02d} and {previous_year}-{previous_month:02d}."
    subject = category_name or "net expense"
    lines = []
    for code in currencies:
        current_minor = net_spent_minor(current_rows, code, category_name)
        previous_minor = net_spent_minor(previous_rows, code, category_name)
        delta = from_minor(current_minor, code) - from_minor(previous_minor, code)
        lines.append(
            f"{year}-{month:02d} {subject} {format_money(current_minor, code)}, "
            f"{previous_year}-{previous_month:02d} was {format_money(previous_minor, code)}, "
            f"difference {delta} {code}."
        )
    return "\n".join(lines)


def _answer_summary(service: FinanceService, text: str, year: int, month: int, currency: str | None) -> str:
    start, end = _month_bounds_dates(year, month)
    rows = service.list_transaction_rows(start_date=start, end_date=end, currency=currency)
    if re.search(r"\bhow many\b|\bhvor mange\b", text, re.IGNORECASE):
        return f"{year}-{month:02d} has {len(rows)} transactions."
    category_name = _category_in_text(text)
    currencies = _currencies_in(rows, currency)
    if category_name:
        if not currencies:
            return f"No {category_name} expenses in {year}-{month:02d}."
        lines = []
        for code in currencies:
            spent = net_spent_minor(rows, code, category_name)
            if spent == 0 and not any(row["currency"] == code and row.get("category_name") == category_name for row in rows):
                continue
            lines.append(f"{year}-{month:02d} {category_name} net expense {format_money(spent, code)}.")
        return "\n".join(lines) if lines else f"No {category_name} expenses in {year}-{month:02d}."
    if not currencies:
        return f"No transactions in {year}-{month:02d}."
    lines = []
    for code in currencies:
        income = _sum_type(rows, code, "income")
        expense = _sum_type(rows, code, "expense")
        refund = _sum_type(rows, code, "refund")
        net = expense - refund
        balance = income - net
        lines.append(
            f"{year}-{month:02d} ({code}): income {format_money(income, code)}, "
            f"expenses {format_money(expense, code)}, refunds {format_money(refund, code)}, "
            f"net expense {format_money(net, code)}, balance {format_money(balance, code)}."
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
    folded = text.casefold()
    for name in sorted(EXPENSE_CATEGORIES + INCOME_CATEGORIES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name.casefold())}\b", folded):
            return name
    return None


def _currency_in_text(text: str) -> str | None:
    folded = text.casefold()
    for token, code in _CURRENCIES:
        if re.search(rf"\b{re.escape(token.casefold())}\b", folded):
            return code
    return None


def _question_keyword(text: str) -> str:
    cleaned = text
    for phrase in (
        "this month", "last month", "previous month", "how much", "how many",
        "can still be refunded", "still be refunded", "refunded", "refund",
        "of", "the", "a", "an", "did", "does", "cost", "spent", "what",
        "hvor mye", "hvor mange", "kan", "fortsatt", "refusjon",
    ):
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\d+(?:\.\d+)?", " ", cleaned)
    for token, _code in _CURRENCIES:
        cleaned = re.sub(rf"\b{re.escape(token)}\b", " ", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.replace("?", " ").split())


def _looks_like_field_update(text: str, today: date) -> bool:
    if _amount_only(text):
        return True
    if extract_amounts(text) or _BOOKING.search(text) or _QUESTION.search(text):
        return False
    _resolved, _assumed, explicit, _start, _end = resolve_date(text, today)
    return explicit and len(text) <= 24


def _amount_only(text: str) -> bool:
    if not extract_amounts(text):
        return False
    stripped = re.sub(r"\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?", "", text)
    for token, _code in _CURRENCIES:
        stripped = re.sub(rf"\b{re.escape(token)}\b", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\b(?:amount|change|to|the|is|set|beløp|belop)\b", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"[\s,.$€£]", "", stripped)
    return stripped == ""


def _drop_stale_warnings(draft: TransactionDraft) -> None:
    """Remove warnings for fields that have now been filled in."""
    stale: list[str] = []
    if draft.transaction_date is not None and not draft.date_assumed:
        stale.extend(["No date found", "Using today"])
    if draft.amount is not None:
        stale.extend(["No amount found", "Several amounts"])
    if draft.type_confident:
        stale.append("Could not tell income from expense")
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
