"""Streamlit interface for the local ledger.

Nothing is written until the user clicks Save.
A rerun does not insert the same row again: the confirmation token is remembered.
"""

from __future__ import annotations

import tempfile
import uuid
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

from agent import AgentResult, handle_user_message
from ai_parser import (
    TransactionDraft,
    apply_merchant_memory,
    list_installed_models,
    parse_transaction_text,
    today_in_timezone,
)
from config import (
    DB_PATH,
    EXPENSE_CATEGORIES,
    INCOME_CATEGORIES,
    Settings,
    load_settings,
    save_settings,
)
from finance_service import (
    DuplicateTransactionError,
    FinanceService,
    format_money,
    from_minor,
)
from receipt_parser import ReceiptError, parse_receipt_bytes
from voice_parser import VoiceError, transcribe_and_parse, validate_audio_name
from database import get_connection, list_audit_logs
from zoneinfo import ZoneInfo


TYPE_OPTIONS = [("expense", "Expense"), ("income", "Income"), ("refund", "Refund")]
CURRENCIES = ["NOK", "CNY", "USD", "EUR", "GBP", "SEK", "DKK", "JPY"]
FREQUENCIES = [("day", "By day"), ("week", "By week"), ("month", "By month")]


def get_service() -> FinanceService:
    """Open or create the local database and return the finance service."""
    return FinanceService(DB_PATH)


def ensure_state() -> None:
    """Initialize the session. Settings and pending drafts live in session state."""
    if "settings" not in st.session_state:
        st.session_state.settings = load_settings()
    if "saved_tokens" not in st.session_state:
        st.session_state.saved_tokens = []
    if "pending" not in st.session_state:
        st.session_state.pending = None
    if "transcript" not in st.session_state:
        st.session_state.transcript = ""
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
    if "agent_queue" not in st.session_state:
        st.session_state.agent_queue = []
    if "agent_proposal" not in st.session_state:
        st.session_state.agent_proposal = None


def current_settings() -> Settings:
    return st.session_state.settings


def current_today() -> date:
    return today_in_timezone(current_settings().timezone)


def show_flash() -> None:
    message = st.session_state.pop("flash", None)
    if message:
        st.success(message)


def stage_draft(draft: TransactionDraft) -> None:
    """Store a draft in the session. The database is not written yet."""
    st.session_state.pending = {
        "token": uuid.uuid4().hex,
        "draft": draft.model_dump(mode="json"),
    }


def remember_saved(service: FinanceService, result: dict) -> None:
    """Remember the merchant category after a save, for the next parse."""
    service.remember_merchant(result.get("merchant"), result["category_name"], result["transaction_type"])


def stage_with_memory(service: FinanceService, draft: TransactionDraft) -> None:
    """A one-off parse also uses merchant categories that were remembered."""
    stage_draft(apply_merchant_memory(draft, service.merchant_memories()))


def try_amount(text: str) -> Decimal | None:
    """Parse an amount typed by the user. An invalid value stays empty so the page can keep rendering."""
    cleaned = text.strip().replace(",", "").replace(" ", "")
    if not cleaned:
        return None
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    if value <= 0:
        return None
    return value


def duplicate_rows(report) -> list[dict]:
    rows = []
    for label, items in (("Exact duplicate", report.exact), ("Possible duplicate", report.suspected)):
        for item in items:
            rows.append(
                {
                    "Match": label,
                    "Id": item["id"],
                    "Date": item["transaction_date"],
                    "Amount": item["amount"],
                    "Currency": item["currency"],
                    "Type": item["type_label"],
                    "Merchant": item.get("merchant") or "",
                    "Description": item.get("description") or "",
                }
            )
    return rows


def threshold_for(currency: str) -> Decimal | None:
    raw = current_settings().thresholds.get(currency)
    if not raw:
        return None
    try:
        return Decimal(str(raw))
    except InvalidOperation:
        return None


def render_editor(
    service: FinanceService,
    *,
    prefix: str,
    initial: dict,
    submit_label: str,
) -> dict | None:
    """Render the editable confirmation form. Fields are returned only when the button is clicked."""
    for warning in initial.get("warnings") or []:
        st.warning(warning)
    if initial.get("clarification"):
        st.info(initial["clarification"])
    if initial.get("items"):
        st.markdown("**Items found**")
        for item in initial["items"]:
            st.write(f"- {item}")
    if initial.get("raw_text"):
        with st.expander("Original text"):
            st.write(initial["raw_text"])

    type_labels = [label for _code, label in TYPE_OPTIONS]
    type_codes = [code for code, _label in TYPE_OPTIONS]
    current_type = initial.get("transaction_type") or "expense"
    if current_type not in type_codes:
        current_type = "expense"
    selected_label = st.selectbox(
        "Type",
        type_labels,
        index=type_codes.index(current_type),
        key=f"{prefix}_type",
    )
    transaction_type = type_codes[type_labels.index(selected_label)]
    if transaction_type == "expense":
        categories = EXPENSE_CATEGORIES
    elif transaction_type == "refund":
        categories = ["Refund"]
    else:
        categories = INCOME_CATEGORIES
    category_value = initial.get("category")
    if category_value not in categories:
        category_value = categories[0]
        category = st.selectbox(
        "Category",
        categories,
        index=categories.index(category_value),
        key=f"{prefix}_category_{transaction_type}",
    )
    date_value = initial.get("transaction_date") or current_today()
    if isinstance(date_value, str):
        date_value = date.fromisoformat(date_value)
    transaction_date = st.date_input("Date", value=date_value, key=f"{prefix}_date")
    currency_options = list(CURRENCIES)
    currency_value = (initial.get("currency") or current_settings().default_currency).upper()
    if currency_value not in currency_options:
        currency_options.append(currency_value)
    currency = st.selectbox(
        "Currency",
        currency_options,
        index=currency_options.index(currency_value),
        key=f"{prefix}_currency",
    )
    possible = initial.get("possible_amounts") or []
    if possible:
        choice = st.radio(
            "Several amounts were found. Choose the one you paid",
            possible + ["Type it in"],
            key=f"{prefix}_amount_choice",
        )
        if choice == "Type it in":
            amount_text = st.text_input("Amount", value="", key=f"{prefix}_amount")
        else:
            amount_text = choice
            st.caption(f"Selected amount: {choice}")
    else:
        amount_text = st.text_input(
            "Amount",
            value=initial.get("amount_text") or "",
            key=f"{prefix}_amount",
        )
    description = st.text_input("Description", value=initial.get("description") or "", key=f"{prefix}_description")
    merchant = st.text_input("Merchant", value=initial.get("merchant") or "", key=f"{prefix}_merchant")
    account = st.text_input("Account", value=initial.get("account") or "", key=f"{prefix}_account")
    amount = try_amount(amount_text)
    if amount is not None:
        limit = threshold_for(currency)
        if limit is not None and amount >= limit:
            st.warning(f"The amount reaches the {currency} {limit} warning line. Every amount still has to be confirmed before it is saved.")

    original_id = None
    if transaction_type == "refund":
        if amount is None:
            st.info("After you enter a refund amount, expenses you can link are listed.")
        else:
            candidates, note = service.find_refund_candidates(
                amount=amount,
                currency=currency,
                merchant=merchant.strip() or None,
                start_date=_as_date(initial.get("search_start")),
                end_date=_as_date(initial.get("search_end")),
                exclude_refund_id=initial.get("exclude_id") if initial.get("transaction_type") == "refund" else None,
            )
            if note:
                st.warning(note)
            if not candidates:
                st.error("No expense with a remaining refund balance was found.")
            else:
                labels = []
                ids = []
                for item in candidates:
                    labels.append(
                        f"#{item['id']} {item['transaction_date']} {item['amount']} {item['currency']} "
                        f"{item.get('category_name') or ''} {item.get('merchant') or ''} remaining {item['remaining']}"
                    )
                    ids.append(int(item["id"]))
                wanted = initial.get("original_transaction_id")
                index = ids.index(wanted) if wanted in ids else 0
                selected = st.selectbox("Original expense", labels, index=index, key=f"{prefix}_original")
                original_id = ids[labels.index(selected)]

    report = None
    allow_duplicate = False
    if amount is not None:
        report = service.find_duplicates(
            transaction_date=transaction_date,
            amount=amount,
            currency=currency,
            transaction_type=transaction_type,
            description=description,
            merchant=merchant,
            exclude_id=initial.get("exclude_id"),
        )
        if report.has_any:
            st.warning("A similar past transaction was found. Saving will not overwrite or delete it.")
            st.dataframe(duplicate_rows(report), width="stretch", hide_index=True)
            allow_duplicate = st.checkbox("I checked this and still want to save", key=f"{prefix}_allow_duplicate")

    submitted = st.button(submit_label, type="primary", key=f"{prefix}_submit")
    if not submitted:
        return None
    if amount is None:
        st.error("Enter an amount greater than 0.")
        return None
    if not description.strip():
        st.error("Enter a description.")
        return None
    if transaction_type == "refund" and original_id is None:
        st.error("Choose the original expense for this refund.")
        return None
    if report is not None and report.has_any and not allow_duplicate:
        st.error("Tick the similar-transaction confirmation, or go back and edit.")
        return None
    return {
        "transaction_date": transaction_date,
        "amount": amount,
        "currency": currency,
        "transaction_type": transaction_type,
        "category_name": category,
        "description": description.strip(),
        "merchant": merchant.strip() or None,
        "account_name": account.strip() or None,
        "original_transaction_id": original_id,
        "source": initial.get("source") or "manual",
        "allow_duplicate": bool(report and report.has_any and allow_duplicate),
    }


def _as_date(value) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def initial_from_draft(draft: TransactionDraft) -> dict:
    return {
        "transaction_date": draft.transaction_date,
        "amount_text": format(draft.amount, "f") if draft.amount is not None else "",
        "currency": draft.currency,
        "transaction_type": draft.transaction_type,
        "category": draft.category,
        "description": draft.description or "",
        "merchant": draft.merchant or "",
        "account": draft.account or "",
        "possible_amounts": [format(item, "f") for item in draft.possible_amounts],
        "original_transaction_id": None,
        "search_start": draft.search_start,
        "search_end": draft.search_end,
        "warnings": draft.warnings,
        "clarification": draft.clarification_question,
        "items": draft.items,
        "raw_text": draft.raw_text,
        "source": draft.source,
        "exclude_id": None,
    }


def render_pending(service: FinanceService) -> None:
    """Show the pending transaction. After a successful save, clear the draft so a rerun does not write it again."""
    pending = st.session_state.pending
    if not pending:
        return
    token = pending["token"]
    if token in st.session_state.saved_tokens:
        st.info("This transaction was just saved.")
        return
    try:
        draft = TransactionDraft.model_validate(pending["draft"])
    except ValidationError:
        st.error("The pending data could not be read. Parse it again.")
        st.session_state.pending = None
        return
    st.subheader("Confirm transaction")
    st.caption("You can edit the fields below. They are written to the local database only after you click Save.")
    result = render_editor(
        service,
        prefix=token,
        initial=initial_from_draft(draft),
        submit_label="Save",
    )
    if result is None:
        return
    try:
        transaction_id = service.create_transaction(**result)
    except DuplicateTransactionError:
        st.error("Another similar transaction was found before saving. Tick the confirmation and save again.")
        return
    except (ValueError, KeyError) as exc:
        st.error(str(exc))
        return
    st.session_state.saved_tokens.append(token)
    st.session_state.saved_tokens = st.session_state.saved_tokens[-200:]
    remember_saved(service, result)
    st.session_state.pending = None
    st.session_state.flash = f"Saved, transaction {transaction_id}."
    st.rerun()


def page_add(service: FinanceService) -> None:
    st.header("Add transaction")
    st.caption("Type a sentence, upload a receipt, or use speech. Each one becomes a draft and is saved only after you confirm.")
    settings = current_settings()
    today = current_today()
    text_tab, receipt_tab, voice_tab, manual_tab = st.tabs(["Text", "Receipt", "Speech", "Manual"])
    with text_tab:
        text = st.text_area(
            "Describe one transaction",
            placeholder="For example: Spent 350 NOK at Kiwi today. Or: I dag brukte jeg 350 kr på Kiwi.",
            height=120,
        )
        if st.button("Parse text", type="primary"):
            if not text.strip():
                st.warning("Enter a transaction.")
            else:
                draft = parse_transaction_text(
                    text,
                    today=today,
                    default_currency=settings.default_currency,
                    settings=settings,
                )
                stage_with_memory(service, draft)
    with receipt_tab:
        image = st.file_uploader("Upload a PNG, JPG, or JPEG", type=["png", "jpg", "jpeg"])
        st.caption("The image is sent only to the Ollama address in Settings, which defaults to this computer.")
        if st.button("Read receipt", type="primary"):
            if image is None:
                st.warning("Choose an image first.")
            else:
                try:
                    draft = parse_receipt_bytes(
                        image.getvalue(),
                        image.name,
                        today=today,
                        default_currency=settings.default_currency,
                        settings=settings,
                    )
                except (ReceiptError, ValidationError, Exception) as exc:
                    if exc.__class__.__name__ in {"OllamaError", "OllamaHTTPError", "ReceiptError"} or isinstance(
                        exc, (ReceiptError, ValidationError)
                    ):
                        st.error(str(exc))
                    else:
                        st.error(f"Could not read the receipt: {exc}")
                else:
                    stage_with_memory(service, draft)
    with voice_tab:
        st.caption("Speech is transcribed on this computer, then parsed and confirmed like typed text.")
        audio_file = st.file_uploader(
            "Upload audio",
            type=["wav", "mp3", "m4a", "ogg", "flac", "webm", "aac"],
        )
        recorded = st.audio_input("Or record here") if hasattr(st, "audio_input") else None
        if not hasattr(st, "audio_input"):
            st.info("This Streamlit version has no recorder. Upload an audio file instead.")
        if "transcript_box" not in st.session_state:
            st.session_state.transcript_box = ""
        if st.session_state.pop("transcript_dirty", False):
            st.session_state.transcript_box = st.session_state.transcript
        edited = st.text_area("You can edit the transcript before parsing", key="transcript_box")
        action_col, reparse_col = st.columns(2)
        with action_col:
            transcribe_clicked = st.button("Transcribe and parse", type="primary")
        with reparse_col:
            reparse_clicked = st.button("Parse the text above")
        chosen = recorded or audio_file
        if transcribe_clicked:
            if chosen is None:
                st.warning("Record audio or upload a file first.")
            else:
                filename = getattr(chosen, "name", "recording.wav")
                try:
                    suffix = validate_audio_name(filename)
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
                        temporary.write(chosen.getvalue())
                        temporary_path = Path(temporary.name)
                    try:
                        transcript, draft = transcribe_and_parse(
                            temporary_path,
                            today=today,
                            default_currency=settings.default_currency,
                            settings=settings,
                        )
                    finally:
                        temporary_path.unlink(missing_ok=True)
                except (VoiceError, ValidationError, Exception) as exc:
                    st.error(str(exc))
                else:
                    st.session_state.transcript = transcript
                    st.session_state.transcript_dirty = True
                    stage_with_memory(service, draft)
                    st.rerun()
        if reparse_clicked:
            if not edited.strip():
                st.warning("There is no text to parse.")
            else:
                draft = parse_transaction_text(
                    edited,
                    today=today,
                    default_currency=settings.default_currency,
                    settings=settings,
                )
                draft.source = "voice"
                stage_with_memory(service, draft)
    with manual_tab:
        st.write("You can fill the form yourself. It still needs confirmation before it is saved.")
        if st.button("Start a manual entry"):
            stage_draft(
                TransactionDraft(
                    transaction_date=today,
                    date_explicit=True,
                    currency=settings.default_currency,
                    currency_assumed=True,
                    transaction_type="expense",
                    type_confident=True,
                    category="Other",
                    description="",
                    source="manual",
                    parser="manual",
                )
            )
    render_pending(service)


def render_budget_alerts(service: FinanceService, today: date) -> None:
    """Show this month's budget warnings at the top of Overview. Both 80% and over-budget use a red message."""
    for status in service.budget_status(year=today.year, month=today.month):
        if status.level == "ok":
            continue
        name = status.category_name or "Total spending"
        spent = format_money(status.spent_minor, status.currency)
        limit = format_money(status.limit_minor, status.currency)
        if status.level == "over":
            st.error(f"Over budget: {name} ({status.currency}) spent {spent}, budget {limit}.")
        else:
            percent = (status.ratio * Decimal(100)).quantize(Decimal("0.1"))
            st.error(f"Budget warning: {name} ({status.currency}) used {percent}%, spent {spent} / budget {limit}.")


def render_charts(summary) -> None:
    """Turn integer amounts into display values for the charts. The totals themselves are not computed with floats."""
    currency = summary.currency
    labels = []
    values = []
    for name, minor in summary.category_net:
        if minor > 0:
            labels.append(name)
            values.append(float(from_minor(minor, currency)))
    if labels:
        pie = px.pie(names=labels, values=values, title=f"{currency} share of net expense")
        st.plotly_chart(pie, width="stretch")
    else:
        st.info(f"{currency} has no positive net expense in this filter, so there is no pie chart.")
    if not summary.periods:
        return
    income = [float(from_minor(value, currency)) for value in summary.period_income]
    net_expense = [float(from_minor(value, currency)) for value in summary.period_net_expense]
    bar = go.Figure()
    bar.add_bar(name="Income", x=summary.periods, y=income)
    bar.add_bar(name="Net expense", x=summary.periods, y=net_expense)
    bar.update_layout(barmode="group", title=f"{currency} income and net expense")
    st.plotly_chart(bar, width="stretch")
    line = px.line(x=summary.periods, y=net_expense, markers=True, title=f"{currency} net expense trend")
    line.update_layout(xaxis_title="Period", yaxis_title=currency)
    st.plotly_chart(line, width="stretch")


def _queue_item(draft: TransactionDraft) -> dict:
    return {"token": uuid.uuid4().hex, "draft": draft.model_dump(mode="json")}


def page_agent(service: FinanceService) -> None:
    """Conversational bookkeeping. Questions and draft edits are not written to the database."""
    st.header("Assistant")
    st.caption("You can keep editing a draft, or ask about the ledger on this computer. Amounts come from the database. Nothing is saved until you confirm.")
    settings = current_settings()
    today = current_today()
    for message in st.session_state.agent_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    proposal = st.session_state.agent_proposal
    if proposal:
        st.warning(proposal["summary"])
        st.dataframe(proposal["preview"], width="stretch", hide_index=True)
        confirm_col, cancel_col = st.columns(2)
        with confirm_col:
            if st.button("Apply these category changes", type="primary"):
                changed = service.recategorize(proposal["transaction_ids"], proposal["category_name"])
                st.session_state.agent_proposal = None
                st.session_state.agent_messages.append(
                    {"role": "assistant", "content": f"Updated the category on {changed} expenses."}
                )
                st.session_state.agent_messages = st.session_state.agent_messages[-40:]
                st.rerun()
        with cancel_col:
            if st.button("Cancel this change"):
                st.session_state.agent_proposal = None
                st.session_state.agent_messages.append({"role": "assistant", "content": "Cancelled. The ledger was not changed."})
                st.rerun()

    queue = st.session_state.agent_queue
    if queue:
        st.subheader(f"To confirm {1} / {len(queue)}")
        try:
            draft = TransactionDraft.model_validate(queue[0]["draft"])
        except ValidationError:
            st.session_state.agent_queue = queue[1:]
            st.error("One draft could not be read and was skipped.")
            st.rerun()
        result = render_editor(
            service,
            prefix=f"agent_{queue[0]['token']}",
            initial=initial_from_draft(draft),
            submit_label="Save this one",
        )
        if st.button("Cancel this draft"):
            st.session_state.agent_queue = queue[1:]
            st.session_state.agent_messages.append({"role": "assistant", "content": "Cancelled the current draft. Nothing was written."})
            st.rerun()
        if result is not None:
            try:
                transaction_id = service.create_transaction(**result)
            except DuplicateTransactionError:
                st.error("Another similar transaction was found before saving. Tick the confirmation and save again.")
                return
            except (ValueError, KeyError) as exc:
                st.error(str(exc))
                return
            remember_saved(service, result)
            st.session_state.agent_queue = st.session_state.agent_queue[1:]
            st.session_state.agent_messages.append(
                {"role": "assistant", "content": f"Saved, transaction {transaction_id}."}
            )
            st.session_state.agent_messages = st.session_state.agent_messages[-40:]
            st.session_state.flash = f"Saved, transaction {transaction_id}."
            st.rerun()

    prompt = st.chat_input("For example: Lunch cost 80 NOK, then the metro cost 30 NOK. Or ask how much Dining cost this month.")
    if not prompt:
        return
    pending = None
    if st.session_state.agent_queue:
        try:
            pending = TransactionDraft.model_validate(st.session_state.agent_queue[0]["draft"])
        except ValidationError:
            pending = None
    outcome: AgentResult = handle_user_message(
        prompt,
        today=today,
        default_currency=settings.default_currency,
        service=service,
        pending=pending,
        memories=service.merchant_memories(),
    )
    st.session_state.agent_messages.append({"role": "user", "content": prompt})
    st.session_state.agent_messages.append({"role": "assistant", "content": outcome.reply})
    st.session_state.agent_messages = st.session_state.agent_messages[-40:]
    if outcome.new_drafts:
        st.session_state.agent_queue.extend(_queue_item(item) for item in outcome.new_drafts)
    if outcome.amended_draft is not None and st.session_state.agent_queue:
        st.session_state.agent_queue[0] = _queue_item(outcome.amended_draft)
    if outcome.proposal is not None:
        st.session_state.agent_proposal = {
            "summary": outcome.proposal.summary,
            "category_name": outcome.proposal.category_name,
            "transaction_ids": outcome.proposal.transaction_ids,
            "preview": outcome.proposal.preview,
        }
    st.rerun()


def page_dashboard(service: FinanceService) -> None:
    st.header("Overview")
    today = current_today()
    render_budget_alerts(service, today)
    for note in service.spending_insights(today):
        st.warning(note)
    st.caption("Each currency is totaled on its own. There is no exchange rate. Refunds are subtracted from expenses.")
    categories = service.list_categories()
    category_labels = ["All"] + [f"{item['name']} ({'expense' if item['category_type']=='expense' else 'income'})" for item in categories]
    filter_col, type_col, currency_col = st.columns(3)
    with filter_col:
        start_date = st.date_input("Start date", value=today.replace(day=1), key="dash_start")
        end_date = st.date_input("End date", value=today, key="dash_end")
    with type_col:
        category_choice = st.selectbox("Category", category_labels, key="dash_category")
        type_choice = st.selectbox("Type", ["All", "Expense", "Income", "Refund"], key="dash_type")
    with currency_col:
        keyword = st.text_input("Keyword", key="dash_keyword")
        frequency_label = st.radio("Period", [label for _code, label in FREQUENCIES], horizontal=True)
    frequency = dict((label, code) for code, label in FREQUENCIES)[frequency_label]
    category_id = None
    if category_choice != "All":
        category_id = categories[category_labels.index(category_choice) - 1]["id"]
    type_map = {"All": None, "Expense": "expense", "Income": "income", "Refund": "refund"}
    known_currencies = []
    for row in service.list_transaction_rows():
        if row["currency"] not in known_currencies:
            known_currencies.append(row["currency"])
    default_currency = current_settings().default_currency
    currency_options = ["All"] + known_currencies
    preferred = default_currency if default_currency in known_currencies else "All"
    currency_choice = st.selectbox(
        "Currency",
        currency_options,
        index=currency_options.index(preferred),
        key="dash_currency",
    )
    summaries = service.summarize(
        start_date=start_date,
        end_date=end_date,
        currency=None if currency_choice == "All" else currency_choice,
        category_id=category_id,
        transaction_type=type_map[type_choice],
        keyword=keyword.strip() or None,
        frequency=frequency,
    )
    if not summaries:
        st.info("There are no transactions in this range.")
    for summary in summaries:
        st.subheader(summary.currency)
        columns = st.columns(5)
        values = [
            ("Income", summary.total_income_minor),
            ("Total spending", summary.total_expense_minor),
            ("Refund", summary.total_refund_minor),
            ("Net expense", summary.net_expense_minor),
            ("Balance", summary.balance_minor),
        ]
        for column, (label, minor) in zip(columns, values):
            column.metric(label, format_money(minor, summary.currency))
        render_charts(summary)
    st.subheader("Recent transactions")
    recent = service.list_transaction_rows(
        start_date=start_date,
        end_date=end_date,
        currency=None if currency_choice == "All" else currency_choice,
        category_id=category_id,
        transaction_type=type_map[type_choice],
        keyword=keyword.strip() or None,
    )[:8]
    st.dataframe(transaction_view(recent), width="stretch", hide_index=True)


def transaction_view(rows: list[dict]) -> list[dict]:
    view = []
    for row in rows:
        view.append(
            {
                "Id": row["id"],
                "Date": row["transaction_date"],
                "Amount": row["amount"],
                "Currency": row["currency"],
                "Type": row["type_label"],
                "Category": row.get("category_name") or "",
                "Merchant": row.get("merchant") or "",
                "Description": row.get("description") or "",
                "Account": row.get("account_name") or "",
                "Original": row.get("original_transaction_id") or "",
                "Status": "Deleted" if row.get("is_deleted") else "Active",
            }
        )
    return view


def page_history(service: FinanceService) -> None:
    st.header("Transactions")
    today = current_today()
    keyword = st.text_input("Search description, merchant, or account")
    left, middle, right = st.columns(3)
    with left:
        start_date = st.date_input("Start", value=date(today.year, 1, 1), key="history_start")
        end_date = st.date_input("End", value=today, key="history_end")
    with middle:
        currency = st.selectbox("Currency", ["All"] + CURRENCIES, key="history_currency")
        type_label = st.selectbox("Type", ["All", "Expense", "Income", "Refund"], key="history_type")
    with right:
        include_deleted = st.checkbox("Show deleted")
        categories = service.list_categories()
        category_labels = ["All"] + [item["name"] + (" / Expense" if item["category_type"] == "expense" else " / Income") for item in categories]
        category_choice = st.selectbox("Category", category_labels)
    type_map = {"All": None, "Expense": "expense", "Income": "income", "Refund": "refund"}
    category_id = None if category_choice == "All" else categories[category_labels.index(category_choice) - 1]["id"]
    filters = {
        "start_date": start_date,
        "end_date": end_date,
        "currency": None if currency == "All" else currency,
        "category_id": category_id,
        "transaction_type": type_map[type_label],
        "keyword": keyword,
        "include_deleted": include_deleted,
    }
    rows = service.list_transaction_rows(**filters)
    st.dataframe(transaction_view(rows), width="stretch", hide_index=True)
    export_left, export_right = st.columns(2)
    with export_left:
        st.download_button(
            "Export this filter as CSV",
            data=service.export_csv(**filters),
            file_name="transactions.csv",
            mime="text/csv",
        )
    with export_right:
        st.download_button(
            "Export this filter as Excel",
            data=service.export_excel(**filters),
            file_name="transactions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    if not rows:
        st.info("No transactions match.")
        return
    labels = [
        f"#{row['id']} {row['transaction_date']} {row['amount']} {row['currency']} {row.get('description') or ''}"
        for row in rows
    ]
    selected_label = st.selectbox("Choose a transaction", labels)
    selected = rows[labels.index(selected_label)]
    if selected["is_deleted"]:
        st.warning("This entry is soft-deleted and is left out of reports by default.")
        if st.button("Restore this transaction"):
            try:
                service.restore_transaction(int(selected["id"]))
            except (ValueError, KeyError) as exc:
                st.error(str(exc))
            else:
                st.session_state.flash = f"Restored transaction {selected['id']}。"
                st.rerun()
        return
    st.subheader("Edit transaction")
    initial = {
        "transaction_date": selected["transaction_date"],
        "amount_text": format(from_minor(int(selected["amount_minor"]), selected["currency"]), "f"),
        "currency": selected["currency"],
        "transaction_type": selected["transaction_type"],
        "category": selected.get("category_name"),
        "description": selected.get("description") or "",
        "merchant": selected.get("merchant") or "",
        "account": selected.get("account_name") or "",
        "possible_amounts": [],
        "original_transaction_id": selected.get("original_transaction_id"),
        "search_start": None,
        "search_end": None,
        "warnings": [],
        "clarification": None,
        "items": [],
        "raw_text": "",
        "source": selected.get("source") or "manual",
        "exclude_id": int(selected["id"]),
    }
    result = render_editor(service, prefix=f"edit_{selected['id']}", initial=initial, submit_label="Save changes")
    if result is not None:
        try:
            service.update_transaction(
                int(selected["id"]),
                transaction_date=result["transaction_date"],
                amount=result["amount"],
                currency=result["currency"],
                transaction_type=result["transaction_type"],
                category_name=result["category_name"],
                description=result["description"],
                merchant=result["merchant"],
                account_name=result["account_name"],
                original_transaction_id=result["original_transaction_id"],
                allow_duplicate=result["allow_duplicate"],
            )
        except DuplicateTransactionError:
            st.error("The edit is too similar to another transaction. Tick the confirmation.")
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
        else:
            st.session_state.flash = f"Updated transaction {selected['id']}。"
            st.rerun()
    confirmed = st.checkbox("I confirm that I want to delete this transaction", key=f"delete_confirm_{selected['id']}")
    if st.button("Delete transaction", key=f"delete_{selected['id']}"):
        if not confirmed:
            st.error("Tick the confirmation before deleting.")
        else:
            try:
                service.delete_transaction(int(selected["id"]), confirmed=True)
            except (ValueError, KeyError) as exc:
                st.error(str(exc))
            else:
                st.session_state.flash = f"Soft-deleted transaction {selected['id']}."
                st.rerun()


def page_budget(service: FinanceService) -> None:
    st.header("Budgets")
    st.caption("Budgets are monthly. Refunds reduce the category and the total. Currencies stay separate.")
    today = current_today()
    categories = EXPENSE_CATEGORIES
    left, right = st.columns(2)
    with left:
        choice = st.selectbox("Budget scope", ["Overall budget"] + categories)
        currency = st.selectbox("Budget currency", CURRENCIES, index=CURRENCIES.index(current_settings().default_currency) if current_settings().default_currency in CURRENCIES else 0)
        amount_text = st.text_input("Monthly amount", value="0")
        if st.button("Save budget", type="primary"):
            amount = try_amount(amount_text)
            if amount is None and amount_text.strip() not in {"0", "0.0", "0.00"}:
                st.error("Enter 0 or a positive amount.")
            else:
                try:
                    service.set_budget(
                        category_name=None if choice == "Overall budget" else choice,
                        currency=currency,
                        amount=Decimal(amount_text.strip().replace(",", "")),
                    )
                except (ValueError, InvalidOperation) as exc:
                    st.error(str(exc))
                else:
                    st.session_state.flash = "Budget saved."
                    st.rerun()
    statuses = {status.budget_id: status for status in service.budget_status(year=today.year, month=today.month)}
    with right:
        st.write(f"{today.year}-{today.month:02d} usage")
    for budget in service.list_budgets():
        status = statuses.get(int(budget["id"]))
        name = budget.get("category_name") or "Total spending"
        st.markdown(f"**{name} · {budget['currency']}** per month {budget['limit']}")
        if status:
            ratio = 0.0 if status.ratio <= 0 else min(float(status.ratio), 1.0)
            st.progress(ratio)
            st.caption(f"Net expense this month {format_money(status.spent_minor, status.currency)}")
        if st.button("Delete this budget", key=f"budget_delete_{budget['id']}"):
            service.delete_budget(int(budget["id"]))
            st.rerun()


def page_settings(service: FinanceService) -> None:
    st.header("Settings")
    settings = current_settings()
    host = st.text_input("Ollama address", value=settings.ollama_host)
    detected: list[str] = []
    detect_error = ""
    if st.button("Check local models"):
        try:
            detected = list_installed_models(host)
        except Exception as exc:
            detect_error = str(exc)
        else:
            st.session_state.detected_models = detected
    detected = st.session_state.get("detected_models", [])
    if detect_error:
        st.error(detect_error)
    text_options = detected or [settings.text_model]
    if settings.text_model not in text_options:
        text_options = [settings.text_model, *text_options]
    vision_options = detected or [settings.vision_model]
    if settings.vision_model not in vision_options:
        vision_options = [settings.vision_model, *vision_options]
    text_model = st.selectbox("Text model", text_options, index=text_options.index(settings.text_model))
    vision_model = st.selectbox("Vision model", vision_options, index=vision_options.index(settings.vision_model))
    currency = st.selectbox(
        "Default currency",
        CURRENCIES,
        index=CURRENCIES.index(settings.default_currency) if settings.default_currency in CURRENCIES else 0,
    )
    timezone_name = st.text_input("Timezone", value=settings.timezone, help="For example Europe/Oslo")
    whisper_model = st.selectbox(
        "Speech model",
        ["tiny", "base", "small", "medium"],
        index=["tiny", "base", "small", "medium"].index(settings.whisper_model)
        if settings.whisper_model in {"tiny", "base", "small", "medium"}
        else 2,
    )
    device = st.selectbox("Speech device", ["cpu", "cuda", "auto"], index=["cpu", "cuda", "auto"].index(settings.whisper_device))
    language_labels = {"auto": "Automatic", "en": "English", "no": "Norwegian"}
    stored_language = settings.whisper_language if settings.whisper_language in language_labels else "auto"
    language_choice = st.selectbox(
        "Speech language",
        list(language_labels.values()),
        index=list(language_labels.keys()).index(stored_language),
    )
    language = {label: code for code, label in language_labels.items()}[language_choice]
    st.caption("Every transaction is confirmed first. The thresholds below only add a warning.")
    threshold_cols = st.columns(3)
    thresholds = dict(settings.thresholds)
    for column, code in zip(threshold_cols, ("CNY", "NOK", "USD")):
        thresholds[code] = column.text_input(f"{code} warning threshold", value=str(thresholds.get(code, "")), key=f"threshold_{code}")
    if st.button("Save settings", type="primary"):
        updated = Settings(
            ollama_host=host.strip(),
            text_model=text_model,
            vision_model=vision_model,
            default_currency=currency,
            timezone=timezone_name.strip(),
            whisper_model=whisper_model,
            whisper_device=device,
            whisper_compute_type="int8" if device == "cpu" else settings.whisper_compute_type,
            whisper_language=language,
            confirm_all=True,
            thresholds=thresholds,
        )
        try:
            ZoneInfo(updated.timezone)
            save_settings(updated)
        except Exception as exc:
            st.error(f"Settings were not saved: {exc}")
        else:
            st.session_state.settings = load_settings()
            st.success("Settings saved to data/settings.json on this computer.")
    st.subheader("Export every active transaction")
    st.download_button("Export CSV", data=service.export_csv(), file_name="transactions.csv", mime="text/csv")
    st.download_button(
        "Export Excel",
        data=service.export_excel(),
        file_name="transactions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.caption(f"Database file: {DB_PATH}")
    st.subheader("Recent activity")
    with get_connection(DB_PATH) as connection:
        logs = list_audit_logs(connection, limit=30)
    st.dataframe(logs, width="stretch", hide_index=True)
    st.caption("The activity log keeps the action and field names only, not amounts, merchants, or descriptions.")


@st.cache_data(ttl=30, show_spinner=False)
def cached_text_model_status(host: str, text_model: str) -> tuple[str, str]:
    """Cache the model check briefly so every click does not wait on local Ollama."""
    try:
        models = list_installed_models(host, timeout=2)
    except Exception as exc:
        return "error", str(exc)
    if text_model in models:
        return "ok", text_model
    return "missing", text_model


def render_sidebar() -> str:
    st.sidebar.title("Local ledger")
    st.sidebar.caption("The ledger is stored in SQLite on this computer. There is no paid cloud database.")
    settings = current_settings()
    status, detail = cached_text_model_status(settings.ollama_host, settings.text_model)
    if status == "error":
        st.sidebar.error("Ollama is not connected. Ordinary sentences still use the rules. Receipts need Ollama.")
    elif status == "ok":
        st.sidebar.success(f"Text model installed: {detail}")
    else:
        st.sidebar.warning(f"Text model not found on this computer: {detail}. Simple sentences still use the rules.")
    return st.sidebar.radio(
        "Pages",
        ["Overview", "Assistant", "Add transaction", "Transactions", "Budgets", "Settings"],
    )


def main() -> None:
    st.set_page_config(page_title="Local ledger", page_icon="📒", layout="wide")
    ensure_state()
    service = get_service()
    show_flash()
    page = render_sidebar()
    pages = {
        "Overview": page_dashboard,
        "Assistant": page_agent,
        "Add transaction": page_add,
        "Transactions": page_history,
        "Budgets": page_budget,
        "Settings": page_settings,
    }
    pages[page](service)


if __name__ == "__main__":
    main()
