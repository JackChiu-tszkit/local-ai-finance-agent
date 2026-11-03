"""本地智能记账的 Streamlit 界面。

所有写入都发生在用户点击“确认保存”之后。
页面重跑不会重复入账：已保存的确认令牌会被记住。
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


TYPE_OPTIONS = [("expense", "支出"), ("income", "收入"), ("refund", "退款")]
CURRENCIES = ["NOK", "CNY", "USD", "EUR", "GBP", "SEK", "DKK", "JPY"]
FREQUENCIES = [("day", "按日"), ("week", "按周"), ("month", "按月")]


def get_service() -> FinanceService:
    """打开或创建本地数据库，并返回业务服务。"""
    return FinanceService(DB_PATH)


def ensure_state() -> None:
    """初始化会话。设置和待确认交易都放在 Session State 中。"""
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
    """把草稿放进会话。此时尚未写数据库。"""
    st.session_state.pending = {
        "token": uuid.uuid4().hex,
        "draft": draft.model_dump(mode="json"),
    }


def remember_saved(service: FinanceService, result: dict) -> None:
    """确认保存后记住商户分类，供下一次解析使用。"""
    service.remember_merchant(result.get("merchant"), result["category_name"], result["transaction_type"])


def stage_with_memory(service: FinanceService, draft: TransactionDraft) -> None:
    """单次解析也使用已经记住的商户分类。"""
    stage_draft(apply_merchant_memory(draft, service.merchant_memories()))


def try_amount(text: str) -> Decimal | None:
    """解析用户输入的金额。格式不对时返回空，便于页面继续渲染。"""
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
    for label, items in (("完全重复", report.exact), ("疑似重复", report.suspected)):
        for item in items:
            rows.append(
                {
                    "匹配": label,
                    "编号": item["id"],
                    "日期": item["transaction_date"],
                    "金额": item["amount"],
                    "币种": item["currency"],
                    "类型": item["type_label"],
                    "商户": item.get("merchant") or "",
                    "描述": item.get("description") or "",
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
    """渲染可修改的确认表单。只有点击按钮时才返回字段。"""
    for warning in initial.get("warnings") or []:
        st.warning(warning)
    if initial.get("clarification"):
        st.info(initial["clarification"])
    if initial.get("items"):
        st.markdown("**识别到的明细**")
        for item in initial["items"]:
            st.write(f"- {item}")
    if initial.get("raw_text"):
        with st.expander("原始内容"):
            st.write(initial["raw_text"])

    type_labels = [label for _code, label in TYPE_OPTIONS]
    type_codes = [code for code, _label in TYPE_OPTIONS]
    current_type = initial.get("transaction_type") or "expense"
    if current_type not in type_codes:
        current_type = "expense"
    selected_label = st.selectbox(
        "收支类型",
        type_labels,
        index=type_codes.index(current_type),
        key=f"{prefix}_type",
    )
    transaction_type = type_codes[type_labels.index(selected_label)]
    if transaction_type == "expense":
        categories = EXPENSE_CATEGORIES
    elif transaction_type == "refund":
        categories = ["退款"]
    else:
        categories = INCOME_CATEGORIES
    category_value = initial.get("category")
    if category_value not in categories:
        category_value = categories[0]
    category = st.selectbox(
        "分类",
        categories,
        index=categories.index(category_value),
        key=f"{prefix}_category_{transaction_type}",
    )
    date_value = initial.get("transaction_date") or current_today()
    if isinstance(date_value, str):
        date_value = date.fromisoformat(date_value)
    transaction_date = st.date_input("日期", value=date_value, key=f"{prefix}_date")
    currency_options = list(CURRENCIES)
    currency_value = (initial.get("currency") or current_settings().default_currency).upper()
    if currency_value not in currency_options:
        currency_options.append(currency_value)
    currency = st.selectbox(
        "币种",
        currency_options,
        index=currency_options.index(currency_value),
        key=f"{prefix}_currency",
    )
    possible = initial.get("possible_amounts") or []
    if possible:
        choice = st.radio(
            "识别到多个金额，请选择实际支付金额",
            possible + ["手动输入"],
            key=f"{prefix}_amount_choice",
        )
        if choice == "手动输入":
            amount_text = st.text_input("金额", value="", key=f"{prefix}_amount")
        else:
            amount_text = choice
            st.caption(f"当前选择的金额：{choice}")
    else:
        amount_text = st.text_input(
            "金额",
            value=initial.get("amount_text") or "",
            key=f"{prefix}_amount",
        )
    description = st.text_input("描述", value=initial.get("description") or "", key=f"{prefix}_description")
    merchant = st.text_input("商户", value=initial.get("merchant") or "", key=f"{prefix}_merchant")
    account = st.text_input("账户", value=initial.get("account") or "", key=f"{prefix}_account")
    amount = try_amount(amount_text)
    if amount is not None:
        limit = threshold_for(currency)
        if limit is not None and amount >= limit:
            st.warning(f"金额达到 {currency} {limit} 的提示线。当前版本无论金额大小，都要确认后才保存。")

    original_id = None
    if transaction_type == "refund":
        if amount is None:
            st.info("填写退款金额后，会列出可以关联的支出。")
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
                st.error("没有找到仍可退款的支出。")
            else:
                labels = []
                ids = []
                for item in candidates:
                    labels.append(
                        f"#{item['id']} {item['transaction_date']} {item['amount']} {item['currency']} "
                        f"{item.get('category_name') or ''} {item.get('merchant') or ''} 剩余 {item['remaining']}"
                    )
                    ids.append(int(item["id"]))
                wanted = initial.get("original_transaction_id")
                index = ids.index(wanted) if wanted in ids else 0
                selected = st.selectbox("原始支出", labels, index=index, key=f"{prefix}_original")
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
            st.warning("发现相似的历史交易。继续保存不会覆盖或删除旧记录。")
            st.dataframe(duplicate_rows(report), width="stretch", hide_index=True)
            allow_duplicate = st.checkbox("我已核对，仍然保存", key=f"{prefix}_allow_duplicate")

    submitted = st.button(submit_label, type="primary", key=f"{prefix}_submit")
    if not submitted:
        return None
    if amount is None:
        st.error("请填写大于 0 的金额。")
        return None
    if not description.strip():
        st.error("请填写描述。")
        return None
    if transaction_type == "refund" and original_id is None:
        st.error("请选择要退款的原始支出。")
        return None
    if report is not None and report.has_any and not allow_duplicate:
        st.error("请先勾选相似交易确认，或返回修改。")
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
    """展示待确认交易。保存成功后清除草稿，避免重跑再次写入。"""
    pending = st.session_state.pending
    if not pending:
        return
    token = pending["token"]
    if token in st.session_state.saved_tokens:
        st.info("这笔交易刚刚已经保存。")
        return
    try:
        draft = TransactionDraft.model_validate(pending["draft"])
    except ValidationError:
        st.error("待确认的数据已损坏，请重新解析。")
        st.session_state.pending = None
        return
    st.subheader("确认交易")
    st.caption("可以修改下面的字段。点击“确认保存”后才会写入本地数据库。")
    result = render_editor(
        service,
        prefix=token,
        initial=initial_from_draft(draft),
        submit_label="确认保存",
    )
    if result is None:
        return
    try:
        transaction_id = service.create_transaction(**result)
    except DuplicateTransactionError:
        st.error("保存前又检测到相似交易，请勾选确认后重新保存。")
        return
    except (ValueError, KeyError) as exc:
        st.error(str(exc))
        return
    st.session_state.saved_tokens.append(token)
    st.session_state.saved_tokens = st.session_state.saved_tokens[-200:]
    remember_saved(service, result)
    st.session_state.pending = None
    st.session_state.flash = f"已保存，交易编号 {transaction_id}。"
    st.rerun()


def page_add(service: FinanceService) -> None:
    st.header("新增交易")
    st.caption("支持自然语言、收据图片和语音。三种方式都会先生成草稿，确认后才入账。")
    settings = current_settings()
    today = current_today()
    text_tab, receipt_tab, voice_tab, manual_tab = st.tabs(["自然语言", "收据图片", "语音", "手动填写"])
    with text_tab:
        text = st.text_area(
            "用一句话记账",
            placeholder="例如：今天在 Kiwi 买菜花了350 NOK。",
            height=120,
        )
        if st.button("解析文字", type="primary"):
            if not text.strip():
                st.warning("请输入交易内容。")
            else:
                draft = parse_transaction_text(
                    text,
                    today=today,
                    default_currency=settings.default_currency,
                    settings=settings,
                )
                stage_with_memory(service, draft)
    with receipt_tab:
        image = st.file_uploader("上传 PNG、JPG 或 JPEG", type=["png", "jpg", "jpeg"])
        st.caption("图片只发送到设置中的 Ollama 地址，默认是本机，不会使用外部视觉 API。")
        if st.button("识别收据", type="primary"):
            if image is None:
                st.warning("请先选择图片。")
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
                        st.error(f"收据识别失败：{exc}")
                else:
                    stage_with_memory(service, draft)
    with voice_tab:
        st.caption("语音在本机转成文字，再使用和键盘输入相同的解析与确认流程。")
        audio_file = st.file_uploader(
            "上传音频",
            type=["wav", "mp3", "m4a", "ogg", "flac", "webm", "aac"],
        )
        recorded = st.audio_input("或直接录音") if hasattr(st, "audio_input") else None
        if not hasattr(st, "audio_input"):
            st.info("当前 Streamlit 版本没有内置录音组件，请上传音频文件。")
        if "transcript_box" not in st.session_state:
            st.session_state.transcript_box = ""
        if st.session_state.pop("transcript_dirty", False):
            st.session_state.transcript_box = st.session_state.transcript
        edited = st.text_area("转写文字可以修改后再解析", key="transcript_box")
        action_col, reparse_col = st.columns(2)
        with action_col:
            transcribe_clicked = st.button("转写并解析", type="primary")
        with reparse_col:
            reparse_clicked = st.button("按上面的文字重新解析")
        chosen = recorded or audio_file
        if transcribe_clicked:
            if chosen is None:
                st.warning("请先录音或上传音频。")
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
                st.warning("没有可解析的文字。")
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
        st.write("不使用模型时，可以手动填写。保存前同样需要点击确认。")
        if st.button("开始手动记账"):
            stage_draft(
                TransactionDraft(
                    transaction_date=today,
                    date_explicit=True,
                    currency=settings.default_currency,
                    currency_assumed=True,
                    transaction_type="expense",
                    type_confident=True,
                    category="其他",
                    description="",
                    source="manual",
                    parser="manual",
                )
            )
    render_pending(service)


def render_budget_alerts(service: FinanceService, today: date) -> None:
    """在总览顶部显示本月预算警告。80% 和超支都使用醒目的红色提示。"""
    for status in service.budget_status(year=today.year, month=today.month):
        if status.level == "ok":
            continue
        name = status.category_name or "总支出"
        spent = format_money(status.spent_minor, status.currency)
        limit = format_money(status.limit_minor, status.currency)
        if status.level == "over":
            st.error(f"已超出预算：{name}（{status.currency}）已支出 {spent}，预算 {limit}。")
        else:
            percent = (status.ratio * Decimal(100)).quantize(Decimal("0.1"))
            st.error(f"预算警告：{name}（{status.currency}）已使用 {percent}%，支出 {spent} / 预算 {limit}。")


def render_charts(summary) -> None:
    """把整数金额换成图表所需的显示值。汇总本身不使用浮点运算。"""
    currency = summary.currency
    labels = []
    values = []
    for name, minor in summary.category_net:
        if minor > 0:
            labels.append(name)
            values.append(float(from_minor(minor, currency)))
    if labels:
        pie = px.pie(names=labels, values=values, title=f"{currency} 分类净支出占比")
        st.plotly_chart(pie, width="stretch")
    else:
        st.info(f"{currency} 在当前筛选下没有正向净支出，因此不绘制饼图。")
    if not summary.periods:
        return
    income = [float(from_minor(value, currency)) for value in summary.period_income]
    net_expense = [float(from_minor(value, currency)) for value in summary.period_net_expense]
    bar = go.Figure()
    bar.add_bar(name="收入", x=summary.periods, y=income)
    bar.add_bar(name="净支出", x=summary.periods, y=net_expense)
    bar.update_layout(barmode="group", title=f"{currency} 收入和净支出")
    st.plotly_chart(bar, width="stretch")
    line = px.line(x=summary.periods, y=net_expense, markers=True, title=f"{currency} 净支出趋势")
    line.update_layout(xaxis_title="周期", yaxis_title=currency)
    st.plotly_chart(line, width="stretch")


def _queue_item(draft: TransactionDraft) -> dict:
    return {"token": uuid.uuid4().hex, "draft": draft.model_dump(mode="json")}


def page_agent(service: FinanceService) -> None:
    """对话式记账。提问和修改草稿都不会直接写入数据库。"""
    st.header("记账助手")
    st.caption("可以连续补充草稿，也可以询问本机账本。金额由数据库计算。点击确认后才入账。")
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
            if st.button("确认修改这些分类", type="primary"):
                changed = service.recategorize(proposal["transaction_ids"], proposal["category_name"])
                st.session_state.agent_proposal = None
                st.session_state.agent_messages.append(
                    {"role": "assistant", "content": f"已修改 {changed} 笔支出的分类。"}
                )
                st.session_state.agent_messages = st.session_state.agent_messages[-40:]
                st.rerun()
        with cancel_col:
            if st.button("取消这次修改"):
                st.session_state.agent_proposal = None
                st.session_state.agent_messages.append({"role": "assistant", "content": "已取消，账本没有改动。"})
                st.rerun()

    queue = st.session_state.agent_queue
    if queue:
        st.subheader(f"待确认 {1} / {len(queue)}")
        try:
            draft = TransactionDraft.model_validate(queue[0]["draft"])
        except ValidationError:
            st.session_state.agent_queue = queue[1:]
            st.error("一条草稿无法读取，已跳过。")
            st.rerun()
        result = render_editor(
            service,
            prefix=f"agent_{queue[0]['token']}",
            initial=initial_from_draft(draft),
            submit_label="确认保存这一笔",
        )
        if st.button("取消当前草稿"):
            st.session_state.agent_queue = queue[1:]
            st.session_state.agent_messages.append({"role": "assistant", "content": "已取消当前草稿，没有写入。"})
            st.rerun()
        if result is not None:
            try:
                transaction_id = service.create_transaction(**result)
            except DuplicateTransactionError:
                st.error("保存前又检测到相似交易，请勾选确认后重新保存。")
                return
            except (ValueError, KeyError) as exc:
                st.error(str(exc))
                return
            remember_saved(service, result)
            st.session_state.agent_queue = st.session_state.agent_queue[1:]
            st.session_state.agent_messages.append(
                {"role": "assistant", "content": f"已保存，交易编号 {transaction_id}。"}
            )
            st.session_state.agent_messages = st.session_state.agent_messages[-40:]
            st.session_state.flash = f"已保存，交易编号 {transaction_id}。"
            st.rerun()

    prompt = st.chat_input("例如：午饭花了80元，地铁花了30元。也可以问这个月餐饮花了多少。")
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
    st.header("财务总览")
    today = current_today()
    render_budget_alerts(service, today)
    for note in service.spending_insights(today):
        st.warning(note)
    st.caption("不同币种分开统计，不进行汇率换算。支出已扣除同期退款。")
    categories = service.list_categories()
    category_labels = ["全部"] + [f"{item['name']}（{'支出' if item['category_type']=='expense' else '收入'}）" for item in categories]
    filter_col, type_col, currency_col = st.columns(3)
    with filter_col:
        start_date = st.date_input("开始日期", value=today.replace(day=1), key="dash_start")
        end_date = st.date_input("结束日期", value=today, key="dash_end")
    with type_col:
        category_choice = st.selectbox("分类", category_labels, key="dash_category")
        type_choice = st.selectbox("收支类型", ["全部", "支出", "收入", "退款"], key="dash_type")
    with currency_col:
        keyword = st.text_input("关键词", key="dash_keyword")
        frequency_label = st.radio("统计周期", [label for _code, label in FREQUENCIES], horizontal=True)
    frequency = dict((label, code) for code, label in FREQUENCIES)[frequency_label]
    category_id = None
    if category_choice != "全部":
        category_id = categories[category_labels.index(category_choice) - 1]["id"]
    type_map = {"全部": None, "支出": "expense", "收入": "income", "退款": "refund"}
    known_currencies = []
    for row in service.list_transaction_rows():
        if row["currency"] not in known_currencies:
            known_currencies.append(row["currency"])
    default_currency = current_settings().default_currency
    currency_options = ["全部"] + known_currencies
    preferred = default_currency if default_currency in known_currencies else "全部"
    currency_choice = st.selectbox(
        "币种",
        currency_options,
        index=currency_options.index(preferred),
        key="dash_currency",
    )
    summaries = service.summarize(
        start_date=start_date,
        end_date=end_date,
        currency=None if currency_choice == "全部" else currency_choice,
        category_id=category_id,
        transaction_type=type_map[type_choice],
        keyword=keyword.strip() or None,
        frequency=frequency,
    )
    if not summaries:
        st.info("这个范围内还没有交易。")
    for summary in summaries:
        st.subheader(summary.currency)
        columns = st.columns(5)
        values = [
            ("总收入", summary.total_income_minor),
            ("总支出", summary.total_expense_minor),
            ("退款", summary.total_refund_minor),
            ("净支出", summary.net_expense_minor),
            ("结余", summary.balance_minor),
        ]
        for column, (label, minor) in zip(columns, values):
            column.metric(label, format_money(minor, summary.currency))
        render_charts(summary)
    st.subheader("最近交易")
    recent = service.list_transaction_rows(
        start_date=start_date,
        end_date=end_date,
        currency=None if currency_choice == "全部" else currency_choice,
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
                "编号": row["id"],
                "日期": row["transaction_date"],
                "金额": row["amount"],
                "币种": row["currency"],
                "类型": row["type_label"],
                "分类": row.get("category_name") or "",
                "商户": row.get("merchant") or "",
                "描述": row.get("description") or "",
                "账户": row.get("account_name") or "",
                "原交易": row.get("original_transaction_id") or "",
                "状态": "已删除" if row.get("is_deleted") else "有效",
            }
        )
    return view


def page_history(service: FinanceService) -> None:
    st.header("交易记录")
    today = current_today()
    keyword = st.text_input("搜索描述、商户或账户")
    left, middle, right = st.columns(3)
    with left:
        start_date = st.date_input("开始", value=date(today.year, 1, 1), key="history_start")
        end_date = st.date_input("结束", value=today, key="history_end")
    with middle:
        currency = st.selectbox("币种筛选", ["全部"] + CURRENCIES, key="history_currency")
        type_label = st.selectbox("类型筛选", ["全部", "支出", "收入", "退款"], key="history_type")
    with right:
        include_deleted = st.checkbox("显示已删除")
        categories = service.list_categories()
        category_labels = ["全部"] + [item["name"] + (" / 支出" if item["category_type"] == "expense" else " / 收入") for item in categories]
        category_choice = st.selectbox("分类筛选", category_labels)
    type_map = {"全部": None, "支出": "expense", "收入": "income", "退款": "refund"}
    category_id = None if category_choice == "全部" else categories[category_labels.index(category_choice) - 1]["id"]
    filters = {
        "start_date": start_date,
        "end_date": end_date,
        "currency": None if currency == "全部" else currency,
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
            "导出当前筛选 CSV",
            data=service.export_csv(**filters),
            file_name="transactions.csv",
            mime="text/csv",
        )
    with export_right:
        st.download_button(
            "导出当前筛选 Excel",
            data=service.export_excel(**filters),
            file_name="transactions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    if not rows:
        st.info("没有符合条件的交易。")
        return
    labels = [
        f"#{row['id']} {row['transaction_date']} {row['amount']} {row['currency']} {row.get('description') or ''}"
        for row in rows
    ]
    selected_label = st.selectbox("选择一笔交易", labels)
    selected = rows[labels.index(selected_label)]
    if selected["is_deleted"]:
        st.warning("这笔记账已软删除，报表默认不统计它。")
        if st.button("恢复这笔交易"):
            try:
                service.restore_transaction(int(selected["id"]))
            except (ValueError, KeyError) as exc:
                st.error(str(exc))
            else:
                st.session_state.flash = f"已恢复交易 {selected['id']}。"
                st.rerun()
        return
    st.subheader("编辑交易")
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
    result = render_editor(service, prefix=f"edit_{selected['id']}", initial=initial, submit_label="保存修改")
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
            st.error("修改后与另一笔交易过于相似，请勾选确认。")
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
        else:
            st.session_state.flash = f"已更新交易 {selected['id']}。"
            st.rerun()
    confirmed = st.checkbox("我确认要删除这笔交易", key=f"delete_confirm_{selected['id']}")
    if st.button("删除交易", key=f"delete_{selected['id']}"):
        if not confirmed:
            st.error("删除前需要勾选确认。")
        else:
            try:
                service.delete_transaction(int(selected["id"]), confirmed=True)
            except (ValueError, KeyError) as exc:
                st.error(str(exc))
            else:
                st.session_state.flash = f"已软删除交易 {selected['id']}。"
                st.rerun()


def page_budget(service: FinanceService) -> None:
    st.header("预算设置")
    st.caption("预算按月计算。退款会减少对应分类和总支出的净额。不同币种不会混在一起。")
    today = current_today()
    categories = EXPENSE_CATEGORIES
    left, right = st.columns(2)
    with left:
        choice = st.selectbox("预算范围", ["总预算"] + categories)
        currency = st.selectbox("预算币种", CURRENCIES, index=CURRENCIES.index(current_settings().default_currency) if current_settings().default_currency in CURRENCIES else 0)
        amount_text = st.text_input("每月金额", value="0")
        if st.button("保存预算", type="primary"):
            amount = try_amount(amount_text)
            if amount is None and amount_text.strip() not in {"0", "0.0", "0.00"}:
                st.error("请输入 0 或正数金额。")
            else:
                try:
                    service.set_budget(
                        category_name=None if choice == "总预算" else choice,
                        currency=currency,
                        amount=Decimal(amount_text.strip().replace(",", "")),
                    )
                except (ValueError, InvalidOperation) as exc:
                    st.error(str(exc))
                else:
                    st.session_state.flash = "预算已保存。"
                    st.rerun()
    statuses = {status.budget_id: status for status in service.budget_status(year=today.year, month=today.month)}
    with right:
        st.write(f"{today.year} 年 {today.month} 月使用情况")
    for budget in service.list_budgets():
        status = statuses.get(int(budget["id"]))
        name = budget.get("category_name") or "总支出"
        st.markdown(f"**{name} · {budget['currency']}** 每月 {budget['limit']}")
        if status:
            ratio = 0.0 if status.ratio <= 0 else min(float(status.ratio), 1.0)
            st.progress(ratio)
            st.caption(f"本月净支出 {format_money(status.spent_minor, status.currency)}")
        if st.button("删除此预算", key=f"budget_delete_{budget['id']}"):
            service.delete_budget(int(budget["id"]))
            st.rerun()


def page_settings(service: FinanceService) -> None:
    st.header("系统设置")
    settings = current_settings()
    host = st.text_input("Ollama 地址", value=settings.ollama_host)
    detected: list[str] = []
    detect_error = ""
    if st.button("检测本机模型"):
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
    text_model = st.selectbox("文本模型", text_options, index=text_options.index(settings.text_model))
    vision_model = st.selectbox("视觉模型", vision_options, index=vision_options.index(settings.vision_model))
    currency = st.selectbox(
        "默认币种",
        CURRENCIES,
        index=CURRENCIES.index(settings.default_currency) if settings.default_currency in CURRENCIES else 0,
    )
    timezone_name = st.text_input("时区", value=settings.timezone, help="例如 Europe/Oslo 或 Asia/Shanghai")
    whisper_model = st.selectbox(
        "语音模型",
        ["tiny", "base", "small", "medium"],
        index=["tiny", "base", "small", "medium"].index(settings.whisper_model)
        if settings.whisper_model in {"tiny", "base", "small", "medium"}
        else 2,
    )
    device = st.selectbox("语音设备", ["cpu", "cuda", "auto"], index=["cpu", "cuda", "auto"].index(settings.whisper_device))
    language_labels = {"auto": "自动", "zh": "中文", "en": "英文", "no": "挪威语"}
    language_choice = st.selectbox(
        "语音语言",
        list(language_labels.values()),
        index=list(language_labels.keys()).index(settings.whisper_language),
    )
    language = {label: code for code, label in language_labels.items()}[language_choice]
    st.caption("所有交易都会先确认。下面的阈值只用于额外提醒，不会跳过确认。")
    threshold_cols = st.columns(3)
    thresholds = dict(settings.thresholds)
    for column, code in zip(threshold_cols, ("CNY", "NOK", "USD")):
        thresholds[code] = column.text_input(f"{code} 提醒阈值", value=str(thresholds.get(code, "")), key=f"threshold_{code}")
    if st.button("保存设置", type="primary"):
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
            st.error(f"设置没有保存：{exc}")
        else:
            st.session_state.settings = load_settings()
            st.success("设置已保存，只写入本机 data/settings.json。")
    st.subheader("导出全部有效交易")
    st.download_button("导出 CSV", data=service.export_csv(), file_name="transactions.csv", mime="text/csv")
    st.download_button(
        "导出 Excel",
        data=service.export_excel(),
        file_name="transactions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.caption(f"数据库文件：{DB_PATH}")
    st.subheader("最近操作")
    with get_connection(DB_PATH) as connection:
        logs = list_audit_logs(connection, limit=30)
    st.dataframe(logs, width="stretch", hide_index=True)
    st.caption("操作记录只保留动作和字段名，不复制金额、商户或描述。")


@st.cache_data(ttl=30, show_spinner=False)
def cached_text_model_status(host: str, text_model: str) -> tuple[str, str]:
    """短时间缓存模型探测，避免每次点击都等待本机 Ollama。"""
    try:
        models = list_installed_models(host, timeout=2)
    except Exception as exc:
        return "error", str(exc)
    if text_model in models:
        return "ok", text_model
    return "missing", text_model


def render_sidebar() -> str:
    st.sidebar.title("本地智能记账")
    st.sidebar.caption("账本保存在本机 SQLite，不使用付费云数据库。")
    settings = current_settings()
    status, detail = cached_text_model_status(settings.ollama_host, settings.text_model)
    if status == "error":
        st.sidebar.error("Ollama 未连接。普通句子仍可用规则解析，收据识别需要先启动 Ollama。")
    elif status == "ok":
        st.sidebar.success(f"文本模型已安装：{detail}")
    else:
        st.sidebar.warning(f"未在本机找到文本模型 {detail}。简单句式仍可用规则解析。")
    return st.sidebar.radio(
        "功能",
        ["财务总览", "记账助手", "新增交易", "交易记录", "预算设置", "系统设置"],
    )


def main() -> None:
    st.set_page_config(page_title="本地智能记账", page_icon="📒", layout="wide")
    ensure_state()
    service = get_service()
    show_flash()
    page = render_sidebar()
    pages = {
        "财务总览": page_dashboard,
        "记账助手": page_agent,
        "新增交易": page_add,
        "交易记录": page_history,
        "预算设置": page_budget,
        "系统设置": page_settings,
    }
    pages[page](service)


if __name__ == "__main__":
    main()
