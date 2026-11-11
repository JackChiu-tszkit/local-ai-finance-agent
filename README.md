# Local ledger

A personal finance app that runs on your own computer. English and Norwegian sentences, receipt images, and speech become drafts first. They are written to the local SQLite database only after you confirm them. Totals, duplicates, refunds, and budgets are not calculated by a language model.

By default it does not call a paid cloud API. Text and receipts go only to the Ollama address in settings, which is `http://127.0.0.1:11434` unless you change it. Speech is transcribed on this computer by faster-whisper.

## Layout

```text
ai-finance-agent/
├── app.py                 # Streamlit interface
├── agent.py               # Multi-turn drafts, ledger questions, category preview
├── ai_parser.py           # Natural-language parsing and Ollama
├── receipt_parser.py      # Receipts and screenshots
├── voice_parser.py        # Local speech to text
├── finance_service.py     # Money, duplicates, refunds, budgets, export
├── database.py            # SQLite access
├── db_setup.py            # Schema and built-in categories
├── config.py              # Paths, currencies, model settings
├── requirements.txt
├── data/                  # Database, settings, and exports; not in Git
└── tests/
```

## Requirements

- Python 3.11 or newer
- Windows 11 or macOS (including Apple Silicon)
- Optional: local [Ollama](https://ollama.com/) for richer sentences and receipt reading
- An NVIDIA GPU is not required. On Windows, set the speech device to `cuda` if you want the GPU; if that fails, it falls back to CPU. On macOS, Ollama uses Apple Metal and faster-whisper stays on CPU.

## Windows 11 (PowerShell)

```powershell
cd $HOME\ai-finance-agent
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

If the activation script is blocked, allow it for the current window only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

After installing Ollama, pull the models:

```powershell
ollama pull qwen2.5:7b
ollama pull qwen2.5vl:7b
```

If memory is tight, use `qwen2.5:3b` for text and select it under Settings.

Initialize the database, run the tests, and start the app:

```powershell
python db_setup.py
python -m pytest -q
streamlit run app.py
```

The browser opens the local page, usually `http://localhost:8501`. On Windows, `launch_app.py` can open the same page in a desktop window.

## macOS (Terminal, including Apple Silicon)

```bash
cd ~/ai-finance-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

Install Ollama from its site, or with:

```bash
brew install ollama
ollama serve
```

In another terminal, pull the models:

```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5vl:7b
```

Then:

```bash
python db_setup.py
python -m pytest -q
streamlit run app.py
```

## Daily use

1. Open Assistant and keep talking. Record one entry, then say "change it to last Wednesday" or "merchant is Kiwi". You can also ask "how much did Dining cost this month". Several amounts in one sentence are split and confirmed one by one. Questions and category changes are shown first and written only after you confirm.
2. Open Add transaction to type a sentence, upload a receipt, or record speech.
3. Check the date, amount, currency, category, type, description, and merchant. After you save, that merchant's category is remembered.
4. If a similar past entry is found, tick the checkbox before saving. The old row is not overwritten.
5. A refund must point at the original expense. It is stored as its own row and reduces that category's net expense.
6. Under Budgets, set a monthly total or a category limit for one currency. At 80% and above 100%, Overview shows a warning. A category that is well above your previous three months is also flagged.
7. Export CSV or Excel from Transactions or Settings. You can say "move all last month taxi to Transport", review the preview, and confirm before any category changes.

Try these:

- Lunch cost 120 yuan today.
- Spent 350 NOK at Kiwi today.
- Received salary 45000 NOK today.
- Yesterday the bus cost 50 kroner.
- I dag brukte jeg 80 kroner.

If the date is missing, the app fills in today in your timezone and asks you to confirm. If the currency is missing, it uses the default. A merchant or amount that the model adds but that is not in your sentence is dropped.

When Ollama is not running, common English and Norwegian sentences still use the rules, and the page says the connection failed. Receipts need the local vision model. The app does not guess a receipt.

The first time you use speech, faster-whisper downloads an open model into the local cache. The audio file exists only while it is being transcribed, then it is deleted. It is not uploaded to a speech service.

## Where the data lives

- Database: `data/finance.db`
- Settings: `data/settings.json`
- Both, and the upload folder, are listed in `.gitignore`

Amounts are stored as integer minor units. NOK keeps two decimal places. Currencies are never added together, and there is no exchange-rate conversion.

Delete is a soft delete. The audit log stores field names only, not amounts, merchants, or descriptions. The transaction row remains and can be restored.

## Tests

```bash
python -m pytest -q
```

The tests cover rounding, database transactions, foreign keys, duplicates, refund limits, budget warnings, category totals, CSV and Excel export, draft edits, ledger questions, and the rule parser without Ollama. The speech test uses a fake transcriber and does not download a model.
