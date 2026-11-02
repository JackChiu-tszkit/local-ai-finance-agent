# 本地智能记账

一个在你自己电脑上运行的个人记账应用。自然语言、收据图片和语音都会先整理成待确认草稿，你核对后才写入本机 SQLite。金额汇总、重复判断、退款和预算都不交给大模型计算。

默认不调用付费云 API。文本和收据只发给设置里的 Ollama 地址，默认是 `http://127.0.0.1:11434`。语音由本机的 faster-whisper 转写。

## 目录

```text
ai-finance-agent/
├── app.py                 # Streamlit 界面
├── ai_parser.py           # 自然语言解析与 Ollama 调用
├── receipt_parser.py      # 收据和截图识别
├── voice_parser.py        # 本地语音转文字
├── finance_service.py     # 金额、去重、退款、预算、统计、导出
├── database.py            # SQLite 读写
├── db_setup.py            # 建库和预设分类
├── config.py              # 路径、币种、模型配置
├── requirements.txt
├── data/                  # 数据库、设置和导出，不进入 Git
└── tests/
```

## 环境要求

- Python 3.11 或更高版本
- Windows 11 或 macOS（含 Apple Silicon）
- 可选：本机 [Ollama](https://ollama.com/)，用于自然语言增强和收据识别
- 不需要 NVIDIA 显卡。Windows 上如果要让语音走 GPU，可在设置里把设备改成 `cuda`；失败时会自动回到 CPU。macOS 上的 Ollama 会使用 Apple Metal，faster-whisper 默认使用 CPU。

## Windows 11（PowerShell）

```powershell
cd $HOME\ai-finance-agent
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

如果激活脚本被执行策略拦住，可以只对当前窗口放行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

安装 Ollama 后拉取模型：

```powershell
ollama pull qwen2.5:7b
ollama pull qwen2.5vl:7b
```

内存比较紧张时，文本模型可以改用 `qwen2.5:3b`，再到应用的“系统设置”里选择它。

初始化数据库、运行测试并启动：

```powershell
python db_setup.py
python -m pytest -q
streamlit run app.py
```

浏览器会打开本地页面，通常是 `http://localhost:8501`。

## macOS（Terminal，含 Apple Silicon）

```bash
cd ~/ai-finance-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

安装 Ollama 可以使用官网安装包，或：

```bash
brew install ollama
ollama serve
```

另一个终端里拉取模型：

```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5vl:7b
```

然后：

```bash
python db_setup.py
python -m pytest -q
streamlit run app.py
```

## 日常使用

1. 在“新增交易”里输入一句话、上传收据，或录音。
2. 核对日期、金额、币种、分类、收支类型、描述和商户。
3. 如果提示可能重复，需要勾选后才能保存。旧记录不会被覆盖。
4. 退款要选择原来的支出。退款单独记账，并冲减该分类的净支出。
5. 在“预算设置”里按币种设置每月总预算或分类预算。达到 80% 和超过 100% 时，总览顶部会显示红色警告。
6. 在“交易记录”或“系统设置”导出 CSV 和 Excel。

可以试这几句：

- 中午吃了火锅，花了120元。
- 今天在 Kiwi 买菜花了350 NOK。
- 今天收到工资45000 NOK。
- 昨天坐公交花了50克朗。

没有写日期时，程序会按设置里的时区暂填今天，并要求你确认。没有写币种时使用默认币种。模型如果补充了原文里不存在的商户或金额，会被丢掉。

Ollama 没启动时，常见中文和英文句式仍会用规则解析，页面会显示连接失败。收据识别必须有本机视觉模型，不会改用猜测结果。

第一次使用语音时，faster-whisper 会把开源模型下载到本机缓存。音频文件只在转写时临时落盘，转写后删除，不会上传到语音识别服务。

## 数据放在哪里

- 数据库：`data/finance.db`
- 个人设置：`data/settings.json`
- 这两个文件以及上传目录都在 `.gitignore` 里

金额以最小货币单位的整数保存。人民币和挪威克朗保留两位小数。不同币种不会相加，也不做汇率换算。

删除是软删除。审计记录只保存“改了哪些字段”，不复制金额、商户和描述。交易本身仍留在数据库里，可以恢复。

## 测试

```bash
python -m pytest -q
```

测试覆盖金额舍入、数据库事务、外键、重复交易、退款上限、预算警告、分类统计、CSV/Excel 导出，以及不依赖 Ollama 的解析规则。语音测试使用假的转写器，不会下载模型。
