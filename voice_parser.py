"""本地语音转文字。

音频不会上传到云端识别服务。faster-whisper 在本机转写后，
文字进入与键盘输入相同的解析和确认流程。
macOS 上默认使用 CPU 和 int8，不依赖 NVIDIA CUDA。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Callable

from ai_parser import TransactionDraft, parse_transaction_text
from config import Settings


class VoiceError(RuntimeError):
    """语音文件或本地转写失败。"""


AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac"}


def validate_audio_name(filename: str) -> str:
    """检查音频扩展名，返回小写后缀。"""
    suffix = Path(filename).suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        allowed = "、".join(sorted(AUDIO_SUFFIXES))
        raise VoiceError(f"不支持该音频格式。可用格式：{allowed}。")
    return suffix


def default_model_loader(model_size: str, device: str, compute_type: str) -> Any:
    """加载 faster-whisper。未安装依赖时给出可操作的中文说明。"""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise VoiceError(
            "未安装 faster-whisper。请在虚拟环境中执行 pip install -r requirements.txt。"
        ) from exc
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_audio(
    audio_path: Path,
    *,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    model_loader: Callable[[str, str, str], Any] | None = None,
) -> str:
    """把音频转成文字。

    device 为 cuda 或 auto 时，如果 GPU 不可用，会自动改用 CPU。
    转写文本不会写入日志。
    """
    if not audio_path.is_file():
        raise VoiceError("找不到音频文件。")
    loader = model_loader or default_model_loader
    whisper_language = None if language in {None, "", "auto"} else language
    attempts = [(device, compute_type)]
    if device in {"cuda", "auto"}:
        attempts.append(("cpu", "int8"))
    model = None
    failures: list[str] = []
    for current_device, current_compute in attempts:
        try:
            model = loader(model_size, current_device, current_compute)
            break
        except VoiceError:
            raise
        except Exception as exc:
            failures.append(f"{current_device}: {exc}")
    if model is None:
        detail = "；".join(failures) if failures else "未知原因"
        raise VoiceError(f"本地语音模型加载失败：{detail}")
    try:
        segments, _info = model.transcribe(str(audio_path), language=whisper_language)
        text = "".join(segment.text for segment in segments).strip()
    except Exception as exc:
        raise VoiceError("语音转写失败。音频仍保留在本机，没有上传识别。") from exc
    if not text:
        raise VoiceError("没有识别到语音内容，请重新录音或改用文字输入。")
    return text


def transcribe_and_parse(
    audio_path: Path,
    *,
    today: date,
    default_currency: str,
    settings: Settings | None = None,
    model_loader: Callable[[str, str, str], Any] | None = None,
    requester: Callable[..., dict[str, Any]] | None = None,
    use_ollama: bool = True,
) -> tuple[str, TransactionDraft]:
    """转写后调用文本解析。确认和入库仍由交易服务负责。"""
    active = settings or Settings()
    transcript = transcribe_audio(
        audio_path,
        model_size=active.whisper_model,
        device=active.whisper_device,
        compute_type=active.whisper_compute_type,
        language=active.whisper_language,
        model_loader=model_loader,
    )
    draft = parse_transaction_text(
        transcript,
        today=today,
        default_currency=default_currency,
        settings=active,
        requester=requester,
        use_ollama=use_ollama,
    )
    draft.source = "voice"
    draft.raw_text = transcript
    draft.description = transcript[:500]
    return transcript, draft
