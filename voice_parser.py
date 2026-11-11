"""Local speech to text.

Audio is not uploaded to a cloud recognition service. faster-whisper
transcribes it on this computer, then the text uses the same parse and
confirm flow as keyboard input. macOS uses CPU and int8 by default,
without NVIDIA CUDA.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Callable

from ai_parser import TransactionDraft, parse_transaction_text
from config import Settings


class VoiceError(RuntimeError):
    """The audio file or the local transcription failed."""


AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac"}


def validate_audio_name(filename: str) -> str:
    """Check the audio extension and return the lowercase suffix."""
    suffix = Path(filename).suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        allowed = ", ".join(sorted(AUDIO_SUFFIXES))
        raise VoiceError(f"That audio format is not supported. Use one of: {allowed}.")
    return suffix


def default_model_loader(model_size: str, device: str, compute_type: str) -> Any:
    """Load faster-whisper. Explain how to install it when the package is missing."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise VoiceError(
            "faster-whisper is not installed. In the virtual environment, run pip install -r requirements.txt."
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
    """Turn audio into text.

    If device is cuda or auto and the GPU is unavailable, this falls back to CPU.
    The transcript is not written to the log.
    """
    if not audio_path.is_file():
        raise VoiceError("The audio file was not found.")
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
        detail = "; ".join(failures) if failures else "unknown reason"
        raise VoiceError(f"The local speech model failed to load: {detail}")
    try:
        segments, _info = model.transcribe(str(audio_path), language=whisper_language)
        text = "".join(segment.text for segment in segments).strip()
    except Exception as exc:
        raise VoiceError("Transcription failed. The audio stayed on this computer and was not uploaded.") from exc
    if not text:
        raise VoiceError("No speech was recognized. Record again, or type the entry instead.")
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
    """Transcribe, then parse. Confirmation and saving still belong to the finance service."""
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
