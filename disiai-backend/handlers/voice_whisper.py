"""Voice handler: Faster-Whisper STT and Edge-TTS Indonesian speech output.

Heavy libraries are imported lazily so /health and the text path still work
when the full voice stack is not installed.
"""

import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Any

from config import settings

_model: Any = None


def _output_dir() -> Path:
    directory = settings.static_dir / "mock" / "out"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _get_model() -> Any:
    global _model
    if _model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Faster-Whisper belum terpasang. Jalankan pip install -r requirements.txt "
                "dengan Python 3.11/3.12."
            ) from exc
        _model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
    return _model


def transcribe(audio_bytes: bytes, *, filename: str = "audio.wav") -> dict[str, Any]:
    """Convert microphone audio bytes into an Indonesian transcript."""
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(audio_bytes)
        temp_path = Path(handle.name)

    try:
        segments, info = _get_model().transcribe(str(temp_path), beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return {
            "text": text,
            "language": info.language,
            "duration": round(float(info.duration), 2),
        }
    finally:
        temp_path.unlink(missing_ok=True)


def _synthesize(text: str, voice: str, path: Path) -> None:
    async def run() -> None:
        import edge_tts

        communicate = edge_tts.Communicate(text, voice, rate=settings.tts_rate)
        await communicate.save(str(path))

    asyncio.run(run())


def synthesize_sync(text: str, *, voice: str | None = None) -> dict[str, str]:
    """Generate an MP3 reply and return a same-origin media URL."""
    try:
        import edge_tts  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Edge-TTS belum terpasang.") from exc

    filename = f"{uuid.uuid4().hex}.mp3"
    target = _output_dir() / filename
    _synthesize(text, voice or settings.tts_voice, target)
    return {
        "audio_url": f"/media/mock/out/{filename}",
        "voice": voice or settings.tts_voice,
    }
