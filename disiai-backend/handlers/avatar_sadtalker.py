"""Avatar handler: SadTalker lip-sync with a local mock fallback."""

import shlex
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from config import settings


def _output_dir() -> Path:
    directory = settings.static_dir / "mock" / "out"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_media_path(url_or_path: str) -> Path:
    """Map a same-origin /media URL (or plain path) onto the local filesystem."""
    path = url_or_path.split("?", 1)[0]
    if path.startswith("/media/"):
        return settings.static_dir / path.removeprefix("/media/")
    candidate = Path(path)
    return candidate if candidate.is_absolute() else settings.resolve_path(path)


def _run_sadtalker(audio_path: Path, image_path: Path, result_dir: Path) -> Path | None:
    if not settings.sadtalker_bin:
        return None
    command = shlex.split(settings.sadtalker_bin)
    command.extend(
        [
            "--driven_audio",
            str(audio_path),
            "--source_image",
            str(image_path),
            "--result_dir",
            str(result_dir),
            "--still",
        ]
    )
    if settings.sadtalker_ckpt:
        command.extend(["--checkpoint", settings.sadtalker_ckpt])

    completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if completed.returncode != 0:
        raise RuntimeError(f"SadTalker gagal: {completed.stderr[-500:]}")

    videos = sorted(result_dir.rglob("*.mp4"))
    return videos[-1] if videos else None


def generate(
    text: str,
    *,
    audio_url: str | None = None,
    image_path: str | None = None,
) -> dict[str, Any]:
    """Return a lip-synced video URL, or a local mock fallback for development."""
    result_dir = _output_dir()
    image = settings.resolve_path(image_path or settings.avatar_image_path)

    if settings.sadtalker_enabled and audio_url:
        audio = resolve_media_path(audio_url)
        video = _run_sadtalker(audio, image, result_dir)
        if video:
            target = result_dir / f"{uuid.uuid4().hex}.mp4"
            shutil.copyfile(video, target)
            return {
                "video_url": f"/media/mock/out/{target.name}",
                "mock": False,
                "text": text,
            }

    mock_source = settings.resolve_path(settings.mock_video_path)
    if mock_source.exists():
        target = result_dir / f"{uuid.uuid4().hex}.mp4"
        shutil.copyfile(mock_source, target)
        return {
            "video_url": f"/media/mock/out/{target.name}",
            "mock": True,
            "text": text,
        }

    return {
        "video_url": None,
        "audio_url": audio_url,
        "mock": True,
        "text": text,
        "note": (
            "SadTalker tidak aktif dan video mock belum tersedia. "
            "Letakkan file MP4 di static/mock/avatar.mp4 atau aktifkan SADTALKER_ENABLED."
        ),
    }
