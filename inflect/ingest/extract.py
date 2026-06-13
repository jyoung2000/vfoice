"""ffmpeg-backed audio extraction and encoding.

Any container (MP4/MKV/MP3/WAV/...) -> 24 kHz mono wav for cloning, plus a
44.1 kHz copy for auditioning. Also handles WAV/MP3 export on the way out.
ffmpeg is assumed to be on PATH; callers should check :func:`check_ffmpeg`
first and surface a friendly message if missing.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..logging_setup import get_logger

log = get_logger("extract")

CLONE_SAMPLE_RATE = 24000
AUDITION_SAMPLE_RATE = 44100


class FFmpegNotFoundError(RuntimeError):
    """ffmpeg could not be located on PATH / at the configured path."""


class ExtractionError(RuntimeError):
    """ffmpeg ran but failed to produce output."""


def check_ffmpeg(ffmpeg_path: str = "ffmpeg") -> str:
    """Return the resolved ffmpeg path or raise :class:`FFmpegNotFoundError`."""
    resolved = shutil.which(ffmpeg_path)
    if not resolved:
        raise FFmpegNotFoundError(
            "ffmpeg was not found. Install it and ensure it is on your PATH, "
            "or set the ffmpeg path in Settings."
        )
    return resolved


def _run(ffmpeg: str, args: list[str]) -> None:
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args]
    log.debug("ffmpeg: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ExtractionError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()}")


def extract_audio(
    src: str | Path,
    dst_wav: str | Path,
    sample_rate: int = CLONE_SAMPLE_RATE,
    mono: bool = True,
    ffmpeg_path: str = "ffmpeg",
) -> Path:
    """Decode ``src`` to a PCM wav at ``sample_rate`` (mono by default)."""
    ffmpeg = check_ffmpeg(ffmpeg_path)
    dst_wav = Path(dst_wav)
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    args = ["-i", str(src), "-vn", "-ac", "1" if mono else "2",
            "-ar", str(sample_rate), "-f", "wav", str(dst_wav)]
    _run(ffmpeg, args)
    if not dst_wav.exists() or dst_wav.stat().st_size == 0:
        raise ExtractionError(f"No audio produced from {src}")
    return dst_wav


def extract_for_ingest(
    src: str | Path,
    work_dir: str | Path,
    ffmpeg_path: str = "ffmpeg",
) -> tuple[Path, Path]:
    """Produce both the 24 kHz cloning wav and a 44.1 kHz audition wav.

    Returns ``(clone_wav_path, audition_wav_path)``.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    clone = extract_audio(src, work_dir / "clone_24k.wav", CLONE_SAMPLE_RATE, True, ffmpeg_path)
    audition = extract_audio(src, work_dir / "audition_44k.wav",
                             AUDITION_SAMPLE_RATE, True, ffmpeg_path)
    return clone, audition


def encode_audio(
    src_wav: str | Path,
    dst: str | Path,
    bitrate: str = "192k",
    ffmpeg_path: str = "ffmpeg",
) -> Path:
    """Encode a wav to the format implied by ``dst`` extension (.mp3/.wav/...)."""
    ffmpeg = check_ffmpeg(ffmpeg_path)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    args = ["-i", str(src_wav)]
    if dst.suffix.lower() == ".mp3":
        args += ["-codec:a", "libmp3lame", "-b:a", bitrate]
    _run(ffmpeg, [*args, str(dst)])
    if not dst.exists():
        raise ExtractionError(f"Failed to encode {dst}")
    return dst
