from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("Whisper sử dụng CUDA GPU")
            return "cuda"
        # MPS (Apple Silicon) gây hallucination/lặp với Whisper, không dùng
    except Exception:
        pass
        
    from app import config
    if config.REQUIRE_GPU:
        raise RuntimeError(
            "REQUIRE_GPU=1 but no CUDA device found for Whisper! "
            "Whisper on CPU is too slow. Please check drivers or set REQUIRE_GPU=0."
        )
    logger.info("Whisper sử dụng CPU")
    return "cpu"


def _extract_audio(video_path: Path) -> Path:
    """Extract audio from video to a temp mp3 file using ffmpeg."""
    import subprocess
    import tempfile

    audio_path = Path(tempfile.mktemp(suffix=".mp3"))
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vn", "-acodec", "libmp3lame", "-ab", "64k", "-ar", "16000", "-ac", "1",
        "-y", str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")
    logger.info("Extracted audio: %s (%.1f MB)", audio_path.name, audio_path.stat().st_size / 1024 / 1024)
    return audio_path


def _clip_media(src: Path, duration_sec: int) -> Path:
    """Cắt N giây đầu của video/audio bằng ffmpeg stream-copy (không re-encode).

    Nhanh ~1s với mọi kích thước file. Trả về temp file cùng đuôi với src.
    """
    import subprocess
    import tempfile

    suffix = src.suffix or ".mp4"
    clip_path = Path(tempfile.mktemp(suffix=suffix))
    cmd = [
        "ffmpeg", "-ss", "0", "-i", str(src),
        "-t", str(int(duration_sec)),
        "-c", "copy", "-y", str(clip_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        clip_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg clip failed: {result.stderr[:500]}")
    logger.info(
        "Clipped first %ds of %s → %s (%.1f MB)",
        duration_sec, src.name, clip_path.name,
        clip_path.stat().st_size / 1024 / 1024,
    )
    return clip_path


_GROQ_MAX_SIZE = 24 * 1024 * 1024  # 24MB safe limit


class GroqTranscriber:
    """Transcribe using Groq Whisper API (whisper-large-v3-turbo)."""

    def __init__(self, api_key: str, language: str = "vi"):
        self.api_key = api_key
        self.language = language

    def transcribe(self, path: str | Path, clip_duration_sec: int | None = None) -> dict:
        from groq import Groq

        client = Groq(api_key=self.api_key)
        path = Path(path)

        logger.info("Groq Whisper: transcribing %s", path.name)

        # Preview mode: cắt N giây đầu trước khi transcribe — tiết kiệm cost + thời gian
        # khi chỉ cần đoạn đầu để gen metadata. Clip xoá ở finally.
        clip_path: Path | None = None
        if clip_duration_sec and clip_duration_sec > 0:
            try:
                clip_path = _clip_media(path, clip_duration_sec)
                path = clip_path
            except Exception as exc:
                logger.warning("Clip failed (%s) — fallback to full transcribe", exc)

        # Extract audio if file is too large or is a video format
        send_path = path
        extracted = False
        if path.stat().st_size > _GROQ_MAX_SIZE or path.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"):
            logger.info("File quá lớn hoặc là video, đang trích xuất audio...")
            send_path = _extract_audio(path)
            extracted = True

        try:
            with open(send_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    file=(send_path.name, f),
                    model="whisper-large-v3-turbo",
                    language=self.language,
                    response_format="verbose_json",
                    prompt=(
                        "Đây là video tiếng Việt có dấu đầy đủ. "
                        "Vui lòng ghi chính xác dấu thanh và dấu mũ tiếng Việt."
                    ),
                )
        finally:
            if extracted:
                send_path.unlink(missing_ok=True)
            if clip_path is not None:
                clip_path.unlink(missing_ok=True)

        segments = []
        for seg in result.segments or []:
            segments.append({
                "start": round(float(seg["start"]), 3),
                "end": round(float(seg["end"]), 3),
                "text": seg["text"].strip(),
            })

        duration = float(result.duration or 0.0)
        if not duration and segments:
            duration = segments[-1]["end"]

        logger.info("Groq Whisper: %d segments, %.1fs duration", len(segments), duration)

        return {
            "segments": segments,
            "language": result.language or self.language,
            "duration": duration,
        }


class WhisperTranscriber:
    """Transcribe using local openai-whisper model."""

    def __init__(self, model_name: str = "medium", language: str = "vi"):
        self.model_name = model_name
        self.language = language
        self._model = None
        self._device = _detect_device()

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            import whisper  # type: ignore
        except ImportError:
            raise ImportError(
                "openai-whisper is not installed. Run: pip install openai-whisper"
            )
        self._model = whisper.load_model(self.model_name, device=self._device)
        logger.info("Loaded Whisper model '%s' on %s", self.model_name, self._device)
        return self._model

    def transcribe(self, path: str | Path, clip_duration_sec: int | None = None) -> dict:
        model = self._load_model()
        path = Path(path)

        clip_path: Path | None = None
        if clip_duration_sec and clip_duration_sec > 0:
            try:
                clip_path = _clip_media(path, clip_duration_sec)
                path = clip_path
            except Exception as exc:
                logger.warning("Clip failed (%s) — fallback to full transcribe", exc)

        # MPS với fp16 gây hallucination, chỉ dùng fp16 cho CUDA
        fp16 = self._device == "cuda"
        try:
            result = model.transcribe(
                str(path),
                verbose=False,
                language=self.language,
                fp16=fp16,
                initial_prompt=(
                    "Đây là video tiếng Việt có dấu đầy đủ. "
                    "Vui lòng ghi chính xác dấu thanh và dấu mũ tiếng Việt."
                ),
                condition_on_previous_text=True,
                temperature=0.0,
                beam_size=5,
                best_of=5,
            )
        finally:
            if clip_path is not None:
                clip_path.unlink(missing_ok=True)
        segments = [
            {
                "start": round(float(s["start"]), 3),
                "end": round(float(s["end"]), 3),
                "text": s["text"].strip(),
            }
            for s in result["segments"]
        ]
        duration = float(result.get("duration", 0.0))
        if not duration and segments:
            duration = segments[-1]["end"]
        return {
            "segments": segments,
            "language": result.get("language", self.language),
            "duration": duration,
        }


def get_transcriber(language: str = "vi"):
    """Return best available transcriber: Groq API if key exists, else local Whisper."""
    from app.core import config
    if config.GROQ_API_KEY:
        logger.info("Sử dụng Groq Whisper API")
        return GroqTranscriber(api_key=config.GROQ_API_KEY, language=language)
    logger.info("Sử dụng local Whisper (medium)")
    return WhisperTranscriber(language=language)
