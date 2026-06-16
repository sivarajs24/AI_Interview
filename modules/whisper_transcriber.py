"""Whisper-based speech-to-text transcription module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import whisper


class WhisperTranscriber:
    """Provide local Whisper transcription with graceful error handling."""

    def __init__(self, model_name: str = "base") -> None:
        """Initialize a transcriber with lazy model loading."""
        self.model_name = model_name
        self._model: Optional[Any] = None
        self._load_error: Optional[str] = None

    def _load_model(self) -> bool:
        """Load the Whisper model once and cache it for subsequent calls."""
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False

        try:
            self._model = whisper.load_model(self.model_name)
            return True
        except Exception as exc:
            self._load_error = str(exc)
            return False

    def _collect_word_timestamps(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract flattened word-level timestamps from Whisper segments."""
        words: List[Dict[str, Any]] = []
        for segment in segments:
            for word_info in segment.get("words", []) or []:
                words.append(
                    {
                        "word": str(word_info.get("word", "")).strip(),
                        "start": float(word_info.get("start", 0.0)),
                        "end": float(word_info.get("end", 0.0)),
                        "probability": float(word_info.get("probability", 0.0)),
                    }
                )
        return words

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe an audio file and return text plus timestamps."""
        path = Path(audio_path)
        if not path.exists():
            return {
                "text": "Transcription unavailable",
                "segments": [],
                "word_timestamps": [],
                "error": f"Audio file not found: {audio_path}",
            }

        if not self._load_model():
            return {
                "text": "Transcription unavailable",
                "segments": [],
                "word_timestamps": [],
                "error": self._load_error or "Whisper model failed to load.",
            }

        try:
            result = self._model.transcribe(audio_path, word_timestamps=True, fp16=False)
            segments = result.get("segments", []) or []
            normalized_segments: List[Dict[str, Any]] = []
            for segment in segments:
                normalized_segments.append(
                    {
                        "id": int(segment.get("id", 0)),
                        "start": float(segment.get("start", 0.0)),
                        "end": float(segment.get("end", 0.0)),
                        "text": str(segment.get("text", "")).strip(),
                    }
                )

            return {
                "text": str(result.get("text", "")).strip(),
                "language": str(result.get("language", "unknown")),
                "segments": normalized_segments,
                "word_timestamps": self._collect_word_timestamps(segments),
                "error": None,
            }
        except Exception as exc:
            return {
                "text": "Transcription unavailable",
                "segments": [],
                "word_timestamps": [],
                "error": str(exc),
            }
