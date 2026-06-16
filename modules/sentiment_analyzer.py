"""RoBERTa sentiment analysis module for interview transcripts."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from transformers import pipeline


class SentimentAnalyzer:
    """Run sentence-level and overall sentiment scoring using HuggingFace Transformers."""

    def __init__(self, model_name: str) -> None:
        """Initialize analyzer with lazy loading for the sentiment pipeline."""
        self.model_name = model_name
        self._pipeline: Optional[Any] = None
        self._load_error: Optional[str] = None

    def _load_pipeline(self) -> bool:
        """Load the transformer pipeline once and cache it."""
        if self._pipeline is not None:
            return True
        if self._load_error is not None:
            return False

        try:
            self._pipeline = pipeline("sentiment-analysis", model=self.model_name, truncation=True)
            return True
        except Exception as exc:
            self._load_error = str(exc)
            return False

    def _normalize_label(self, raw_label: str) -> str:
        """Normalize model label outputs to positive, neutral, or negative."""
        label = raw_label.lower().strip()
        if "pos" in label:
            return "positive"
        if "neg" in label:
            return "negative"
        if "neu" in label:
            return "neutral"

        label_map = {
            "label_0": "negative",
            "label_1": "neutral",
            "label_2": "positive",
        }
        return label_map.get(label, "neutral")

    def _split_sentences(self, text: str) -> List[str]:
        """Split transcript text into clean sentence-like chunks."""
        chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk.strip()]
        return chunks if chunks else [text.strip()] if text.strip() else []

    def _sentiment_score(self, label: str, confidence: float) -> float:
        """Convert model label and confidence into a 0-100 sentiment score."""
        if label == "positive":
            return confidence
        if label == "neutral":
            return 50.0 + ((confidence - 50.0) * 0.2)
        return max(0.0, 100.0 - confidence)

    def empty_result(self) -> Dict[str, Any]:
        """Return a consistent empty sentiment payload when transcription is unavailable."""
        return {
            "label": "neutral",
            "confidence": 0.0,
            "sentiment_score": 50.0,
            "sentence_scores": [],
            "unsure_moments": [],
            "error": "No transcript available for sentiment analysis.",
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze transcript text and return overall and per-sentence sentiment insights."""
        cleaned = text.strip()
        if not cleaned:
            return self.empty_result()

        if not self._load_pipeline():
            payload = self.empty_result()
            payload["error"] = self._load_error or "Sentiment pipeline unavailable."
            return payload

        try:
            overall_raw = self._pipeline(cleaned)[0]
            overall_label = self._normalize_label(str(overall_raw.get("label", "neutral")))
            overall_confidence = float(overall_raw.get("score", 0.0)) * 100.0

            sentence_scores: List[Dict[str, Any]] = []
            for sentence in self._split_sentences(cleaned):
                raw = self._pipeline(sentence)[0]
                sentence_label = self._normalize_label(str(raw.get("label", "neutral")))
                sentence_confidence = float(raw.get("score", 0.0)) * 100.0
                sentence_scores.append(
                    {
                        "sentence": sentence,
                        "label": sentence_label,
                        "confidence": sentence_confidence,
                        "sentiment_score": self._sentiment_score(sentence_label, sentence_confidence),
                    }
                )

            unsure_moments = sorted(
                [item for item in sentence_scores if item["label"] == "negative"],
                key=lambda item: item["confidence"],
                reverse=True,
            )[:5]

            return {
                "label": overall_label,
                "confidence": overall_confidence,
                "sentiment_score": self._sentiment_score(overall_label, overall_confidence),
                "sentence_scores": sentence_scores,
                "unsure_moments": unsure_moments,
                "error": None,
            }
        except Exception as exc:
            payload = self.empty_result()
            payload["error"] = str(exc)
            return payload
