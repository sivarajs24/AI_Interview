"""Generate the final interview report card from multimodal analysis outputs."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Dict, List, Tuple


class ReportGenerator:
    """Compute composite interview metrics and actionable coaching feedback."""

    STRESS_EMOTIONS = {"fear", "sad", "disgust"}
    POSITIVE_EMOTIONS = {"happy", "neutral", "surprise"}
    FILLER_WORDS = {
        "um",
        "uh",
        "like",
        "basically",
        "actually",
        "literally",
        "you",
        "know",
        "sort",
        "kind",
    }
    MAX_QUESTION_DURATION_SECONDS = 180.0
    MAX_TOTAL_TIMELINE_SECONDS = 1800.0
    MAX_TIMELINE_POINTS = 60

    def _clamp(self, value: float, low: float = 0.0, high: float = 100.0) -> float:
        """Clamp a numeric value into the configured score range."""
        return max(low, min(high, value))

    def _compute_fluency_score(self, transcript: str, duration_seconds: float) -> float:
        """Estimate fluency based on pacing, filler words, and answer length."""
        cleaned = transcript.strip()
        if not cleaned or cleaned.lower() == "transcription unavailable":
            return 45.0

        words = [word.strip(".,!?;:\"'()[]{}").lower() for word in cleaned.split() if word.strip()]
        word_count = len(words)
        if word_count == 0:
            return 45.0

        duration = max(duration_seconds, 1.0)
        words_per_minute = word_count / (duration / 60.0)

        pacing_penalty = min(abs(words_per_minute - 135.0) * 0.30, 30.0)
        filler_count = sum(1 for word in words if word in self.FILLER_WORDS)
        filler_ratio = filler_count / max(word_count, 1)
        filler_penalty = min(filler_ratio * 180.0, 30.0)

        short_answer_penalty = 0.0
        if duration < 20:
            short_answer_penalty = min((20.0 - duration) * 1.2, 20.0)

        score = 100.0 - pacing_penalty - filler_penalty - short_answer_penalty
        return self._clamp(score)

    def _sanitize_duration(self, duration: Any, fallback: float = 1.0) -> float:
        """Normalize a duration into a finite, bounded number of seconds."""
        try:
            candidate = float(duration)
        except (TypeError, ValueError):
            candidate = fallback

        if not math.isfinite(candidate) or candidate <= 0.0:
            candidate = fallback

        if not math.isfinite(candidate) or candidate <= 0.0:
            candidate = 1.0

        return max(1.0, min(self.MAX_QUESTION_DURATION_SECONDS, candidate))

    def _aggregate_emotions(self, response_items: List[Dict[str, Any]]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Combine per-question emotion distributions into interview-level percentages."""
        weighted_counts: Counter[str] = Counter()
        total_weight = 0.0
        timeline: List[Dict[str, Any]] = []

        for item in response_items:
            question_index = int(item.get("question_index", 0))
            response = item.get("response", {})
            emotion_data = response.get("emotions", {})
            distribution = emotion_data.get("emotion_distribution", {}) or {}
            analyzed_frames = float(emotion_data.get("analyzed_frames", 0.0) or 0.0)
            weight = analyzed_frames if analyzed_frames > 0 else (1.0 if distribution else 0.0)

            for emotion, percentage in distribution.items():
                weighted_counts[str(emotion).lower()] += (float(percentage) / 100.0) * weight

            total_weight += weight

            for point in emotion_data.get("timeline", []) or []:
                timeline.append(
                    {
                        "question_index": question_index,
                        "timestamp": float(point.get("timestamp", 0.0)),
                        "emotion": str(point.get("dominant_emotion", "unknown")).lower(),
                        "confidence": float(point.get("confidence", 0.0)),
                    }
                )

        if total_weight <= 0:
            return {}, timeline

        distribution = {
            emotion: (count / total_weight) * 100.0 for emotion, count in weighted_counts.items()
        }
        return distribution, timeline

    def _build_question_breakdown(self, response_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Produce a report row for every question in the interview sequence."""
        breakdown: List[Dict[str, Any]] = []

        for item in response_items:
            question_index = int(item.get("question_index", 0))
            question_text = str(item.get("question", ""))
            category = str(item.get("category", "General"))
            response = item.get("response", {})

            transcript = str(response.get("transcript", "Transcription unavailable"))
            duration = float(response.get("duration_seconds", 0.0) or 0.0)

            sentiment = response.get("sentiment", {}) or {}
            sentiment_label = str(sentiment.get("label", "neutral"))
            sentiment_confidence = float(sentiment.get("confidence", 0.0) or 0.0)
            sentiment_score = float(sentiment.get("sentiment_score", 50.0) or 50.0)

            emotions = response.get("emotions", {}) or {}
            emotion_distribution = emotions.get("emotion_distribution", {}) or {}
            dominant_emotion = "unknown"
            if emotion_distribution:
                dominant_emotion = max(emotion_distribution, key=emotion_distribution.get)

            stress_level = sum(float(emotion_distribution.get(e, 0.0)) for e in self.STRESS_EMOTIONS)
            fluency_score = self._compute_fluency_score(transcript, duration)

            breakdown.append(
                {
                    "question_index": question_index,
                    "category": category,
                    "question": question_text,
                    "duration_seconds": duration,
                    "transcript": transcript,
                    "sentiment": {
                        "label": sentiment_label,
                        "confidence": sentiment_confidence,
                        "score": sentiment_score,
                        "sentence_scores": sentiment.get("sentence_scores", []),
                        "unsure_moments": sentiment.get("unsure_moments", []),
                    },
                    "emotions": {
                        "dominant_emotion": dominant_emotion,
                        "distribution": emotion_distribution,
                        "stress_level": stress_level,
                    },
                    "fluency_score": fluency_score,
                }
            )

        return sorted(breakdown, key=lambda row: row["question_index"])

    def _generate_suggestions(
        self,
        question_breakdown: List[Dict[str, Any]],
        stress_score: float,
        sentiment_score: float,
        fluency_score: float,
    ) -> List[str]:
        """Generate targeted coaching recommendations from observed interview signals."""
        suggestions: List[str] = []

        for row in question_breakdown:
            q_num = int(row["question_index"]) + 1
            stress_level = float(row["emotions"]["stress_level"])
            dominant = str(row["emotions"]["dominant_emotion"])
            sentiment_label = str(row["sentiment"]["label"])
            local_fluency = float(row["fluency_score"])
            unsure_moments = row["sentiment"].get("unsure_moments", [])

            if stress_level >= 35.0:
                suggestions.append(
                    f"At question {q_num}, your facial expression showed signs of {dominant}. "
                    "Try slowing your breathing and holding a neutral-to-positive expression before speaking."
                )
            if sentiment_label == "negative":
                suggestions.append(
                    f"Your answer to question {q_num} carried a negative tone. "
                    "Reframe challenges around actions taken, outcomes delivered, and lessons learned."
                )
            if local_fluency < 60.0:
                suggestions.append(
                    f"Question {q_num} had lower fluency. Practice this answer using the STAR method "
                    "and trim filler words to sound more confident."
                )
            if unsure_moments:
                suggestions.append(
                    f"Question {q_num} contained hesitation language in key sentences. "
                    "Rehearse those exact lines and replace uncertain wording with decisive verbs."
                )
            if len(suggestions) >= 8:
                break

        if stress_score > 45.0:
            suggestions.append(
                "Your overall stress score was elevated. Add a 60-second pre-answer routine: posture reset, "
                "single deep breath, and one-sentence structure in your head before speaking."
            )
        if sentiment_score < 55.0:
            suggestions.append(
                "Your sentiment trend suggests cautious framing. Lead with impact statements and quantifiable results "
                "to project stronger confidence."
            )
        if fluency_score < 65.0:
            suggestions.append(
                "Your fluency score can improve with timed drills. Record 90-second responses and review for "
                "pace, pauses, and repeated filler patterns."
            )

        fallback_suggestions = [
            "Maintain eye contact with the camera lens for 3-5 seconds at a time to strengthen executive presence.",
            "Use a concise opening sentence for each answer, then support it with one concrete example and measurable outcome.",
            "End each response with a short reflection on business impact to sound strategic rather than purely task-focused.",
            "Practice technical answers aloud with a whiteboard-style structure: context, decision, trade-off, and result.",
            "Prepare two backup examples for leadership and conflict questions so you can adapt quickly under pressure.",
        ]

        for suggestion in fallback_suggestions:
            if len(suggestions) >= 5:
                break
            if suggestion not in suggestions:
                suggestions.append(suggestion)

        return suggestions[:8]

    def _build_emotion_timeline(self, question_breakdown: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a finite per-question emotion timeline for chart rendering."""
        emotion_scale = {
            "happy": 5,
            "neutral": 4,
            "surprise": 3,
            "fear": 2,
            "sad": 1,
            "disgust": 0,
            "angry": 0,
            "unknown": 2,
        }

        labels: List[str] = []
        values: List[int] = []
        emotions: List[str] = []
        seconds: List[float] = []
        timeline_offset = 0.0

        for row in sorted(question_breakdown, key=lambda item: int(item.get("question_index", 0))):
            if len(labels) >= self.MAX_TIMELINE_POINTS:
                break

            q_index = int(row.get("question_index", 0))
            emotion = str(
                row.get("emotions", {}).get("dominant_emotion", "unknown")
            ).lower()
            duration = self._sanitize_duration(row.get("duration_seconds", 0.0), fallback=1.0)

            remaining = self.MAX_TOTAL_TIMELINE_SECONDS - timeline_offset
            if remaining <= 0.0:
                break

            step = min(duration, remaining)
            timeline_offset += step

            safe_second = max(0.0, min(self.MAX_TOTAL_TIMELINE_SECONDS, timeline_offset))
            minutes = int(safe_second // 60)
            secs = int(safe_second % 60)

            labels.append(f"Q{q_index + 1} · {minutes:02d}:{secs:02d}")
            values.append(emotion_scale.get(emotion, 2))
            emotions.append(emotion)
            seconds.append(round(safe_second, 2))

        return {
            "labels": labels,
            "values": values,
            "emotions": emotions,
            "seconds": seconds,
            "total_duration_seconds": round(
                min(self.MAX_TOTAL_TIMELINE_SECONDS, timeline_offset),
                2,
            ),
        }

    def generate_report(self, session_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Compute full report card scores, charts, and recommendations."""
        responses = session_payload.get("responses", []) or []
        breakdown = self._build_question_breakdown(responses)

        sentiment_scores = [float(row["sentiment"]["score"]) for row in breakdown]
        fluency_scores = [float(row["fluency_score"]) for row in breakdown]

        sentiment_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 50.0
        fluency_score = sum(fluency_scores) / len(fluency_scores) if fluency_scores else 50.0

        emotion_distribution, _ = self._aggregate_emotions(responses)
        positive_emotion_pct = sum(float(emotion_distribution.get(e, 0.0)) for e in self.POSITIVE_EMOTIONS)
        stress_score = self._clamp(
            sum(float(emotion_distribution.get(e, 0.0)) for e in self.STRESS_EMOTIONS)
        )

        confidence_score = self._clamp(
            (0.4 * sentiment_score) + (0.4 * positive_emotion_pct) + (0.2 * fluency_score)
        )

        clarity_score = self._clamp((sentiment_score * 0.35) + (fluency_score * 0.65))
        body_language_score = self._clamp(positive_emotion_pct - (stress_score * 0.30) + 25.0)
        eye_contact_score = self._clamp(70.0 - (stress_score * 0.25) + (positive_emotion_pct * 0.15))

        suggestions = self._generate_suggestions(breakdown, stress_score, sentiment_score, fluency_score)

        return {
            "meta": {
                "session_id": session_payload.get("session_id", ""),
                "candidate_name": session_payload.get("candidate_name", "Candidate"),
                "target_role": session_payload.get("target_role", "Software Engineer"),
                "created_at": session_payload.get("created_at", ""),
                "total_questions": len(session_payload.get("questions", [])),
            },
            "scores": {
                "confidence_score": round(confidence_score, 2),
                "stress_score": round(stress_score, 2),
                "sentiment_score": round(sentiment_score, 2),
                "fluency_score": round(fluency_score, 2),
                "positive_emotion_percentage": round(positive_emotion_pct, 2),
                "clarity_score": round(clarity_score, 2),
                "body_language_score": round(body_language_score, 2),
                "eye_contact_score": round(eye_contact_score, 2),
            },
            "emotion_distribution": emotion_distribution,
            "question_breakdown": breakdown,
            "suggestions": suggestions,
            "charts": {
                "radar": {
                    "labels": [
                        "Confidence",
                        "Clarity",
                        "Body Language",
                        "Sentiment",
                        "Fluency",
                        "Eye Contact",
                    ],
                    "values": [
                        round(confidence_score, 2),
                        round(clarity_score, 2),
                        round(body_language_score, 2),
                        round(sentiment_score, 2),
                        round(fluency_score, 2),
                        round(eye_contact_score, 2),
                    ],
                },
                "emotion_timeline": self._build_emotion_timeline(breakdown),
            },
        }
