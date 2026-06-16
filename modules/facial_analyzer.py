"""Facial emotion analysis module powered by DeepFace and OpenCV."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from deepface import DeepFace


class FacialAnalyzer:
    """Analyze interview video and webcam frames for emotional signals."""

    STRESS_EMOTIONS = {"fear", "sad", "disgust"}

    def __init__(self, frame_step: int = 30) -> None:
        """Initialize analyzer with a configurable frame sampling interval."""
        self.frame_step = max(1, frame_step)

    def _run_deepface(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Run DeepFace on an image and normalize the result structure."""
        try:
            result = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv",
                silent=True,
            )
            if isinstance(result, list):
                result = result[0]

            emotions_raw = result.get("emotion", {}) or {}
            emotion_scores = {str(k).lower(): float(v) for k, v in emotions_raw.items()}
            dominant = str(result.get("dominant_emotion", "unknown")).lower()
            confidence = float(emotion_scores.get(dominant, 0.0))

            return {
                "dominant_emotion": dominant,
                "confidence": confidence,
                "emotion_scores": emotion_scores,
            }
        except Exception:
            return None

    def analyze_realtime_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze one webcam frame for real-time feedback in the interview UI."""
        result = self._run_deepface(frame)
        if result is None:
            return {
                "dominant_emotion": "unknown",
                "confidence": 0.0,
                "emotion_scores": {},
            }
        return result

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Analyze sampled video frames and return emotion timeline and aggregates."""
        if not Path(video_path).exists():
            return {
                "timeline": [],
                "emotion_distribution": {},
                "stress_moments": [],
                "error": f"Video file not found: {video_path}",
            }

        capture = cv2.VideoCapture(video_path)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        if not np.isfinite(fps) or fps < 5.0 or fps > 120.0:
            fps = 30.0

        frame_idx = 0
        timeline: List[Dict[str, Any]] = []
        stress_moments: List[Dict[str, Any]] = []
        counts: Counter[str] = Counter()

        while capture.isOpened():
            has_frame, frame = capture.read()
            if not has_frame:
                break

            if frame_idx % self.frame_step == 0:
                result = self._run_deepface(frame)
                if result is not None:
                    timestamp = float(frame_idx / fps)
                    dominant_emotion = str(result["dominant_emotion"])
                    confidence = float(result["confidence"])

                    timeline.append(
                        {
                            "timestamp": timestamp,
                            "dominant_emotion": dominant_emotion,
                            "confidence": confidence,
                            "emotion_scores": result["emotion_scores"],
                        }
                    )
                    counts[dominant_emotion] += 1

                    if dominant_emotion in self.STRESS_EMOTIONS:
                        stress_moments.append(
                            {
                                "timestamp": timestamp,
                                "emotion": dominant_emotion,
                                "confidence": confidence,
                            }
                        )

            frame_idx += 1

        capture.release()

        analyzed_frames = sum(counts.values())
        if analyzed_frames == 0:
            return {
                "timeline": [],
                "emotion_distribution": {},
                "stress_moments": [],
                "error": "No analyzable faces detected in the video.",
            }

        distribution = {
            emotion: (count / analyzed_frames) * 100.0 for emotion, count in counts.items()
        }

        return {
            "timeline": timeline,
            "emotion_distribution": distribution,
            "stress_moments": stress_moments,
            "analyzed_frames": analyzed_frames,
            "error": None,
        }

    def get_overall_dominant_emotion(self, distribution: Dict[str, float]) -> str:
        """Return the highest-percentage emotion from a distribution map."""
        if not distribution:
            return "unknown"
        return max(distribution, key=distribution.get)
