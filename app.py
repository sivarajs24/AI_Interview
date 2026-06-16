"""Main Flask application for InterviewIQ AI Interview Coach."""

from __future__ import annotations

import base64
import io
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import librosa
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from werkzeug.exceptions import HTTPException

from modules.facial_analyzer import FacialAnalyzer
from modules.interview_engine import InterviewEngine
from modules.qwen_interviewer import QwenInterviewLLM
from modules.report_generator import ReportGenerator
from modules.sentiment_analyzer import SentimentAnalyzer
from modules.whisper_transcriber import WhisperTranscriber

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-me-in-production")
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024

CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=os.getenv("SOCKETIO_ASYNC_MODE", "eventlet"))

transcriber = WhisperTranscriber(model_name="base")
sentiment_analyzer = SentimentAnalyzer(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest")
facial_analyzer = FacialAnalyzer(frame_step=30)
qwen_llm = QwenInterviewLLM()
interview_engine = InterviewEngine(llm_client=qwen_llm)
report_generator = ReportGenerator()


def _json_error(message: str, status_code: int = 400, code: str = "BAD_REQUEST") -> Tuple[Any, int]:
    """Create a consistent JSON error response body and HTTP status code."""
    return jsonify({"success": False, "error": {"code": code, "message": message}}), status_code


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert an arbitrary value to float and fall back when conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = -1) -> int:
    """Convert an arbitrary value to int and fall back when conversion fails."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_extension(filename: str, fallback: str) -> str:
    """Resolve a safe file extension from an uploaded filename."""
    candidate = Path(filename).suffix.strip().lower()
    if candidate and len(candidate) <= 8:
        return candidate
    return fallback


def _save_upload(file_obj: Any, prefix: str, fallback_ext: str) -> Path:
    """Persist an uploaded file under uploads with a unique session-aware name."""
    ext = _resolve_extension(file_obj.filename or "", fallback_ext)
    safe_name = f"{prefix}_{uuid.uuid4().hex}{ext}"
    out_path = UPLOAD_DIR / safe_name
    file_obj.save(out_path)
    return out_path


def _decode_frame(frame_data_uri: str) -> Optional[np.ndarray]:
    """Decode a base64 JPEG data URI into an OpenCV image matrix."""
    if not frame_data_uri:
        return None

    payload = frame_data_uri.split(",", 1)[1] if "," in frame_data_uri else frame_data_uri
    try:
        binary = base64.b64decode(payload)
    except (ValueError, TypeError):
        return None

    arr = np.frombuffer(binary, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def _build_pdf(report_payload: Dict[str, Any], session_id: str) -> io.BytesIO:
    """Build a downloadable PDF summary from the final interview report payload."""
    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter, title="InterviewIQ Report")
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("InterviewIQ - Interview Report", styles["Title"]))
    story.append(Paragraph(f"Session ID: {session_id}", styles["Normal"]))
    story.append(Paragraph(f"Generated: {datetime.now(timezone.utc).isoformat()}", styles["Normal"]))
    story.append(Spacer(1, 12))

    scores = report_payload.get("scores", {})
    story.append(Paragraph(f"Confidence Score: {scores.get('confidence_score', 0):.1f}", styles["Heading3"]))
    story.append(Paragraph(f"Sentiment Score: {scores.get('sentiment_score', 0):.1f}", styles["Heading3"]))
    story.append(Paragraph(f"Stress Score: {scores.get('stress_score', 0):.1f}", styles["Heading3"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Actionable Suggestions", styles["Heading2"]))
    for idx, suggestion in enumerate(report_payload.get("suggestions", []), start=1):
        story.append(Paragraph(f"{idx}. {suggestion}", styles["Normal"]))
        story.append(Spacer(1, 6))

    document.build(story)
    buffer.seek(0)
    return buffer


@app.get("/")
def home() -> str:
    """Render the InterviewIQ landing page."""
    return render_template("index.html", title="InterviewIQ | AI Interview Coach")


@app.get("/interview")
def interview_page() -> str:
    """Render the live interview workspace page."""
    return render_template("interview.html", title="InterviewIQ | Interview Session")


@app.get("/report/<session_id>")
def report_page(session_id: str) -> str:
    """Render the report dashboard page for a session."""
    return render_template("report.html", title="InterviewIQ | Report", session_id=session_id)


@app.get("/favicon.ico")
def favicon() -> Any:
    """Serve the application favicon to avoid noisy browser 404 requests."""
    favicon_path = BASE_DIR / "static" / "assets" / "logo.svg"
    return send_file(favicon_path, mimetype="image/svg+xml")


@app.post("/api/start-session")
def start_session() -> Tuple[Any, int]:
    """Initialize a new interview session and return its generated question flow."""
    data = request.get_json(silent=True) or {}
    candidate_name = str(data.get("candidate_name", "Candidate")).strip() or "Candidate"
    target_role = str(data.get("target_role", "Software Engineer")).strip() or "Software Engineer"

    session = interview_engine.create_session(candidate_name=candidate_name, target_role=target_role)
    questions = interview_engine.get_questions(session.session_id)
    questions = questions[:1]
    session.questions = questions

    return (
        jsonify(
            {
                "success": True,
                "session_id": session.session_id,
                "candidate_name": session.candidate_name,
                "target_role": session.target_role,
                "questions": questions,
                "total_questions": len(questions),
                "llm_model": interview_engine.active_llm_model(),
            }
        ),
        200,
    )


@app.post("/api/upload-response")
def upload_response() -> Tuple[Any, int]:
    """Accept media blobs for one answer, run multimodal analysis, and store the result."""
    session_id = str(request.form.get("session_id", "")).strip()
    question_index = _safe_int(request.form.get("question_index"), -1)
    duration_seconds = _safe_float(request.form.get("duration_seconds"), 0.0)

    if not session_id:
        return _json_error("session_id is required.", code="MISSING_SESSION")
    if not interview_engine.session_exists(session_id):
        return _json_error("Session not found.", status_code=404, code="SESSION_NOT_FOUND")
    if question_index < 0:
        return _json_error("question_index is required.", code="MISSING_QUESTION_INDEX")

    session_questions = interview_engine.get_questions(session_id)
    if question_index >= len(session_questions):
        return _json_error(
            "question_index is out of range for this session.",
            status_code=422,
            code="QUESTION_INDEX_OUT_OF_RANGE",
        )

    audio_file = request.files.get("audio")
    video_file = request.files.get("video")
    if not audio_file or not video_file:
        return _json_error("Both audio and video files are required.", code="MISSING_MEDIA")

    audio_path = _save_upload(audio_file, f"{session_id}_q{question_index}_audio", ".webm")
    video_path = _save_upload(video_file, f"{session_id}_q{question_index}_video", ".webm")

    if duration_seconds <= 0:
        try:
            duration_seconds = float(librosa.get_duration(path=str(audio_path)))
        except Exception:
            duration_seconds = 0.0

    if not np.isfinite(duration_seconds) or duration_seconds < 0:
        duration_seconds = 0.0
    duration_seconds = min(duration_seconds, 180.0)

    if duration_seconds < 3:
        return _json_error(
            "Audio is too short. Please answer for at least 3 seconds.",
            status_code=422,
            code="AUDIO_TOO_SHORT",
        )

    transcription = transcriber.transcribe(str(audio_path))
    transcript_text = str(transcription.get("text", "")).strip() or "Transcription unavailable"

    if transcript_text == "Transcription unavailable":
        sentiment = sentiment_analyzer.empty_result()
    else:
        sentiment = sentiment_analyzer.analyze(transcript_text)

    emotions = facial_analyzer.analyze_video(str(video_path))
    dominant_emotion = facial_analyzer.get_overall_dominant_emotion(emotions.get("emotion_distribution", {}))

    response_record: Dict[str, Any] = {
        "question_index": question_index,
        "duration_seconds": duration_seconds,
        "transcript": transcript_text,
        "transcription": transcription,
        "sentiment": sentiment,
        "emotions": emotions,
        "audio_path": str(audio_path),
        "video_path": str(video_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    interview_engine.record_response(session_id, question_index, response_record)

    return (
        jsonify(
            {
                "success": True,
                "question_index": question_index,
                "duration_seconds": duration_seconds,
                "transcript": transcript_text,
                "sentiment": sentiment,
                "emotions": emotions,
                "dominant_emotion": dominant_emotion,
            }
        ),
        200,
    )


@app.post("/api/finish-interview")
def finish_interview() -> Tuple[Any, int]:
    """Finalize a session and compute a complete report card."""
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id", "")).strip()

    if not session_id:
        return _json_error("session_id is required.", code="MISSING_SESSION")
    if not interview_engine.session_exists(session_id):
        return _json_error("Session not found.", status_code=404, code="SESSION_NOT_FOUND")

    session_payload = interview_engine.session_to_dict(session_id)
    report_payload = report_generator.generate_report(session_payload)
    interview_engine.store_report(session_id, report_payload)

    return (
        jsonify(
            {
                "success": True,
                "session_id": session_id,
                "report_url": f"/report/{session_id}",
                "summary": report_payload.get("scores", {}),
            }
        ),
        200,
    )


@app.get("/api/report/<session_id>")
def get_report(session_id: str) -> Tuple[Any, int]:
    """Return a freshly generated report payload for a given session."""
    if not interview_engine.session_exists(session_id):
        return _json_error("Session not found.", status_code=404, code="SESSION_NOT_FOUND")

    session_payload = interview_engine.session_to_dict(session_id)
    report_payload = report_generator.generate_report(session_payload)
    interview_engine.store_report(session_id, report_payload)

    return jsonify({"success": True, "session_id": session_id, "report": report_payload}), 200


@app.get("/api/report/<session_id>/pdf")
def download_report_pdf(session_id: str) -> Any:
    """Generate and return a PDF export from a freshly generated session report."""
    if not interview_engine.session_exists(session_id):
        return _json_error("Session not found.", status_code=404, code="SESSION_NOT_FOUND")

    session_payload = interview_engine.session_to_dict(session_id)
    report_payload = report_generator.generate_report(session_payload)
    interview_engine.store_report(session_id, report_payload)

    pdf_stream = _build_pdf(report_payload, session_id)
    return send_file(
        pdf_stream,
        as_attachment=True,
        download_name=f"InterviewIQ_Report_{session_id}.pdf",
        mimetype="application/pdf",
    )


@socketio.on("connect", namespace="/socket")
def on_socket_connect() -> None:
    """Handle Socket.IO client connections for emotion streaming."""
    emit("socket_ready", {"success": True, "message": "Emotion stream connected."})


@socketio.on("emotion_frame", namespace="/socket")
def on_emotion_frame(data: Dict[str, Any]) -> None:
    """Analyze an incoming webcam frame and emit real-time emotion feedback."""
    frame_data = str(data.get("frame", ""))
    session_id = str(data.get("session_id", ""))

    if not frame_data:
        emit("emotion_update", {"success": False, "error": "Missing frame payload."})
        return

    frame = _decode_frame(frame_data)
    if frame is None:
        emit("emotion_update", {"success": False, "error": "Invalid frame payload."})
        return

    analysis = facial_analyzer.analyze_realtime_frame(frame)
    emit(
        "emotion_update",
        {
            "success": True,
            "session_id": session_id,
            "emotion": analysis.get("dominant_emotion", "unknown"),
            "confidence": analysis.get("confidence", 0.0),
            "emotion_scores": analysis.get("emotion_scores", {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.errorhandler(413)
def file_too_large(_: Exception) -> Tuple[Any, int]:
    """Return a JSON message when uploads exceed configured body size."""
    return _json_error("Uploaded file is too large.", status_code=413, code="FILE_TOO_LARGE")


@app.errorhandler(HTTPException)
def handle_http_exception(error: HTTPException) -> Any:
    """Preserve HTTP status semantics and return JSON only for API routes."""
    if request.path.startswith("/api/"):
        code = (error.name or "HTTP_ERROR").upper().replace(" ", "_")
        return _json_error(error.description or "HTTP error", status_code=error.code or 500, code=code)
    return error


@app.errorhandler(Exception)
def handle_unexpected_error(error: Exception) -> Tuple[Any, int]:
    """Return consistent JSON errors for API endpoints and HTML for pages."""
    app.logger.exception("Unhandled server error on %s", request.path)
    if request.path.startswith("/api/"):
        return _json_error("Unexpected server error.", status_code=500, code="INTERNAL_ERROR")
    return "Unexpected server error.", 500


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
    socketio.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=debug_mode)
