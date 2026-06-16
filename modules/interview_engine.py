"""Interview session engine with curated question orchestration."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .qwen_interviewer import QwenInterviewLLM


@dataclass
class InterviewSession:
    """Represent one interview run and all data collected during it."""

    session_id: str
    candidate_name: str
    target_role: str
    created_at: str
    questions: List[Dict[str, Any]]
    responses: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    report: Optional[Dict[str, Any]] = None


class InterviewEngine:
    """Manage interview sessions, question sequencing, and response storage."""

    QUESTION_BANK: Dict[str, List[str]] = {
        "Introduction": [
            "Tell me about yourself and the journey that brought you here.",
            "What strengths make you a great fit for this role?",
            "Which recent project are you most proud of, and why?",
            "How do you prefer to collaborate with cross-functional teams?",
            "What motivates you to do your best work consistently?",
            "How do you stay current with changes in your domain?",
            "What does success in your next role look like to you?",
            "Describe the work environment where you perform at your best.",
            "What are your long-term career goals and how does this role fit?",
            "How do you approach learning a completely new technology quickly?",
            "What kind of feedback helps you improve the fastest?",
            "How would your colleagues describe your communication style?",
        ],
        "Technical": [
            "How would you design a scalable REST API for millions of requests per day?",
            "Explain the trade-offs between SQL and NoSQL databases in production systems.",
            "How do you identify and fix a memory leak in a backend service?",
            "Walk through how you would optimize a slow machine learning inference pipeline.",
            "How do you structure error handling and observability in microservices?",
            "Describe your process for reviewing pull requests effectively.",
            "How would you secure user authentication and session management in a web app?",
            "Explain eventual consistency and where it can cause user-facing issues.",
            "How would you containerize and deploy a Python service to Kubernetes?",
            "What steps do you take to reduce model drift after deployment?",
            "How do you design feature engineering pipelines for reproducibility?",
            "Describe how you would benchmark two competing algorithmic approaches.",
            "How would you debug intermittent latency spikes in a distributed API?",
            "Explain CI/CD best practices for safe releases in production.",
            "How do you choose between synchronous and asynchronous processing patterns?",
            "What is your strategy for schema evolution without downtime?",
            "How do you validate data quality before training an ML model?",
            "Describe how you would implement role-based access control in a SaaS app.",
            "How do you evaluate precision-recall trade-offs for an imbalanced dataset?",
            "Explain caching strategy decisions for read-heavy platforms.",
            "How do you isolate and reproduce a bug that appears only in production?",
            "What telemetry would you instrument for an AI-powered user workflow?",
        ],
        "Behavioral": [
            "Tell me about a time you handled a difficult stakeholder conversation.",
            "Describe a project where requirements changed midway. How did you adapt?",
            "Share an example of receiving tough feedback and how you responded.",
            "Tell me about a time you missed a deadline and what you learned.",
            "Describe a moment where you helped unblock a teammate.",
            "Give an example of when you had to make a decision with incomplete data.",
            "Describe a time you led without formal authority.",
            "Tell me about a conflict on your team and how it was resolved.",
            "Share an example of balancing speed with quality under pressure.",
            "Describe a failure that changed how you approach your work.",
            "Tell me about a time you had to communicate complex ideas simply.",
            "Give an example of how you prioritized competing tasks effectively.",
        ],
        "Situational": [
            "If production goes down during your on-call shift, what do you do first?",
            "How would you respond if an executive asks for an unrealistic delivery date?",
            "If your model accuracy drops suddenly after deployment, how do you investigate?",
            "How would you handle a teammate repeatedly bypassing code review guidelines?",
            "If you inherit legacy code with no tests, where would you start?",
            "How would you plan a migration from monolith to microservices safely?",
            "What would you do if customer feedback contradicts current product priorities?",
            "How would you triage multiple critical bugs reported at the same time?",
            "If you disagree with a senior engineer's architectural decision, how do you proceed?",
            "How would you recover trust after shipping a defect that impacts customers?",
            "If a junior engineer asks for help minutes before release, what is your approach?",
            "How would you estimate a feature with high technical uncertainty?",
        ],
    }

    def __init__(self, llm_client: Optional[QwenInterviewLLM] = None) -> None:
        """Create an in-memory session store suitable for single-instance deployments."""
        self.sessions: Dict[str, InterviewSession] = {}
        self.llm_client = llm_client

    def _is_valid_sequence(self, sequence: List[Dict[str, Any]]) -> bool:
        """Validate question sequence size and category distribution constraints."""
        if len(sequence) != 10:
            return False

        expected = ["Introduction", "Introduction", "Technical", "Technical", "Technical", "Technical", "Behavioral", "Behavioral", "Situational", "Situational"]
        categories = [str(item.get("category", "")) for item in sequence]
        return categories == expected

    def _build_question_sequence(self, candidate_name: str, target_role: str) -> List[Dict[str, Any]]:
        """Build a 10-question sequence with the required category distribution."""
        sequence: List[Dict[str, Any]] = []
        category_plan = [
            ("Introduction", 2),
            ("Technical", 4),
            ("Behavioral", 2),
            ("Situational", 2),
        ]

        for category, count in category_plan:
            picked_questions = random.sample(self.QUESTION_BANK[category], k=count)
            for question in picked_questions:
                sequence.append({"category": category, "question": question})

        for idx, item in enumerate(sequence):
            item["index"] = idx

        if self.llm_client is not None:
            llm_sequence = self.llm_client.rewrite_questions(
                base_questions=sequence,
                candidate_name=candidate_name,
                target_role=target_role,
            )
            if self._is_valid_sequence(llm_sequence):
                sequence = llm_sequence

        return sequence

    def create_session(self, candidate_name: str, target_role: str) -> InterviewSession:
        """Create and persist a new interview session."""
        session_id = uuid.uuid4().hex
        session = InterviewSession(
            session_id=session_id,
            candidate_name=candidate_name,
            target_role=target_role,
            created_at=datetime.now(timezone.utc).isoformat(),
            questions=self._build_question_sequence(candidate_name=candidate_name, target_role=target_role),
        )
        self.sessions[session_id] = session
        return session

    def active_llm_model(self) -> Optional[str]:
        """Return the active local Qwen model when LLM rewriting is enabled."""
        if self.llm_client is None:
            return None
        return self.llm_client.active_model()

    def session_exists(self, session_id: str) -> bool:
        """Return whether a session exists in storage."""
        return session_id in self.sessions

    def get_session(self, session_id: str) -> Optional[InterviewSession]:
        """Retrieve a session by identifier."""
        return self.sessions.get(session_id)

    def get_questions(self, session_id: str) -> List[Dict[str, Any]]:
        """Return the ordered question list for a session."""
        session = self.get_session(session_id)
        if session is None:
            return []
        return session.questions

    def record_response(self, session_id: str, question_index: int, response: Dict[str, Any]) -> None:
        """Store a per-question analysis payload for a given session."""
        session = self.get_session(session_id)
        if session is None:
            return
        session.responses[question_index] = response

    def store_report(self, session_id: str, report: Dict[str, Any]) -> None:
        """Attach a generated report payload to a session."""
        session = self.get_session(session_id)
        if session is None:
            return
        session.report = report

    def get_report(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return the report payload for a session if it exists."""
        session = self.get_session(session_id)
        if session is None:
            return None
        return session.report

    def session_to_dict(self, session_id: str) -> Dict[str, Any]:
        """Serialize session state for report generation and API responses."""
        session = self.get_session(session_id)
        if session is None:
            return {}

        ordered_responses: List[Dict[str, Any]] = []
        for question in session.questions:
            question_index = int(question["index"])
            ordered_responses.append(
                {
                    "question_index": question_index,
                    "category": question["category"],
                    "question": question["question"],
                    "response": session.responses.get(question_index, {}),
                }
            )

        return {
            "session_id": session.session_id,
            "candidate_name": session.candidate_name,
            "target_role": session.target_role,
            "created_at": session.created_at,
            "questions": session.questions,
            "responses": ordered_responses,
            "report": session.report,
        }
