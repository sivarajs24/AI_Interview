class ReportDashboard {
    /**
     * Render report card data, charts, and interactive breakdown widgets.
     * @param {{sessionId: string, reportApiUrl: string, downloadUrl: string}} config
     */
    constructor(config) {
        this.config = config;
        this.report = null;
        this.radarChart = null;
        this.timelineChart = null;

        this.elements = {
            reportMeta: document.getElementById("reportMeta"),
            confidenceRing: document.getElementById("confidenceRing"),
            sentimentRing: document.getElementById("sentimentRing"),
            stressRing: document.getElementById("stressRing"),
            confidenceScore: document.getElementById("confidenceScore"),
            sentimentScore: document.getElementById("sentimentScore"),
            stressScore: document.getElementById("stressScore"),
            radarCanvas: document.getElementById("radarChart"),
            timelineCanvas: document.getElementById("timelineChart"),
            questionAccordion: document.getElementById("questionAccordion"),
            suggestionsList: document.getElementById("suggestionsList"),
            downloadBtn: document.getElementById("downloadReportBtn"),
        };
    }

    /**
     * Initialize report page by fetching and painting all sections.
     */
    async init() {
        this.bindEvents();

        try {
            window.UI.setLoading(true, "Loading report card...");
            this.report = await this.fetchReport();
            this.renderMeta();
            this.renderScores();
            this.renderCharts();
            this.renderQuestionAccordion();
            this.renderSuggestions();
        } catch (error) {
            window.UI.showToast(`Unable to load report: ${error.message}`, "error");
        } finally {
            window.UI.setLoading(false);
        }
    }

    /**
     * Attach UI events.
     */
    bindEvents() {
        this.elements.downloadBtn.addEventListener("click", () => {
            window.open(this.config.downloadUrl, "_blank", "noopener,noreferrer");
        });
    }

    /**
     * Fetch report payload from backend API.
     * @returns {Promise<any>}
     */
    async fetchReport() {
        const response = await fetch(this.config.reportApiUrl, {
            method: "GET",
            cache: "no-store",
            headers: {
                "Cache-Control": "no-cache",
                Pragma: "no-cache",
            },
        });
        const payload = await response.json();

        if (!response.ok || !payload.success) {
            throw new Error(payload.error?.message || "Report endpoint returned an error.");
        }
        return payload.report;
    }

    /**
     * Render candidate and session metadata.
     */
    renderMeta() {
        const meta = this.report.meta || {};
        this.elements.reportMeta.textContent = `${meta.candidate_name || "Candidate"} • ${meta.target_role || "Role"} • ${meta.total_questions || 0} questions`;
    }

    /**
     * Animate and paint main score rings.
     */
    renderScores() {
        const scores = this.report.scores || {};
        this.animateRing(this.elements.confidenceRing, this.elements.confidenceScore, Number(scores.confidence_score || 0), "var(--accent-sky)");
        this.animateRing(this.elements.sentimentRing, this.elements.sentimentScore, Number(scores.sentiment_score || 0), "var(--ok)");
        this.animateRing(this.elements.stressRing, this.elements.stressScore, Number(scores.stress_score || 0), "var(--accent-coral)");

        const feedbackText = this.describeScoreState(scores);
        if (feedbackText) {
            window.UI.showToast(feedbackText, "info");
        }
    }

    /**
     * Animate a circular progress ring and score text.
     * @param {HTMLElement} ring
     * @param {HTMLElement} label
     * @param {number} target
     * @param {string} color
     */
    animateRing(ring, label, target, color) {
        const clamped = Math.max(0, Math.min(100, target));
        const duration = 1200;
        const start = performance.now();

        const tick = (now) => {
            const progress = Math.min(1, (now - start) / duration);
            const value = clamped * progress;
            label.textContent = `${Math.round(value)}`;
            ring.style.background = `conic-gradient(${color} ${value * 3.6}deg, rgba(255,255,255,0.08) 0deg)`;

            if (progress < 1) {
                requestAnimationFrame(tick);
            }
        };

        requestAnimationFrame(tick);
    }

    /**
     * Render radar and timeline charts.
     */
    renderCharts() {
        const charts = this.report.charts || {};
        const radar = charts.radar || { labels: [], values: [] };
        const timeline = this.normalizeTimeline(
            charts.emotion_timeline || {
                labels: [],
                values: [],
                emotions: [],
                seconds: [],
                total_duration_seconds: 0,
            },
            Number(this.report?.meta?.total_questions || 0)
        );

        if (this.radarChart) {
            this.radarChart.destroy();
            this.radarChart = null;
        }

        if (this.timelineChart) {
            this.timelineChart.destroy();
            this.timelineChart = null;
        }

        if (!radar.labels.length || !radar.values.length) {
            this.elements.radarCanvas.closest("article").classList.add("hidden");
        }

        if (!timeline.labels.length || !timeline.values.length) {
            this.elements.timelineCanvas.closest("article").classList.add("hidden");
        }

        if (!radar.labels.length || !radar.values.length || !timeline.labels.length || !timeline.values.length) {
            window.UI.showToast("Some chart sections were unavailable in this report.", "warning");
        }

        if (radar.labels.length && radar.values.length) {
            this.radarChart = new Chart(this.elements.radarCanvas, {
                type: "radar",
                data: {
                    labels: radar.labels,
                    datasets: [
                        {
                            label: "Interview Scores",
                            data: radar.values,
                            fill: true,
                            borderColor: "#4ec5ff",
                            backgroundColor: "rgba(78, 197, 255, 0.24)",
                            pointBackgroundColor: "#84ff9e",
                            pointRadius: 4,
                            pointHoverRadius: 5,
                            borderWidth: 2,
                        },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    plugins: {
                        legend: { labels: { color: "#f5f9ff" } },
                    },
                    scales: {
                        r: {
                            min: 0,
                            max: 100,
                            ticks: { color: "#9fb7cd", stepSize: 20, backdropColor: "rgba(0,0,0,0)" },
                            grid: { color: "rgba(176, 219, 252, 0.2)" },
                            angleLines: { color: "rgba(176, 219, 252, 0.2)" },
                            pointLabels: { color: "#eaf5ff", font: { size: 12 } },
                        },
                    },
                },
            });
        }

        if (timeline.labels.length && timeline.values.length) {
            const timelinePoints = timeline.seconds.map((second, index) => ({
                x: second,
                y: timeline.values[index],
            }));

            this.timelineChart = new Chart(this.elements.timelineCanvas, {
                type: "line",
                data: {
                    datasets: [
                        {
                            label: "Emotion Trend",
                            data: timelinePoints,
                            borderColor: "#ff6f61",
                            backgroundColor: "rgba(255, 111, 97, 0.18)",
                            tension: 0.25,
                            fill: true,
                            pointRadius: 2,
                            pointHoverRadius: 4,
                            borderWidth: 2,
                        },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    parsing: false,
                    normalized: true,
                    plugins: {
                        legend: { labels: { color: "#f5f9ff" } },
                        tooltip: {
                            callbacks: {
                                title: (items) => {
                                    const first = items[0];
                                    if (!first) {
                                        return "";
                                    }
                                    return this.formatClock(Number(first.parsed.x || 0));
                                },
                                label: (context) => {
                                    const emotion = timeline.emotions[context.dataIndex] || "unknown";
                                    return ` ${emotion}`;
                                },
                            },
                        },
                    },
                    scales: {
                        x: {
                            type: "linear",
                            min: 0,
                            max: timeline.maxSeconds,
                            ticks: {
                                color: "#9fb7cd",
                                maxTicksLimit: 8,
                                callback: (value) => this.formatClock(Number(value || 0)),
                            },
                            grid: { color: "rgba(176, 219, 252, 0.14)" },
                        },
                        y: {
                            min: 0,
                            max: 5,
                            ticks: {
                                color: "#9fb7cd",
                                stepSize: 1,
                                callback(value) {
                                    const labels = ["Disgust", "Sad", "Fear", "Surprise", "Neutral", "Happy"];
                                    return labels[value] || value;
                                },
                            },
                            grid: { color: "rgba(176, 219, 252, 0.14)" },
                        },
                    },
                },
            });
        }
    }

    /**
     * Normalize backend timeline payload and enforce finite chart bounds.
     * @param {any} timeline
     * @returns {{labels: string[], values: number[], emotions: string[], seconds: number[], maxSeconds: number}}
     */
    normalizeTimeline(timeline, totalQuestions = 0) {
        const rawLabels = Array.isArray(timeline?.labels) ? timeline.labels : [];
        const rawValues = Array.isArray(timeline?.values) ? timeline.values : [];
        const rawEmotions = Array.isArray(timeline?.emotions) ? timeline.emotions : [];
        const rawSeconds = Array.isArray(timeline?.seconds) ? timeline.seconds : [];

        const maxDuration = 1800;
        const maxPoints = 80;
        const questionBound = Number.isFinite(totalQuestions) && totalQuestions > 0
            ? Math.min(maxDuration, totalQuestions * 180)
            : maxDuration;
        const payloadDuration = Number(timeline?.total_duration_seconds);
        const hasPayloadDuration = Number.isFinite(payloadDuration) && payloadDuration > 0;
        const upperBound = hasPayloadDuration
            ? Math.max(1, Math.min(questionBound, payloadDuration))
            : questionBound;

        const total = Math.min(rawValues.length, Math.max(rawLabels.length, rawSeconds.length));
        const labels = [];
        const values = [];
        const emotions = [];
        const seconds = [];

        let lastSecond = 0;
        for (let i = 0; i < total && labels.length < maxPoints; i += 1) {
            const value = Number(rawValues[i]);
            if (!Number.isFinite(value)) {
                continue;
            }

            let second = Number(rawSeconds[i]);
            if (!Number.isFinite(second)) {
                second = this.parseClockLabel(rawLabels[i]);
            }
            if (!Number.isFinite(second)) {
                second = lastSecond;
            }

            second = Math.max(lastSecond, second);
            second = Math.max(0, Math.min(upperBound, second));
            lastSecond = second;

            values.push(Math.max(0, Math.min(5, Math.round(value))));
            emotions.push(String(rawEmotions[i] || "unknown"));
            seconds.push(second);
            labels.push(this.formatClock(second));
        }

        const derivedMax = seconds.length ? seconds[seconds.length - 1] : upperBound;
        let maxSeconds = Math.max(1, Math.min(questionBound, derivedMax));
        if (hasPayloadDuration) {
            maxSeconds = Math.max(maxSeconds, Math.min(questionBound, payloadDuration));
        }

        return { labels, values, emotions, seconds, maxSeconds };
    }

    /**
     * Parse mm:ss labels into total seconds.
     * @param {string} value
     * @returns {number}
     */
    parseClockLabel(value) {
        const text = String(value || "");
        const match = text.match(/(\d{1,3}):(\d{2})/);
        if (!match) {
            return Number.NaN;
        }

        const minutes = Number(match[1]);
        const seconds = Number(match[2]);
        if (!Number.isFinite(minutes) || !Number.isFinite(seconds)) {
            return Number.NaN;
        }
        return (minutes * 60) + seconds;
    }

    /**
     * Format seconds into mm:ss.
     * @param {number} value
     * @returns {string}
     */
    formatClock(value) {
        const safe = Math.max(0, Math.min(1800, Number(value || 0)));
        const minutes = Math.floor(safe / 60)
            .toString()
            .padStart(2, "0");
        const seconds = Math.floor(safe % 60)
            .toString()
            .padStart(2, "0");
        return `${minutes}:${seconds}`;
    }

    /**
     * Render expandable per-question analysis rows.
     */
    renderQuestionAccordion() {
        const rows = this.report.question_breakdown || [];
        this.elements.questionAccordion.innerHTML = "";

        rows.forEach((row) => {
            const details = document.createElement("details");
            details.className = "accordion-item";

            const summary = document.createElement("summary");
            summary.innerHTML = `
                <span>Q${row.question_index + 1} • ${this.escapeHtml(row.category)}</span>
                <span>${this.escapeHtml(row.sentiment.label)} • Fluency ${Math.round(row.fluency_score)}</span>
            `;
            details.appendChild(summary);

            const body = document.createElement("div");
            body.className = "accordion-content";

            const highlightedTranscript = this.renderHighlightedTranscript(
                row.transcript,
                row.sentiment.sentence_scores || []
            );

            const unsureMoments = (row.sentiment.unsure_moments || [])
                .map((item) => `<li>${this.escapeHtml(item.sentence)}</li>`)
                .join("");

            body.innerHTML = `
                <p class="question-text"><strong>Question:</strong> ${this.escapeHtml(row.question)}</p>
                <p><strong>Dominant Emotion:</strong> ${this.escapeHtml(row.emotions.dominant_emotion)}</p>
                <p><strong>Duration:</strong> ${Number(row.duration_seconds).toFixed(1)}s</p>
                <p><strong>Transcript:</strong></p>
                <div class="transcript-highlight">${highlightedTranscript}</div>
                <p><strong>Unsure Moments:</strong></p>
                <ul>${unsureMoments || "<li>None detected.</li>"}</ul>
            `;
            details.appendChild(body);

            this.elements.questionAccordion.appendChild(details);
        });
    }

    /**
     * Render transcript with sentence-level color-coded sentiment highlights.
     * @param {string} transcript
     * @param {Array<any>} sentenceScores
     * @returns {string}
     */
    renderHighlightedTranscript(transcript, sentenceScores) {
        if (!sentenceScores.length) {
            return this.escapeHtml(transcript || "Transcription unavailable");
        }

        return sentenceScores
            .map((item) => {
                const label = String(item.label || "neutral").toLowerCase();
                const cls = label === "positive" ? "sent-positive" : label === "negative" ? "sent-negative" : "sent-neutral";
                return `<span class="${cls}">${this.escapeHtml(item.sentence)}</span>`;
            })
            .join(" ");
    }

    /**
     * Render numbered actionable coaching cards.
     */
    renderSuggestions() {
        const suggestions = this.report.suggestions || [];
        this.elements.suggestionsList.innerHTML = "";

        if (!suggestions.length) {
            const empty = document.createElement("p");
            empty.textContent = "No suggestions were generated for this session.";
            this.elements.suggestionsList.appendChild(empty);
            return;
        }

        suggestions.forEach((suggestion, index) => {
            const card = document.createElement("article");
            card.className = "suggestion-card";
            card.innerHTML = `
                <span class="suggestion-number">${index + 1}</span>
                <p>${this.escapeHtml(suggestion)}</p>
            `;
            this.elements.suggestionsList.appendChild(card);
        });
    }

    /**
     * Escape HTML-sensitive characters for safe rendering.
     * @param {string} value
     * @returns {string}
     */
    escapeHtml(value) {
        return String(value || "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    /**
     * Build a short user-friendly score summary line for immediate context.
     * @param {any} scores
     * @returns {string}
     */
    describeScoreState(scores) {
        const confidence = Number(scores?.confidence_score || 0);
        const sentiment = Number(scores?.sentiment_score || 0);
        const stress = Number(scores?.stress_score || 0);

        if (confidence >= 75 && sentiment >= 70 && stress <= 45) {
            return "Strong interview baseline detected. Keep this structure for your next practice run.";
        }

        if (confidence < 55) {
            return "Confidence score is still building. Try shorter sentences and clearer end points.";
        }

        if (stress > 65) {
            return "Stress indicators were elevated. Add brief pauses to reset pace between ideas.";
        }

        return "Review your per-question breakdown to target the biggest upside areas first.";
    }
}

window.addEventListener("DOMContentLoaded", () => {
    const dashboard = new ReportDashboard(window.REPORT_CONFIG);
    dashboard.init();
});
