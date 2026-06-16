# InterviewIQ

**InterviewIQ** is a production-ready, multi-modal AI Interview Coach that runs a full mock interview and delivers actionable, data-driven coaching feedback.

It captures webcam + microphone responses in the browser, performs local speech-to-text with Whisper, evaluates transcript sentiment with RoBERTa, analyzes facial emotions with DeepFace/OpenCV, and generates a polished interview report card with visual analytics.

---

## 🚀 Features

- 🎙️ **Live Mock Interview Flow**: 10-question structured interview across Introduction, Technical, Behavioral, and Situational categories.
- 🎥 **Browser Recording**: Simultaneous webcam and microphone recording via MediaRecorder API.
- 🧠 **Whisper Transcription**: Local OpenAI Whisper (`base`) transcription with timestamps.
- 🙂 **Facial Emotion Analysis**: DeepFace + OpenCV frame-by-frame emotion detection and stress moment extraction.
- 📝 **Sentiment Intelligence**: RoBERTa sentiment scoring for full responses and sentence-level breakdown.
- 📊 **Professional Report Card**: Confidence, Sentiment, Stress scoring with radar and timeline charts.
- 🧭 **Actionable Coaching**: Auto-generated, specific suggestions based on weak points and response patterns.
- 📄 **PDF Export**: Downloadable summary report for interview prep tracking.

---

## 🧰 Tech Stack

| Layer | Technologies |
|---|---|
| Backend | Python 3.10+, Flask, Flask-SocketIO, Flask-CORS |
| Speech-to-Text | openai-whisper (`base`) |
| NLP Sentiment | HuggingFace Transformers (`cardiffnlp/twitter-roberta-base-sentiment-latest`) |
| Vision / Emotions | DeepFace, OpenCV |
| Audio Processing | NumPy, librosa |
| Frontend | HTML5, CSS3, Vanilla JavaScript, Socket.IO client |
| Visualization | Chart.js |
| Reporting | ReportLab (PDF generation) |

---

## ⚙️ Setup Instructions

### 1) Clone and enter project

```bash
git clone <your-repo-url>
cd ai-interview-coach
```

### 2) Create and activate virtual environment

#### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Mac / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Configure environment variables

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Optional local Qwen settings in `.env`:

- `QWEN_PRIMARY_MODEL=Qwen/Qwen2.5-3B-Instruct`
- `QWEN_FALLBACK_MODEL=Qwen/Qwen2.5-1.5B-Instruct`
- `QWEN_TIMEOUT_SECONDS=12`
- `QWEN_GPU_LOW_VRAM_MB=1600`

The app uses Qwen 3B by default and automatically switches to Qwen 1.5B when generation is too slow or GPU memory is constrained.

### 5) Install FFmpeg (required by Whisper)

- **Windows**: Install FFmpeg and add it to PATH.
- **Mac**: `brew install ffmpeg`
- **Linux (Debian/Ubuntu)**: `sudo apt-get install ffmpeg`

### 6) Run the app

```bash
python app.py
```

Open in browser:

- http://127.0.0.1:5000

---

## ▶️ How To Run

1. Start a session from the landing page.
2. Allow webcam + microphone permissions.
3. Answer each question with "Start Speaking" and "Next Question".
4. Finish interview to generate the report.
5. Review charts, breakdowns, and download PDF.

---

## 📁 Project Structure

```text
ai-interview-coach/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── modules/
│   ├── __init__.py
│   ├── whisper_transcriber.py
│   ├── sentiment_analyzer.py
│   ├── facial_analyzer.py
│   ├── interview_engine.py
│   └── report_generator.py
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   ├── recorder.js
│   │   ├── interview.js
│   │   └── report.js
│   └── assets/
│       └── logo.svg
└── templates/
    ├── base.html
    ├── index.html
    ├── interview.html
    └── report.html
```

---

## 🖼️ Screenshots

- Landing Page: `docs/screenshots/landing.png` (placeholder)
- Interview Session: `docs/screenshots/interview.png` (placeholder)
- Report Dashboard: `docs/screenshots/report.png` (placeholder)

---

## ⚠️ Known Limitations

- Initial model warm-up (Whisper, DeepFace, Transformers) can take noticeable time on first use.
- CPU-only environments may experience slower analysis latency.
- Browser speech preview uses the Web Speech API and may vary by browser support.
- In-memory session storage is used by default (single-instance runtime); for multi-instance deployments, use persistent storage.

---

## 🛣️ Future Roadmap

- Multi-user authentication and role-based dashboards.
- Persistent storage with PostgreSQL + object storage.
- More detailed delivery metrics (filler words, pause cadence, eye-gaze tracking).
- Domain-specific interview packs (SWE, Data Science, Product, Management).
- Historical progress tracking across repeated interview attempts.

---

## 📄 License

MIT License
