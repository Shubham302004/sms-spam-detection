# 📱 SMS Spam Detector

A machine learning web app that classifies SMS messages as **Spam** or **Ham (Not Spam)** in real time, with a confidence score.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🖥️ Demo

> Enter any SMS text → click **Analyze Message** → get an instant prediction with confidence %.

---

## 🗂️ Project Structure

```
spam-detector-app/
├── backend/
│   └── app.py               # Flask API server
├── frontend/
│   └── index.html           # Single-page UI (Tailwind CSS)
├── notebooks/
│   └── spam_detection.ipynb # Model training & evaluation
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/spam-detector-app.git
cd spam-detector-app
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the backend

```bash
cd backend
python app.py
```

The API will be available at `http://127.0.0.1:5000`.

### 7. Open the frontend

Open `frontend/index.html` directly in your browser. No build step required.

---

## 🔌 API Reference

### `GET /`

Health check.

```json
{ "status": "ready", "service": "SMS Spam Detector API" }
```

### `POST /predict`

**Request body:**
```json
{ "message": "You have won a prize! Call now." }
```

**Response:**
```json
{
  "message": "You have won a prize! Call now.",
  "prediction": "Spam",
  "confidence": 98.7
}
```

**Error responses:**

| Status | Reason |
|--------|--------|
| 400 | Missing or empty message |
| 400 | Message exceeds 1000 characters |
| 503 | Model not loaded (run the notebook first) |

---

## 🧠 How It Works

1. **Dataset:** UCI SMS Spam Collection (~5,500 labeled messages)
2. **Preprocessing:** Lowercasing, punctuation removal, stopword filtering
3. **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
4. **Classifier:** Naive Bayes / Logistic Regression (see notebook for details)
5. **Serving:** Flask REST API with CORS enabled

---

## 🌍 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_DEBUG` | `false` | Set to `true` for hot-reload during development |
| `PORT` | `5000` | Port the Flask server listens on |

Example:
```bash
FLASK_DEBUG=true PORT=8080 python backend/app.py
```

---

## 🛡️ Security Notes

- User input is HTML-escaped before rendering in the history panel (XSS prevention)
- Message length is capped at 1000 characters on both client and server
- Debug mode is off by default

---

## 📄 License

MIT — feel free to use, modify, and distribute.
