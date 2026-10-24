"""
app.py — Flask backend for the SMS Spam Detector.
Run:  python app.py
Env vars:
  FLASK_DEBUG   - set to "true" to enable debug mode (default: false)
  PORT          - port to listen on (default: 5000)
  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD - PostgreSQL settings
                  (prediction logging is skipped if DB_HOST is not set)
  MODEL_VERSION - label stored with each logged prediction (default: v1)
"""

import os

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import psycopg2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

MAX_MESSAGE_LENGTH = 1000  # characters

# ---------------------------------------------------------------------------
# Database settings
# ---------------------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
model = None
vectorizer = None

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"[WARNING] Model file not found at '{MODEL_PATH}'. Run the notebook to generate it.")

try:
    vectorizer = joblib.load(VECTORIZER_PATH)
except FileNotFoundError:
    print(f"[WARNING] Vectorizer file not found at '{VECTORIZER_PATH}'. Run the notebook to generate it.")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)


def log_prediction(message, prediction, confidence):
    """Save one prediction to PostgreSQL. Never raises, so a DB problem can't break /predict."""
    if not DB_HOST:
        return
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
            user=DB_USER, password=DB_PASSWORD, connect_timeout=3,
        )
        try:
            with conn, conn.cursor() as cur:  # "with conn" commits the transaction
                cur.execute(
                    "INSERT INTO predictions (message, prediction, confidence, model_version) "
                    "VALUES (%s, %s, %s, %s)",
                    (message, prediction, confidence, MODEL_VERSION),
                )
        finally:
            conn.close()
    except Exception as exc:
        print(f"[WARNING] Could not log prediction: {exc}", flush=True)


@app.route("/")
def home():
    status = "ready" if (model and vectorizer) else "model not loaded"
    return jsonify({"status": status, "service": "SMS Spam Detector API"})


@app.route("/ui")
def ui():
    """Serve the frontend page so it can be opened at http://127.0.0.1:5000/ui"""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: { "message": "<sms text>" }
    Returns: { "message", "prediction", "confidence" }
    """
    if not model or not vectorizer:
        return jsonify({"error": "Model or vectorizer not loaded on the server."}), 503

    # NOTE: `request.get_json` also accepts JSON arrays/strings/numbers, so the
    # payload must be checked for being an object before calling .get() on it —
    # otherwise a list body raised AttributeError and returned HTTP 500.
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON body."}), 400

    message = data.get("message", "")
    if not isinstance(message, str):
        return jsonify({"error": "Field 'message' must be a string."}), 400

    message = message.strip()

    if not message:
        return jsonify({"error": "No message provided."}), 400

    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({"error": f"Message too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed."}), 400

    message_vector = vectorizer.transform([message])
    prediction_code = model.predict(message_vector)[0]
    probabilities = model.predict_proba(message_vector)[0]

    prediction_label = "Spam" if prediction_code == 1 else "Ham"
    confidence = round(float(max(probabilities)) * 100, 2)

    log_prediction(message, prediction_label, confidence)

    return jsonify({
        "message": message,
        "prediction": prediction_label,
        "confidence": confidence,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug)