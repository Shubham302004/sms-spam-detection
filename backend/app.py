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

try:
    import psycopg2
except ImportError:
    psycopg2 = None

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

_table_initialized = False

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
model = None
vectorizer = None

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    print(f"[WARNING] Could not load model from '{MODEL_PATH}': {exc}")

try:
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as exc:
    print(f"[WARNING] Could not load vectorizer from '{VECTORIZER_PATH}': {exc}")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)


def check_db_status():
    """Quick check for PostgreSQL connectivity ('connected', 'disconnected', or 'disabled')."""
    if not DB_HOST or psycopg2 is None:
        return "disabled"
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
            user=DB_USER, password=DB_PASSWORD, connect_timeout=2,
        )
        conn.close()
        return "connected"
    except Exception:
        return "disconnected"


def init_db():
    """Ensure the predictions table exists in PostgreSQL. Safe to run multiple times."""
    global _table_initialized
    if not DB_HOST or psycopg2 is None:
        return
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
            user=DB_USER, password=DB_PASSWORD, connect_timeout=3,
        )
        try:
            with conn, conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        message TEXT NOT NULL,
                        prediction VARCHAR(10) NOT NULL,
                        confidence REAL NOT NULL,
                        model_version VARCHAR(50) DEFAULT 'v1',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            _table_initialized = True
        finally:
            conn.close()
    except Exception as exc:
        print(f"[WARNING] Could not initialize database table: {exc}", flush=True)


def log_prediction(message, prediction, confidence):
    """Save one prediction to PostgreSQL. Never raises, so a DB problem can't break /predict."""
    if not DB_HOST or psycopg2 is None:
        return
    if not _table_initialized:
        init_db()
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


# Attempt initial table check at startup
init_db()


@app.route("/")
def home():
    model_loaded = model is not None and vectorizer is not None
    status = "ready" if model_loaded else "model not loaded"
    return jsonify({
        "status": status,
        "service": "SMS Spam Detector API",
        "model": "loaded" if model_loaded else "not loaded",
        "database": check_db_status(),
    })


@app.route("/ui")
def ui():
    """Serve the frontend page so it can be opened at http://127.0.0.1:5000/ui"""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/predictions", methods=["GET"])
def get_predictions():
    """
    GET /predictions?limit=20
    Returns recent predictions stored in PostgreSQL.
    """
    if not DB_HOST or psycopg2 is None:
        return jsonify({
            "predictions": [],
            "status": "disabled",
            "message": "Database logging is disabled (DB_HOST not set)."
        }), 200

    try:
        limit = max(1, min(int(request.args.get("limit", 20)), 100))
    except (ValueError, TypeError):
        limit = 20

    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
            user=DB_USER, password=DB_PASSWORD, connect_timeout=3,
        )
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, message, prediction, confidence, model_version, created_at
                    FROM predictions
                    ORDER BY id DESC
                    LIMIT %s
                """, (limit,))
                rows = cur.fetchall()
        finally:
            conn.close()

        results = [
            {
                "id": r[0],
                "message": r[1],
                "prediction": r[2],
                "confidence": r[3],
                "model_version": r[4],
                "created_at": r[5].isoformat() if r[5] else None,
            }
            for r in rows
        ]
        return jsonify({"predictions": results, "count": len(results)})
    except Exception as exc:
        return jsonify({
            "predictions": [],
            "error": f"Could not retrieve predictions: {exc}"
        }), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: { "message": "<sms text>" }
    Returns: { "message", "prediction", "confidence" }
    """
    if model is None or vectorizer is None:
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
