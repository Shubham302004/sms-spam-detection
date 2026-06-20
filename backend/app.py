"""
app.py — Flask backend for the SMS Spam Detector.
Run:  python app.py
Env vars:
  FLASK_DEBUG   - set to "true" to enable debug mode (default: false)
  PORT          - port to listen on (default: 5000)
"""

import os
import string
import warnings

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

MAX_MESSAGE_LENGTH = 1000  # characters

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


@app.route("/")
def home():
    status = "ready" if (model and vectorizer) else "model not loaded"
    return jsonify({"status": status, "service": "SMS Spam Detector API"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: { "message": "<sms text>" }
    Returns: { "message", "prediction", "confidence" }
    """
    if not model or not vectorizer:
        return jsonify({"error": "Model or vectorizer not loaded on the server."}), 503

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body."}), 400

    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "No message provided."}), 400

    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({"error": f"Message too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed."}), 400

    message_vector = vectorizer.transform([message])
    prediction_code = model.predict(message_vector)[0]
    probabilities = model.predict_proba(message_vector)[0]

    prediction_label = "Spam" if prediction_code == 1 else "Ham"
    confidence = round(float(max(probabilities)) * 100, 2)

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
