from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Restrict this to your deployed frontend URL after deployment.

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
_bundle = None

def get_bundle():
    global _bundle
    if _bundle is None and MODEL_PATH.exists():
        _bundle = joblib.load(MODEL_PATH)
    return _bundle

@app.get("/")
def index():
    return jsonify({
        "name": "PhiUSIIL Phishing Detection API",
        "endpoints": ["/health", "/sample", "/predict"]
    })

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_ready": MODEL_PATH.exists()})

@app.get("/sample")
def sample():
    """Returns an example feature vector from the trained feature schema, not a real URL scan."""
    bundle = get_bundle()
    if bundle is None:
        return jsonify({"error": "Model not trained yet. Run python backend/train.py"}), 503
    # Use neutral numeric values. This is a UI/demo payload, not a real URL's extracted features.
    return jsonify({
        "features": {name: 0 for name in bundle["features"]},
        "note": "Example feature vector only. It is not extracted from a real URL."
    })

@app.post("/predict")
def predict():
    bundle = get_bundle()
    if bundle is None:
        return jsonify({"error": "Model not trained yet. Run python backend/train.py"}), 503
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("features"), dict):
        return jsonify({"error": "Send JSON with a 'features' object."}), 400

    expected = bundle["features"]
    supplied = body["features"]
    row = {}
    for feature in expected:
        value = supplied.get(feature, None)
        if value is None or value == "":
            row[feature] = np.nan
        else:
            try:
                row[feature] = float(value)
            except (TypeError, ValueError):
                return jsonify({"error": f"Feature '{feature}' must be numeric or null."}), 400

    X = pd.DataFrame([row], columns=expected)
    pipeline = bundle["pipeline"]
    prediction = str(pipeline.predict(X)[0])

    probability = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X)[0]
        classes = [str(c) for c in pipeline.classes_]
        probability = {c: float(p) for c, p in zip(classes, probs)}

    # PhiUSIIL convention: label 0 = phishing, label 1 = legitimate.
    # Confirm this against your downloaded CSV's data dictionary before presenting predictions.
    if prediction == "0":
        category = "Phishing"
    elif prediction == "1":
        category = "Legitimate"
    else:
        category = f"Predicted class: {prediction}"

    return jsonify({
        "prediction": prediction,
        "category": category,
        "class_probabilities": probability,
        "note": "Prediction uses supplied dataset features. It is not a live webpage scan."
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
