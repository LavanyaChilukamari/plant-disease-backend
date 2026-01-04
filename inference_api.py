import json
import os
import uuid
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image

from database import init_db, ensure_user, save_scan
from model_loader import load_model_and_classes

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "Frontend")

# ============================================================
# APP
# ============================================================
app = Flask(
    __name__,
    static_folder=FRONTEND_DIR if os.path.exists(FRONTEND_DIR) else None,
    static_url_path=""
)
CORS(app)

# ============================================================
# CONFIG
# ============================================================
IMG_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

REJECT_THRESHOLD = 35.0   # definitely NOT a leaf
WARN_THRESHOLD = 60.0     # leaf but unclear

# ============================================================
# INIT
# ============================================================
init_db()
model, class_names = load_model_and_classes()

with open(os.path.join(BASE_DIR, "disease_metadata.json"), "r", encoding="utf-8") as f:
    DISEASE_METADATA = json.load(f)

# ============================================================
# FRONTEND ROUTES (ONLY IF PRESENT)
# ============================================================
if os.path.exists(FRONTEND_DIR):

    @app.route("/")
    def index():
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.route("/<path:path>")
    def static_files(path):
        return send_from_directory(FRONTEND_DIR, path)

# ============================================================
# HEALTH
# ============================================================
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# ============================================================
# UTILS
# ============================================================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def normalize_key(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
    )

# ============================================================
# PREDICT
# ============================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ---------------- USER ----------------
        user_id = request.form.get("user_id")
        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400

        uuid.UUID(user_id)
        ensure_user(user_id)

        # ---------------- IMAGE ----------------
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        if not allowed_file(file.filename):
            return jsonify({
                "error": "Unsupported image format",
                "accepted_images": list(ALLOWED_EXTENSIONS)
            }), 400

        try:
            img = Image.open(file).convert("RGB").resize(IMG_SIZE)
        except Exception:
            return jsonify({"error": "Invalid image file"}), 400

        img_array = np.expand_dims(
            np.array(img, dtype=np.float32) / 255.0,
            axis=0
        )

        # ---------------- MODEL ----------------
        preds = model.predict(img_array, verbose=0)
        class_idx = int(np.argmax(preds))
        confidence = round(float(np.max(preds)) * 100, 2)

        # ---------------- HARD REJECT ----------------
        if confidence < REJECT_THRESHOLD:
            return jsonify({
                "status": "rejected",
                "reason": "not_leaf",
                "message": "This image does not appear to be a plant leaf.",
                "instruction": "Upload a clear photo of a single leaf on a plain background.",
                "confidence": confidence
            }), 422

        # ---------------- CLASS ----------------
        disease_raw = class_names.get(class_idx, "Unknown").strip()
        disease_key = normalize_key(disease_raw)
        low_confidence = confidence < WARN_THRESHOLD

        # ---------------- RESULT BUILD ----------------
        if "healthy" in disease_key:
            result = {
                "status": "healthy",
                "disease": "Healthy Leaf",
                "cure": ["No treatment required"],
                "prevention": [
                    "Water regularly",
                    "Provide adequate sunlight",
                    "Inspect leaves weekly"
                ],
                "leaf_uses": "Healthy leaves enable photosynthesis and plant growth.",
                "summary": "The leaf appears healthy."
            }
        else:
            meta = DISEASE_METADATA.get(disease_key)

            if not meta:
                # SAFE FALLBACK — NEVER EMPTY UI
                result = {
                    "status": "diseased",
                    "disease": disease_raw,
                    "cure": [
                        "Remove infected leaves",
                        "Avoid overhead irrigation",
                        "Apply appropriate fungicide if needed"
                    ],
                    "prevention": [
                        "Ensure good air circulation",
                        "Avoid wet foliage",
                        "Use disease-resistant plant varieties"
                    ],
                    "leaf_uses": "Disease may reduce plant productivity.",
                    "summary": "Disease detected, but detailed metadata is unavailable."
                }
            else:
                result = {
                    "status": meta.get("status", "diseased"),
                    "disease": disease_raw,
                    "cure": meta.get("cure", []),
                    "prevention": meta.get("prevention", []),
                    "leaf_uses": meta.get("leaf_uses", ""),
                    "summary": meta.get(
                        "summary",
                        "Disease detected. Follow recommended steps."
                    )
                }

        save_scan(user_id, disease_raw, confidence)

        return jsonify({
            **result,
            "confidence": confidence,
            "low_confidence": low_confidence
        })

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
