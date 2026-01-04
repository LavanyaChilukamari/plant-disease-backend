import os
import json
import tensorflow as tf

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_model.keras")
CLASS_PATH = os.path.join(MODEL_DIR, "class_names.json")

# ============================================================
# OPTIONAL: Silence TensorFlow warnings (safe for prod)
# ============================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ============================================================
# LOAD MODEL + CLASSES
# ============================================================
def load_model_and_classes():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    if not os.path.exists(CLASS_PATH):
        raise FileNotFoundError(f"Class names file not found at: {CLASS_PATH}")

    # Load model (compile=False is correct for inference)
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    # Load class names
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    # Ensure keys are integers (important)
    class_names = {int(k): v for k, v in class_names.items()}

    return model, class_names
