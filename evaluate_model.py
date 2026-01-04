import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_loader import load_model_and_classes

# -----------------------------
# Constants
# -----------------------------
DATASET_PATH = r"C:\Users\chilu\OneDrive\Desktop\Plant_disease_detection\Dataset\PlantVillage"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# -----------------------------
# Load model
# -----------------------------
model, class_names = load_model_and_classes()

# -----------------------------
# Validation Generator
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# -----------------------------
# Predict
# -----------------------------
predictions = model.predict(val_gen)
y_pred = np.argmax(predictions, axis=1)
y_true = val_gen.classes

# -----------------------------
# Metrics
# -----------------------------
print("\nClassification Report:\n")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=[class_names[i] for i in range(len(class_names))]
    )
)

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))
