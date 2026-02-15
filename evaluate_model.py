import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# 1️⃣ Paths & parameters
# -----------------------------
MODEL_PATH = "final_model.keras"
DATASET_DIR = "Medicinal_Leaf_Dataset"   # same dataset
CLASS_INDEX_PATH = "class_indices.json"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# -----------------------------
# 2️⃣ Load model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# -----------------------------
# 3️⃣ Load saved class indices
# -----------------------------
with open(CLASS_INDEX_PATH, "r") as f:
    train_class_indices = json.load(f)

# Reverse mapping (index → class)
index_to_class = {v: k for k, v in train_class_indices.items()}
NUM_CLASSES = len(index_to_class)
print(f"📂 Total classes: {NUM_CLASSES}")

# -----------------------------
# 4️⃣ Test data generator
# -----------------------------
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ✅ Safety check
assert test_generator.class_indices == train_class_indices, \
    "❌ Class index mismatch between training and evaluation!"

print("✅ Class indices match perfectly")

# -----------------------------
# 5️⃣ Predictions
# -----------------------------
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# -----------------------------
# 6️⃣ Metrics
# -----------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

# -----------------------------
# 7️⃣ Results
# -----------------------------
print("\n📊 MODEL EVALUATION RESULTS")
print(f"Accuracy   : {accuracy * 100:.2f}%")
print(f"Precision  : {precision * 100:.2f}%")
print(f"Recall     : {recall * 100:.2f}%")
print(f"F1-score   : {f1 * 100:.2f}%")
