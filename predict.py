import tensorflow as tf
import numpy as np
import json
import sys
from tensorflow.keras.preprocessing import image

# -----------------------------
# 1️⃣ Paths
# -----------------------------
MODEL_PATH = "final_model.keras"
CLASS_INDEX_PATH = "class_indices.json"
IMG_SIZE = (224, 224)

# -----------------------------
# 2️⃣ Load model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# -----------------------------
# 3️⃣ Load class names (FIXED)
# -----------------------------
with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)

# 🔥 VERY IMPORTANT FIX
index_to_class = {v: k for k, v in class_indices.items()}

print(f"📂 Total classes: {len(index_to_class)}")

# -----------------------------
# 4️⃣ Image preprocessing
# -----------------------------
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0   # SAME as training
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# 5️⃣ Prediction function
# -----------------------------
def predict_plant(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)

    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index] * 100
    plant_name = index_to_class[class_index]

    print("\n🌿 PREDICTION RESULT")
    print(f"Plant Name : {plant_name}")
    print(f"Confidence : {confidence:.2f}%")

# -----------------------------
# 6️⃣ Run from command line
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path_to_leaf_image")
        sys.exit(1)

    image_path = sys.argv[1]
    predict_plant(image_path)
