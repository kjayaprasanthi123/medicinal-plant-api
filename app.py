import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Plant Identification",
    layout="centered"
)

st.title("🌿 Plant Identification and Its Uses")
st.write("Upload a leaf image to predict the plant name and its medicinal uses")

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("final_model.keras")

model = load_model()

# -----------------------------
# Load class indices (CRITICAL FIX)
# -----------------------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# index → class name mapping
index_to_class = {v: k for k, v in class_indices.items()}

IMG_SIZE = (224, 224)

# -----------------------------
# 🌿 Plant Uses Dictionary
# -----------------------------
plant_uses = {
    "Aloevera": "Used for skin care, wound healing, and digestion.",
    "Amla": "Rich in Vitamin C, boosts immunity and digestion.",
    "Amruthaballi": "Improves immunity, helps in fever and diabetes.",
    "Arali": "Used in traditional medicine and as an ornamental plant.",
    "Ashoka": "Used in gynecological treatments and pain relief.",
    "Astma_weed": "Helps in asthma, cough, and respiratory problems.",
    "Badipala": "Anti-inflammatory and used in folk medicine.",
    "Balloon_Vine": "Used for joint pain and inflammation.",
    "Bamboo": "Young shoots improve digestion; used in construction.",
    "Beans": "Rich in protein and fiber.",
    "Betel": "Improves digestion and has antibacterial properties.",
    "Bhrami": "Improves memory and reduces stress.",
    "Bringaraja": "Promotes hair growth and liver health.",
    "Camphor": "Used for cold relief and circulation.",
    "Caricature": "Used for skin diseases in traditional medicine.",
    "Castor": "Castor oil is a laxative and anti-inflammatory.",
    "Catharanthus": "Used in cancer and diabetes treatment.",
    "Chakte": "Used in folk medicine for infections.",
    "Chilly": "Boosts metabolism and immunity.",
    "Citron lime (herelikai)": "Rich in Vitamin C, improves digestion.",
    "Coffee": "Improves alertness and antioxidant source.",
    "Common rue(naagdalli)": "Used for digestion and menstrual issues.",
    "Coriender": "Improves digestion and controls blood sugar.",
    "Curry": "Improves digestion and hair health.",
    "Doddpathre": "Used for cold, cough, and digestion.",
    "Drumstick": "Highly nutritious and boosts immunity.",
    "Ekka": "Used in pain relief and skin treatments.",
    "Eucalyptus": "Used for respiratory and cold relief.",
    "Ganigale": "Used for wound healing.",
    "Ganike": "Traditional medicinal plant.",
    "Gasagase": "Improves digestion and sleep.",
    "Ginger": "Reduces inflammation and aids digestion.",
    "Globe Amarnath": "Used in herbal remedies and decoration.",
    "Guava": "Improves digestion and immunity.",
    "Henna": "Used for hair care and cooling effect.",
    "Hibiscus": "Improves hair growth and heart health.",
    "Honge": "Used for wound healing and skin diseases.",
    "Insulin": "Helps regulate blood sugar levels.",
    "Jackfruit": "Rich in fiber and vitamins.",
    "Jasmine": "Used for stress relief and skincare.",
    "Kamakastur": "Used in traditional medicine.",
    "Kasambruga": "Used for respiratory problems.",
    "Kohlrabi": "Rich in fiber and vitamins.",
    "Lantana": "Used as insect repellent.",
    "Lemon": "Boosts immunity and digestion.",
    "Lemongrass": "Reduces stress and aids digestion.",
    "Malabar_Nut": "Used for asthma and cough.",
    "Malabar_Spinach": "Rich in iron and nutrients.",
    "Mango": "Rich in vitamins A and C.",
    "Marigold": "Used for wound healing.",
    "Mint": "Improves digestion and freshness.",
    "Neem": "Purifies blood and antibacterial.",
    "Nelavembu": "Used for fever and immunity.",
    "Nerale": "Controls diabetes.",
    "Nooni": "Boosts immunity and antioxidants.",
    "Papaya": "Improves digestion.",
    "Pepper": "Improves metabolism.",
    "Pumpkin": "Rich in vitamins and fiber.",
    "Rose": "Used in skincare and stress relief.",
    "Spinach1": "Rich in iron and vitamins.",
    "Tamarind": "Improves digestion.",
    "Tomato": "Rich in antioxidants.",
    "Tulsi": "Boosts immunity and reduces stress.",
    "Turmeric": "Powerful anti-inflammatory and antioxidant."
}

# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=700)

    if st.button("🔍 Predict"):
        with st.spinner("Predicting..."):
            img_array = preprocess_image(image)
            predictions = model.predict(img_array)[0]

            best_index = np.argmax(predictions)
            confidence = predictions[best_index] * 100
            plant_name = index_to_class[best_index]

        if confidence < 50:
            st.warning("⚠️ Low confidence. Please upload a clearer leaf image.")
        else:
            st.success("✅ Prediction Successful")
            st.markdown(f"### 🌱 Plant Name: **{plant_name}**")
            st.markdown(f"### 📊 Confidence: **{confidence:.2f}%**")

            uses = plant_uses.get(
                plant_name,
                "Medicinal uses information not available."
            )

            st.markdown("### 💊 Medicinal Uses")
            st.info(uses)
