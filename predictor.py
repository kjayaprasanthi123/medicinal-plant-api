import tensorflow as tf
import numpy as np
from PIL import Image
import io

class PlantPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

        self.class_names = [
            "Aloevera","Amla","Amruthaballi","Arali","Ashoka","Astma_weed",
            "Badipala","Balloon_Vine","Bamboo","Beans","Betel","Bhrami",
            "Bringaraja","Camphor","Caricature","Castor","Catharanthus",
            "Chakte","Chilly","Citron lime (herelikai)","Coffee",
            "Common rue(naagdalli)","Coriender","Curry","Doddpathre",
            "Drumstick","Ekka","Eucalyptus","Ganigale","Ganike","Gasagase",
            "Ginger","Globe Amarnath","Guava","Henna","Hibiscus","Honge",
            "Insulin","Jackfruit","Jasmine","Kamakastur","Kambajala",
            "Kasambruga","Kepala","Kohlrabi","Lantana","Lemon","Lemongrass",
            "Malabar_Nut","Malabar_Spinach","Mango","Marigold","Mint","Neem",
            "Nelavembu","Nerale","Nooni","Onion","Padri","Palak(Spinach)",
            "Papaya","Parijatha","Pea","Pepper","Pomoegranate","Pumpkin",
            "Raddish","Rose","Sampige","Sapota","Seethaashoka","Seethapala",
            "Spinach1","Tamarind","Taro","Tecoma","Thumbe","Tomato","Tulsi",
            "Turmeric"
        ]

    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        preds = self.model.predict(image)
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds) * 100)

        return self.class_names[idx], confidence
