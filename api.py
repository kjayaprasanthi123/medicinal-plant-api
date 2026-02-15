from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from plant_app.ml.predictor import PlantPredictor
from plant_app.medicinal_data import MedicinalData

# Initialize FastAPI app
app = FastAPI(title="Medicinal Plant Identification API")

# -----------------------------
# CORS Configuration
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace "*" with frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load ML Model
# -----------------------------
try:
    predictor = PlantPredictor("final_model.keras")
except Exception as e:
    print(f"Error loading model: {e}")
    predictor = None


# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
def home():
    return {"message": "Medicinal Plant API is running"}


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
async def predict_plant(file: UploadFile = File(...)):
    
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")

    try:
        image_bytes = await file.read()

        plant_name, confidence = predictor.predict(image_bytes)
        medicinal_uses = MedicinalData.get_uses(plant_name)

        return {
            "plant_name": plant_name,
            "confidence": float(confidence),
            "medicinal_uses": medicinal_uses
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
