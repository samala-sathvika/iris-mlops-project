from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "iris_model.joblib"
if not MODEL_PATH.exists():
    raise RuntimeError("Model file not found. Run `python train.py` first.")

saved = joblib.load(MODEL_PATH)
model = saved["model"]
feature_names = saved["feature_names"]
class_names = saved["class_names"]

app = FastAPI(title="Iris MLOps API", version="1.0.0",
              description="Predict Iris flower species from four measurements.")

class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0, description="Sepal length in cm")
    sepal_width: float = Field(gt=0, description="Sepal width in cm")
    petal_length: float = Field(gt=0, description="Petal length in cm")
    petal_width: float = Field(gt=0, description="Petal width in cm")

@app.get("/")
def home():
    return {"message": "Iris ML API is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(data: IrisInput):
    values = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    try:
        prediction = int(model.predict(values)[0])
        probabilities = model.predict_proba(values)[0]
        return {
            "predicted_class": class_names[prediction],
            "class_id": prediction,
            "confidence": round(float(max(probabilities)), 4),
            "probabilities": {
                class_names[i]: round(float(probabilities[i]), 4)
                for i in range(len(class_names))
            }
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
