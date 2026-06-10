from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords
import warnings

warnings.filterwarnings('ignore')
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

app = FastAPI(
    title="Financial SEC Filing Classifier API",
    description="E-Cell AI & Automation Task: Predicts financial risk from 10-K text.",
    version="1.0.0"
)

try:
    print("Loading models...")
    model = joblib.load('../models/best_model.joblib')
    vectorizer = joblib.load('../models/tfidf_vectorizer.joblib')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}. Make sure they are in the 'models' folder.")

class FilingInput(BaseModel):
    text: str
def clean_text(text: str) -> str:
    text = text[:15000]
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()
    words = text.split()
    cleaned_words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(cleaned_words)

risk_mapping = {0: "High Risk / Negative", 1: "Medium Risk / Neutral", 2: "Low Risk / Positive"}
@app.post("/predict")
async def predict_risk(filing: FilingInput):
    if not filing.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    
    try:
        cleaned_input = clean_text(filing.text)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = int(model.predict(vectorized_input)[0])
        return {
            "status": "success",
            "prediction_class": prediction,
            "risk_level": risk_mapping.get(prediction, "Unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "API is running. Go to /docs to test the predict endpoint."}
