from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

app = FastAPI(title="Sentiment Classifier API", version="1.0")

# Load model on startup
MODEL_NAME = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Request schema
class ReviewRequest(BaseModel):
    text: str

# Clean text
def clean(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

@app.get("/")
def root():
    return {"message": "Sentiment Classifier API is running"}

@app.post("/predict")
def predict(request: ReviewRequest):
    cleaned = clean(request.text)
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return {
        "text": request.text,
        "sentiment": "positive" if pred == 1 else "negative",
        "confidence": round(confidence * 100, 2),
        "model": "DistilBERT"
    }