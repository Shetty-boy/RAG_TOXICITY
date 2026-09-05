from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from contextlib import asynccontextmanager

# Global variables to hold our model and tokenizer
model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print("Starting up: Loading model into memory...")
    model_path = "./best_toxicity_model"
    
    # Load them explicitly instead of using the black-box pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Put the model in evaluation mode (saves memory, disables dropout)
    model.eval() 
    print("Model loaded successfully!")
    yield
    print("Shutting down: Clearing memory...")

app = FastAPI(lifespan=lifespan, title="Toxicity Detection API")

class PredictionRequest(BaseModel):
    text: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "distilbert-toxicity-custom-inference"}

@app.post("/predict")
def predict(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    # 1. Tokenize the text into PyTorch tensors
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=128)
    
    # 2. Mentor Hack: Manually strip out the problematic 'token_type_ids' if they exist!
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    
    # 3. Run Inference without tracking gradients (saves massive memory)
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 4. Convert raw math logits into human-readable percentages (0.0 to 1.0)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    toxic_prob = probabilities[0][1].item()
    
    return {
        "input_text": request.text,
        "toxic_probability": round(toxic_prob, 4),
        "is_toxic": bool(toxic_prob >= 0.5)
    }