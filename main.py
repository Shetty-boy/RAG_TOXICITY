from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from contextlib import asynccontextmanager
import json
import os
from rag_pipeline import RAGPipeline

# Global variables to hold our model, tokenizer, and config
model = None
tokenizer = None
label_names = []
label_thresholds = {}
rag_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, label_names, label_thresholds, rag_pipeline
    print("Starting up: Loading model and config into memory...")
    model_path = "./best_multilabel_model"
    
    # Load label config
    config_path = os.path.join(model_path, "label_config.json")
    if not os.path.exists(config_path):
        raise RuntimeError(f"Config file not found at {config_path}. Did you run train_multilabel.py?")
        
    with open(config_path, "r") as f:
        config = json.load(f)
        label_names = config["labels"]
        label_thresholds = config["thresholds"]
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Put the model in evaluation mode
    model.eval() 
    
    # Initialize RAG Pipeline
    rag_pipeline = RAGPipeline(db_path="chroma_db", bm25_dir="bm25_indices")
    
    print("Model and config loaded successfully!")
    yield
    print("Shutting down: Clearing memory...")

app = FastAPI(lifespan=lifespan, title="Toxicity Detection API (Multi-Label)")

class PredictionRequest(BaseModel):
    text: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "distilbert-toxicity-multilabel"}

@app.post("/predict")
def predict(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    # 1. Tokenize
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=128)
    
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    
    # 2. Run Inference
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 3. Apply Sigmoid for independent multi-label probabilities
    probabilities = torch.sigmoid(outputs.logits)[0].tolist()
    
    # 4. Map probabilities to category names
    category_scores = {label: round(prob, 4) for label, prob in zip(label_names, probabilities)}
    
    # 5. Routing Logic (Confidence Gate)
    predicted_labels = [cat for cat in label_names if category_scores[cat] > label_thresholds[cat]]
    max_score = max(category_scores.values())
    
    if any(category_scores[cat] > label_thresholds[cat] for cat in label_names):
        route = "TOXIC"
    elif max_score >= 0.30:
        route = "AMBIGUOUS"
    else:
        route = "CLEAN"
        
    # Find dominant category
    dominant_category = max(category_scores, key=category_scores.get)
    
    return {
        "input_text": request.text,
        "route": route,
        "dominant_category": dominant_category,
        "confidence": max_score,
        "is_toxic": route == "TOXIC",
        "predicted_labels": predicted_labels,
        "category_scores": category_scores
    }

@app.post("/explain")
def explain(request: PredictionRequest):
    # First, run the standard prediction
    pred_result = predict(request)
    
    route = pred_result["route"]
    dominant_category = pred_result["dominant_category"]
    
    # Only invoke RAG pipeline if content is AMBIGUOUS or TOXIC
    if route == "CLEAN":
        pred_result["explanation"] = "Content is deemed safe. No policy matches."
        pred_result["policy_matches"] = []
        pred_result["suggested_action"] = "None"
        pred_result["insufficient_evidence"] = False
        return pred_result
        
    # Execute RAG explanation
    explanation_result = rag_pipeline.explain(request.text, dominant_category)
    
    # Merge results
    pred_result.update(explanation_result)
    
    return pred_result