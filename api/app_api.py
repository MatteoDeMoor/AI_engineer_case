# FastAPI application for IMDB review sentiment prediction
# Exposes a POST endpoint `/predict` that takes a review text and a model name,
# then returns the predicted sentiment label and confidence score.

import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from models import get_pipeline, load_all_pipelines

# We will populate this mapping once models are loaded in the startup event
pipeline_map = {}

# Define how to convert each model's raw label into "Positive"/"Negative"
label_mappers = {
    "DistilBERT-SST2":        lambda raw: raw.capitalize(),
    "RoBERTa-large SST2":     lambda raw: raw.capitalize(),
    "nlptown Multilingual":   lambda raw: "Positive" if int(raw.split()[0]) >= 3 else "Negative",
    "TextAttack BERT-SST2":   lambda raw: "Positive" if raw == "LABEL_1" else "Negative",
    "TextAttack RoBERTa-SST2":lambda raw: "Positive" if raw == "LABEL_1" else "Negative",
    "Twitter RoBERTa":        lambda raw: "Positive" if raw == "LABEL_2" else "Negative",
}

# Initialize FastAPI app
app = FastAPI(title="IMDB Sentiment API")

# Startup event handler: preload all pipelines into the HF cache.
@app.on_event("startup")
def load_models():
    print("⏳ Preloading all sentiment-analysis pipelines...", flush=True)
    # Load all models into cache
    load_all_pipelines()

    # Build the mapping from display names to internal model keys
    global pipeline_map
    pipeline_map = {
        "DistilBERT-SST2":        "distilbert",
        "RoBERTa-large SST2":     "roberta_large",
        "nlptown Multilingual":   "multilingual",
        "TextAttack BERT-SST2":   "textattack_bert",
        "TextAttack RoBERTa-SST2":"textattack_roberta",
        "Twitter RoBERTa":        "twitter_roberta",
    }

    print("🚀 API is now available at http://localhost:8000/docs", flush=True)

# Pydantic models for request and response bodies
class ReviewRequest(BaseModel):
    text: str
    model: str = "DistilBERT-SST2"

class PredictionResponse(BaseModel):
    model: str
    label: str
    score: float

@app.post("/predict", response_model=PredictionResponse)
def predict(req: ReviewRequest):
    # Check if requested model is available
    if req.model not in pipeline_map:
        return {
            "model": req.model,
            "label": "",
            "score": 0.0,
            "error": f"Unknown model '{req.model}'. Choose from {list(pipeline_map.keys())}"
        }

    # Retrieve pipeline
    model_key = pipeline_map[req.model]
    pipe = get_pipeline(model_key)

    # Perform the inference (truncate to max length)
    result = pipe(req.text[:512], truncation=True, max_length=512)[0]
    raw_label = result["label"]
    score = float(result["score"])

    # Map raw label to human-readable
    label = label_mappers[req.model](raw_label)

    return PredictionResponse(model=req.model, label=label, score=round(score, 4))

if __name__ == "__main__":
    uvicorn.run("app_api:app", host="0.0.0.0", port=8000, reload=True)
