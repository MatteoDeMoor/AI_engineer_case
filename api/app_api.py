# app_api.py
"""
FastAPI application for IMDB review sentiment prediction.
Exposes a POST endpoint `/predict` that takes a review and model name.
"""
import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from models import pipelines  # import loaded pipelines

# Map human-readable names to pipeline instances
pipeline_map = {
    "DistilBERT-SST2": pipelines["distilbert"],
    "RoBERTa-large SST2": pipelines["roberta_large"],
    "nlptown Multilingual": pipelines["multilingual"],
    "TextAttack BERT-SST2": pipelines["textattack_bert"],
    "TextAttack RoBERTa-SST2": pipelines["textattack_roberta"],
    "Twitter RoBERTa": pipelines["twitter_roberta"],
}

# Unified label mappers for each model
label_mappers = {
    "DistilBERT-SST2": lambda raw: raw.capitalize(),
    "RoBERTa-large SST2": lambda raw: raw.capitalize(),
    "nlptown Multilingual": lambda raw: "Positive" if int(raw.split()[0]) >= 3 else "Negative",
    "TextAttack BERT-SST2": lambda raw: "Positive" if raw == "LABEL_1" else "Negative",
    "TextAttack RoBERTa-SST2": lambda raw: "Positive" if raw == "LABEL_1" else "Negative",
    "Twitter RoBERTa": lambda raw: "Positive" if raw == "LABEL_2" else "Negative",
}

app = FastAPI(title="IMDB Sentiment API")

class ReviewRequest(BaseModel):
    text: str
    model: str = "DistilBERT-SST2"

class PredictionResponse(BaseModel):
    model: str
    label: str
    score: float

@app.post("/predict", response_model=PredictionResponse)
def predict(req: ReviewRequest):
    """
    Predict sentiment for given text using the specified model.
    """
    if req.model not in pipeline_map:
        return {
            "model": req.model,
            "label": "",
            "score": 0.0,
            "error": f"Unknown model '{req.model}'. Choose from {list(pipeline_map.keys())}"  # type: ignore
        }
    pipe = pipeline_map[req.model]
    # Single inference call; pipeline truncates internally
    result = pipe(req.text[:512], truncation=True, max_length=512)[0]
    raw_label = result["label"]
    score = float(result["score"])
    label = label_mappers[req.model](raw_label)
    return PredictionResponse(model=req.model, label=label, score=round(score, 4))

if __name__ == "__main__":
    # For local development: uvicorn can reload on code changes
    uvicorn.run("app_api:app", host="0.0.0.0", port=8000, reload=True)
