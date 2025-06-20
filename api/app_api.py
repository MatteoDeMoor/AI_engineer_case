# FastAPI application for IMDB review sentiment prediction
# Exposes a POST endpoint `/predict` that takes a review text and a model name,
# then returns the predicted sentiment label and confidence score.

import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from models import pipeline_constructors, pipelines  # constructors: callables to create pipelines; pipelines: initially empty

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

@app.on_event("startup")
def load_models():
    """
    Startup event handler: load each sentiment-analysis pipeline once,
    printing progress, and then build the human-readable pipeline map.
    """
    print("⏳ Loading model pipelines...", flush=True)
    for key, constructor in pipeline_constructors.items():
        print(f"⏳ Loading {key}...", end="", flush=True)
        pipelines[key] = constructor()   # instantiate the pipeline
        print(" ✅", flush=True)

    # After all pipelines are ready, build the mapping from display names to pipeline objects
    global pipeline_map
    pipeline_map = {
        "DistilBERT-SST2":        pipelines["distilbert"],
        "RoBERTa-large SST2":     pipelines["roberta_large"],
        "nlptown Multilingual":   pipelines["multilingual"],
        "TextAttack BERT-SST2":   pipelines["textattack_bert"],
        "TextAttack RoBERTa-SST2":pipelines["textattack_roberta"],
        "Twitter RoBERTa":        pipelines["twitter_roberta"],
    }

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
    """
    Predict endpoint:
    - Validates the requested model name.
    - Runs one inference call with truncation.
    - Applies the label mapper to produce "Positive"/"Negative".
    """
    if req.model not in pipeline_map:
        # Return an error structure if the model name is unknown
        return {
            "model": req.model,
            "label": "",
            "score": 0.0,
            "error": f"Unknown model '{req.model}'. Choose from {list(pipeline_map.keys())}"
        }

    # Perform the inference
    pipe = pipeline_map[req.model]
    result = pipe(req.text[:512], truncation=True, max_length=512)[0]

    raw_label = result["label"]
    score = float(result["score"])
    label = label_mappers[req.model](raw_label)

    # Return the structured response
    return PredictionResponse(model=req.model, label=label, score=round(score, 4))

if __name__ == "__main__":
    # Run the app with Uvicorn for local development with auto-reload
    uvicorn.run("app_api:app", host="0.0.0.0", port=8000, reload=True)
