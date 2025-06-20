import os
from transformers import pipeline
from functools import lru_cache

# prevent tokenisers from spawning too many threads
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# constants
DEVICE     = -1
BATCH_SIZE = 64

# Functions to create pipelines
pipeline_constructors = {
    "distilbert": lambda: pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=DEVICE,
        batch_size=BATCH_SIZE
    ),
    "roberta_large": lambda: pipeline(
        "sentiment-analysis",
        model="siebert/sentiment-roberta-large-english",
        device=DEVICE,
        batch_size=BATCH_SIZE
    ),
    "multilingual": lambda: pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=DEVICE,
        batch_size=BATCH_SIZE
    ),
    "textattack_bert": lambda: pipeline(
        "sentiment-analysis",
        model="textattack/bert-base-uncased-SST-2",
        device=DEVICE,
        batch_size=BATCH_SIZE
    ),
    "textattack_roberta": lambda: pipeline(
        "sentiment-analysis",
        model="textattack/roberta-base-SST-2",
        device=DEVICE,
        batch_size=BATCH_SIZE
    ),
    "twitter_roberta": lambda: pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        device=DEVICE,
        batch_size=BATCH_SIZE
    ),
}

# Cache and retrieve pipelines so each model is loaded only once
@lru_cache(maxsize=None)
def get_pipeline(model_key: str):
    """
    Return a sentiment-analysis pipeline for the given model key.
    Raises a ValueError if the key is unknown.
    """
    try:
        constructor = pipeline_constructors[model_key]
    except KeyError:
        raise ValueError(f"Unknown model '{model_key}'. Available models: {list(pipeline_constructors.keys())}")
    return constructor()


def load_all_pipelines():
    """
    Pre-load all pipelines into the cache. Call this once at startup if desired.
    """
    for key in pipeline_constructors:
        get_pipeline(key)
