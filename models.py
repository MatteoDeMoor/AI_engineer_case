import os
from transformers import pipeline

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

# This dict will be filled in app_api.py during startup
pipelines = {}
