import os
from transformers import pipeline

# Prevent excessive threading overhead in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants
DEVICE = -1  # CPU-only, change to 0 for GPU
BATCH_SIZE = 64


def load_pipelines():
    # Load and return a dictionary of sentiment-analysis pipelines
    # Keys are model identifiers, values are HuggingFace pipeline objects
    return {
        "distilbert": pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=DEVICE,
            batch_size=BATCH_SIZE
        ),
        "roberta_large": pipeline(
            "sentiment-analysis",
            model="siebert/sentiment-roberta-large-english",
            device=DEVICE,
            batch_size=BATCH_SIZE
        ),
        "multilingual": pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=DEVICE,
            batch_size=BATCH_SIZE
        ),
        "textattack_bert": pipeline(
            "sentiment-analysis",
            model="textattack/bert-base-uncased-SST-2",
            device=DEVICE,
            batch_size=BATCH_SIZE
        ),
        "textattack_roberta": pipeline(
            "sentiment-analysis",
            model="textattack/roberta-base-SST-2",
            device=DEVICE,
            batch_size=BATCH_SIZE
        ),
        "twitter_roberta": pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            device=DEVICE,
            batch_size=BATCH_SIZE
        ),
    }

# Load pipelines once at import time
df = None
pipelines = load_pipelines()
