import os
import pandas as pd
import torch
from pathlib import Path
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# Prevent excessive threading overhead in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants
DEVICE = -1  # CPU-only
BATCH_SIZE = 64
MAX_LENGTH = 512
INPUT_PATH = Path('data') / 'IMDB-movie-reviews.csv'
OUTPUT_CSV = Path('result') / 'sentiment_benchmarks.csv'
BENCHMARK_REPORT = Path('result') / 'benchmark_report.txt'

# Load data
df = pd.read_csv(INPUT_PATH, sep=';', encoding='ISO-8859-1')

# Create sentiment pipelines
pipelines = {
    'distilbert': pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=DEVICE,
        batch_size=BATCH_SIZE
    ),
    'roberta_large': pipeline(
        "sentiment-analysis",
        model="siebert/sentiment-roberta-large-english",
        device=DEVICE,
        batch_size=BATCH_SIZE
    ),
    'multilingual': pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=DEVICE,
        batch_size=BATCH_SIZE
    ),
    'textattack_bert': pipeline(
        "sentiment-analysis",
        model="textattack/bert-base-uncased-SST-2",
        device=DEVICE,
        batch_size=BATCH_SIZE
    ),
    'textattack_roberta': pipeline(
        "sentiment-analysis",
        model="textattack/roberta-base-SST-2",
        device=DEVICE,
        batch_size=BATCH_SIZE
    ),
    'twitter_roberta': pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        device=DEVICE,
        batch_size=BATCH_SIZE
    )
}

# Prepare texts
texts = df['review'].astype(str).tolist()

# Run inference with progress bars
preds = {}
for name, pipe in pipelines.items():
    preds[name] = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE),
                  desc=f"Infereren met {name}",
                  unit="batch"):
        batch = texts[i : i + BATCH_SIZE]
        outputs = pipe(batch, truncation=True, max_length=MAX_LENGTH)
        preds[name].extend(outputs)

# Store predictions
for name, output in preds.items():
    labels, scores = zip(*[(o['label'], o['score']) for o in output])
    df[f'label_{name}'] = labels
    df[f'score_{name}'] = scores

# Save results
df.to_csv(OUTPUT_CSV, index=False)

# Helper functions for binary mapping
def star2bin(label):
    stars = int(label.split()[0])
    return 'negative' if stars <= 2 else 'positive'

def ta_label2bin(label):
    return 'positive' if label == 'LABEL_1' else 'negative'

def tw_roberta3bin(label):
    return 'positive' if label == 'LABEL_2' or label.lower() == 'positive' else 'negative'

# Create binary predictions
df['pred_distilbert'] = df['label_distilbert'].str.lower()
df['pred_roberta_large'] = df['label_roberta_large'].str.lower()
df['pred_multilingual'] = df['label_multilingual'].apply(star2bin)
df['pred_textattack_bert'] = df['label_textattack_bert'].apply(ta_label2bin)
df['pred_textattack_roberta'] = df['label_textattack_roberta'].apply(ta_label2bin)
df['pred_twitter_roberta'] = df['label_twitter_roberta'].apply(tw_roberta3bin)

# True labels
y_true = df['sentiment']

model_names = {
    'distilbert': "DistilBERT-SST2",
    'roberta_large': "Siebert RoBERTa-large",
    'multilingual': "nlptown 1-5 stars",
    'textattack_bert': "TextAttack BERT-SST2",
    'textattack_roberta': "TextAttack RoBERTa-SST2",
    'twitter_roberta': "CardiffNLP Twitter RoBERTa",
}

# Write benchmark report
with open(BENCHMARK_REPORT, 'w', encoding='utf-8') as f:
    for key, name in model_names.items():
        y_pred = df[f'pred_{key}']
        f.write(f"{name}\n")
        f.write(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n\n")
        f.write(classification_report(y_true, y_pred, zero_division=0))
        f.write("\n" + "-"*50 + "\n\n")
