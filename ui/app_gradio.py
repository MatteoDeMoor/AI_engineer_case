# Standalone Gradio app for IMDB review sentiment analysis
# Imports pretrained pipelines from models.py and exposes a simple UI

import gradio as gr
from transformers import pipeline

from models import pipelines  # dict of pretrained pipelines

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

# Classification function for Gradio
def classify(text: str, model_name: str):
    # Run the selected pipeline on input text and map raw label to binary sentiment.
    pipe = pipeline_map[model_name]
    # Truncate to max length expected by pipelines
    truncated = text[:512]
    # Single inference call returns list of dicts
    result = pipe(truncated, truncation=True, max_length=512)[0]
    raw_label = result["label"]
    score = float(result["score"])
    label = label_mappers[model_name](raw_label)
    return label, score

# Build Gradio interface
iface = gr.Interface(
    fn=classify,
    inputs=[
        gr.Textbox(lines=5, placeholder="Type a movie review…", label="Review"),
        gr.Radio(choices=list(pipeline_map.keys()), label="Model", value="DistilBERT-SST2"),
    ],
    outputs=[
        gr.Label(num_top_classes=1, label="Predicted Sentiment"),
        gr.Number(label="Confidence"),
    ],
    title="IMDB Review Sentiment Demo",
    description="Pick a model and see Positive vs. Negative"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
