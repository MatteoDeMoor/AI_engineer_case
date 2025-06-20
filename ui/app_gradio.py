# Gradio UI for IMDB movie review sentiment analysis
# Loads all sentiment-analysis pipelines at startup and exposes a simple interface

import nest_asyncio
nest_asyncio.apply()

import gradio as gr
from models import pipeline_constructors, pipelines  # constructors + initially empty pipelines dict

# Load all pipelines with detailed progress logging
print("⏳ Beginning to load model pipelines for Gradio UI...", flush=True)
for key, constructor in pipeline_constructors.items():
    print(f"⏳ Loading {key}...", flush=True)
    pipelines[key] = constructor()
    print(f"✅ Successfully loaded {key}", flush=True)

# Map human-readable names to pipeline instances
pipeline_map = {
    "DistilBERT-SST2":        pipelines["distilbert"],
    "RoBERTa-large SST2":     pipelines["roberta_large"],
    "nlptown Multilingual":   pipelines["multilingual"],
    "TextAttack BERT-SST2":   pipelines["textattack_bert"],
    "TextAttack RoBERTa-SST2":pipelines["textattack_roberta"],
    "Twitter RoBERTa":        pipelines["twitter_roberta"],
}

# Unified label mappers for each model
label_mappers = {
    "DistilBERT-SST2":        lambda raw: raw.capitalize(),
    "RoBERTa-large SST2":     lambda raw: raw.capitalize(),
    "nlptown Multilingual":   lambda raw: "Positive" if int(raw.split()[0]) >= 3 else "Negative",
    "TextAttack BERT-SST2":   lambda raw: "Positive" if raw == "LABEL_1" else "Negative",
    "TextAttack RoBERTa-SST2":lambda raw: "Positive" if raw == "LABEL_1" else "Negative",
    "Twitter RoBERTa":        lambda raw: "Positive" if raw == "LABEL_2" else "Negative",
}

def classify(text: str, model_name: str):
    # Run sentiment analysis on a single review using the specified model
    # Returns a human-readable label and confidence score

    pipe = pipeline_map[model_name]
    # Perform inference with truncation
    result = pipe(text[:512], truncation=True, max_length=512)[0]
    raw_label = result["label"]
    score = float(result["score"])
    label = label_mappers[model_name](raw_label)
    return label, score

# Build the Gradio interface
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
    description="Choose a model and see whether the review is Positive or Negative.",
)

# Launch the Gradio app, binding to all interfaces
print("🚀 Gradio UI is now available at  http://localhost:7860", flush=True)
iface.launch(
    server_name="0.0.0.0",
    server_port=7860,
    quiet=False,
    prevent_thread_lock=False
)
