# Gradio UI for IMDB movie review sentiment analysis
# Loads all sentiment-analysis pipelines at startup via a cached factory

import nest_asyncio
nest_asyncio.apply()

import gradio as gr
from models import get_pipeline, load_all_pipelines

# Preload all pipelines into the HF cache for fast inference
print("⏳ Preloading all sentiment-analysis pipelines into cache...", flush=True)
load_all_pipelines()

# Map human-readable names to internal model keys for get_pipeline
pipeline_map = {
    "DistilBERT-SST2":        "distilbert",
    "RoBERTa-large SST2":     "roberta_large",
    "nlptown Multilingual":   "multilingual",
    "TextAttack BERT-SST2":   "textattack_bert",
    "TextAttack RoBERTa-SST2":"textattack_roberta",
    "Twitter RoBERTa":        "twitter_roberta",
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
    """
    Run sentiment analysis using the specified model and return
    a human-readable label and confidence score.
    """
    # Fetch the pipeline via the cached factory
    model_key = pipeline_map[model_name]
    pipe = get_pipeline(model_key)

    # Perform inference with truncation to max length
    result = pipe(text[:512], truncation=True, max_length=512)[0]
    raw_label = result["label"]
    score = float(result["score"])

    # Convert raw label into Positive/Negative
    label = label_mappers[model_name](raw_label)
    return label, round(score, 4)

# Build the Gradio interface
face = gr.Interface(
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

# Launch the Gradio app on all interfaces
print("🚀 Gradio UI is now available at http://localhost:7860", flush=True)
face.launch(
    server_name="0.0.0.0",
    server_port=7860,
    quiet=False,
    prevent_thread_lock=False
)
