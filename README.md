# IMDB Sentiment Benchmark (AI Engineer Technical Case)

This repository contains a reproducible pipeline for benchmarking six pre-trained sentiment analysis models on the IMDB movie reviews dataset.

## 1. Clone the repository

```bash
git clone https://github.com/matteodemoor/AI_engineer_case
cd AI_engineer_case
```

## 2. Set up a virtual environment

Create and activate a Python virtual environment (venv) in the project folder:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Upgrade `pip` and install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Notebook walkthrough

Open the Jupyter notebook to explore data loading, model pipelines, and evaluation metrics:
```bash
jupyter notebook emotion_evaluator.ipynb
```
Make sure to select the `.venv` kernel when prompted.

## 4. Sentiment analysis pipelines

We evaluate the following six models via Hugging Face Transformers:

| Key                  | Model ID                                                        | Description                                           |
|----------------------|-----------------------------------------------------------------|-------------------------------------------------------|
| `distilbert`         | `distilbert-base-uncased-finetuned-sst-2-english`               | DistilBERT fine-tuned on SST-2 for speed & accuracy   |
| `roberta_large`      | `siebert/sentiment-roberta-large-english`                       | Large RoBERTa fine-tuned on SST-2 for top accuracy    |
| `multilingual`       | `nlptown/bert-base-multilingual-uncased-sentiment`             | Multilingual BERT for 5-star ratings                  |
| `textattack_bert`    | `textattack/bert-base-uncased-SST-2`                           | BERT (TextAttack fork) on SST-2                       |
| `textattack_roberta` | `textattack/roberta-base-SST-2`                                | RoBERTa (TextAttack fork) on SST-2                    |
| `twitter_roberta`    | `cardiffnlp/twitter-roberta-base-sentiment`                     | Twitter-tuned RoBERTa with neutral class              |

## 5. Benchmark results

Below are the evaluation metrics on 100 IMDB reviews (58 negative, 42 positive):

| Model                         | Accuracy | Precision (neg) | Recall (neg) | Precision (pos) | Recall (pos) |
|-------------------------------|----------|-----------------|--------------|-----------------|--------------|
| DistilBERT-SST2               | 0.86     | 0.87            | 0.90         | 0.85            | 0.81         |
| Siebert RoBERTa-large         | 0.93     | 0.96            | 0.91         | 0.89            | 0.95         |
| nlptown 1-5 stars             | 0.81     | 0.91            | 0.74         | 0.72            | 0.90         |
| TextAttack BERT-SST2          | 0.87     | 0.88            | 0.90         | 0.85            | 0.83         |
| TextAttack RoBERTa-SST2       | 0.88     | 0.86            | 0.95         | 0.92            | 0.79         |
| CardiffNLP Twitter RoBERTa    | 0.83     | 0.81            | 0.93         | 0.88            | 0.69         |

*Macro- and weighted averages are also reported in the notebook and `result/benchmark_report.txt`.*

## 6. Interactive demo with Gradio

Launch a local Gradio app to test any review against your six models:

```bash
python -c "import emotion_evaluator; emotion_evaluator.launch_gradio()"
```

Alternatively, run inside the notebook:

```python
# In a cell
iface.launch()
```

## 7. Predict API with FastAPI

Start the FastAPI server for programmatic predictions:

```bash
uvicorn api:app --reload
```

Browse to `http://127.0.0.1:8000/docs` for an interactive Swagger UI. Example request body:

```json
{
  "text": "I loved this movie!",
  "model": "RoBERTa-large SST2"
}
```

## 8. Version control & workflow

- **Branching model**  
  ```bash
  # make changes...
  git status
  git add .
  git commit -m "message"
  git push -u origin
  ```

- **Ignore files**  
  Add to `.gitignore`:  
  ```
  result/
  ```
