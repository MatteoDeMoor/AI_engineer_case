# NLP Sentiment Analysis

This repository contains a reproducible pipeline for benchmarking six pre-trained sentiment analysis models on the IMDB movie reviews dataset, plus an interactive Gradio demo and a FastAPI prediction endpoint.

---

## 1. Clone the repository

```bash
git clone https://github.com/MatteoDeMoor/NLP_Sentiment_Analysis
cd NLP_Sentiment_Analysis
```

---

## 2. Set up a Python virtual environment

Create and activate a new venv in your project folder:

```bash
python -m venv venv
```

```powershell
.\venv\Scripts\Activate.ps1
```

---

## 3. Install & lock your dependencies

1. Upgrade pip and install **pip-tools**:  
   ```bash
   pip install --upgrade pip
   pip install pip-tools
   ```
2. Compile your lockfile (`requirements.txt`) from your top-level specs (`requirements.in`):  
   ```bash
   pip-compile requirements.in --output-file=requirements.txt
   ```
3. Install all dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) **Sync** your venv so that only those packages in `requirements.txt` remain:  
   ```bash
   pip-sync requirements.txt
   ```

---

## 4. Run the sentiment benchmark

This will load the IMDB dataset, run all six models in batches, save a CSV of raw scores, and produce a `benchmark_report.txt` with accuracy & classification metrics:

```bash
python sentiment_benchmark.py
```

---

## 5. Launch the services with Docker Compose

Build and start both the FastAPI backend and the Gradio UI in separate containers:

```bash
docker-compose up --build
```

- **FastAPI** will be available at → `http://localhost:8000/docs`
- **Gradio demo** will be available at → `http://localhost:7860`

Use **Ctrl+C** to stop both services, or in a new shell:

```bash
docker-compose down
```

---

## 6. Version control & workflow

- **Commit & push** your changes:
  ```bash
  git status
  git add .
  git commit -m "Your message"
  git push
  ```
- **.gitignore** should include:
  ```
  result/
  venv/
  ```

---

## Benchmark Results

| Model                         | Accuracy | Precision (neg) | Recall (neg) | F1-score (neg) | Precision (pos) | Recall (pos) | F1-score (pos) |
|-------------------------------|----------|-----------------|--------------|----------------|-----------------|--------------|----------------|
| DistilBERT-SST2               | 0.8900   | 0.86            | 0.97         | 0.91           | 0.94            | 0.79         | 0.86           |
| Siebert RoBERTa-large         | 0.9500   | 0.95            | 0.97         | 0.96           | 0.95            | 0.93         | 0.94           |
| nlptown 1-5 stars             | 0.8400   | 0.94            | 0.78         | 0.85           | 0.75            | 0.93         | 0.83           |
| TextAttack BERT-SST2          | 0.9200   | 0.92            | 0.95         | 0.93           | 0.93            | 0.88         | 0.90           |
| TextAttack RoBERTa-SST2       | 0.9500   | 0.92            | 1.00         | 0.96           | 1.00            | 0.88         | 0.94           |
| CardiffNLP Twitter RoBERTa    | 0.8300   | 0.80            | 0.95         | 0.87           | 0.90            | 0.67         | 0.77           |

---

## License

```text
Copyright (c) 2025 Matteo De Moor
All Rights Reserved.

No permission is granted to copy, modify or redistribute this software without explicit written consent from the author.
```
