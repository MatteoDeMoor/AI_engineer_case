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

Output files will be written to the `result/` directory.

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

## 7. Version control & workflow

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

## 8. Further development

- **API-only** deployments: edit `api/app_api.py` and build `api/Dockerfile`  
- **UI-only** deployments: edit `ui/app_gradio.py` and build `ui/Dockerfile`  
- **CI/CD**: see `.github/workflows/ci.yml` for linting, testing, lockfile compilation and Docker image builds.  

Happy benchmarking & deploying!
