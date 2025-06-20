FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Pre-download all HuggingFace models into the image layer cache
COPY models.py .
RUN python - <<EOF
from models import load_all_pipelines
load_all_pipelines()
EOF

# Copy the FastAPI app code
COPY api/app_api.py .

# Expose port for the API
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "8000"]
