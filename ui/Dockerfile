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

# Copy the Gradio UI code
COPY ui/app_gradio.py .

# Expose port for the Gradio UI
EXPOSE 7860

# Start Gradio app
CMD ["python", "app_gradio.py"]