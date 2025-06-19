# ui/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy the shared model loader and the Gradio code
COPY models.py .
COPY ui/app_gradio.py .

# Expose Gradio port
EXPOSE 7860

# Start Gradio
CMD ["python", "app_gradio.py"]
