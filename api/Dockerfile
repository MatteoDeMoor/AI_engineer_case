# api/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy the shared model loader and the API code
# Paths are relative to the build context (project root)
COPY models.py .
COPY api/app_api.py .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "8000"]
