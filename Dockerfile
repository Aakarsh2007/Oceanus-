FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY oceanus/ ./oceanus/
COPY dashboard/ ./dashboard/
COPY train/ ./train/

# Expose ports
EXPOSE 8000  
EXPOSE 8501  

# Default: run FastAPI backend
CMD ["uvicorn", "dashboard.api:app", "--host", "0.0.0.0", "--port", "8000"]
