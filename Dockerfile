FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source
COPY oceanus/ ./oceanus/
COPY dashboard/ ./dashboard/
COPY server/ ./server/
COPY data/ ./data/
COPY models.py .
COPY client.py .
COPY openenv.yaml .

# HuggingFace Spaces runs on port 7860
EXPOSE 7860

# Start the 3D dashboard (FastAPI + WebSocket) — this is the primary demo interface
# The OpenEnv-compliant server is also available via server/app.py
CMD ["uvicorn", "dashboard.server:app", "--host", "0.0.0.0", "--port", "7860"]
