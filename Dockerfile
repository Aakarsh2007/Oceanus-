FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY oceanus/ ./oceanus/
COPY dashboard/ ./dashboard/
COPY data/ ./data/

EXPOSE 7860

CMD ["uvicorn", "dashboard.server:app", "--host", "0.0.0.0", "--port", "7860"]
