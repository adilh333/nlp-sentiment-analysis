FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models
COPY data/sample ./data/sample

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "fos_triage.api:app", "--host", "0.0.0.0", "--port", "8000"]
