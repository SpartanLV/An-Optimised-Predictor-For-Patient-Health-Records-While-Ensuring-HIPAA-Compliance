FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN groupadd -r appuser && useradd -r -g appuser appuser

ENV STORAGE_DIR=/data/documents
RUN mkdir -p ${STORAGE_DIR} && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /data

USER appuser

EXPOSE 8002
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
