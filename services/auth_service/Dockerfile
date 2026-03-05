FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

ENV PYTHONUNBUFFERED=1

RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN mkdir -p /data && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /data

USER appuser
EXPOSE 8004

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
