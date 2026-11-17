FROM python:3.12.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/artifacts/final_model.joblib

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --disable-pip-version-check --timeout=120 --retries=5 -r requirements.txt \
    && groupadd --system app && useradd --system --gid app --home-dir /app app

COPY config.toml ./config.toml
COPY src ./src
COPY artifacts/final_model.joblib ./artifacts/final_model.joblib
ENV PYTHONPATH=/app/src
USER app:app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "wb_sentiment.api:app", "--host", "0.0.0.0", "--port", "8000"]
