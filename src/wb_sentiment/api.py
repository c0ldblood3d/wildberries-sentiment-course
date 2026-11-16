"""HTTP inference API."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="Wildberries Review Sentiment API", version="1.0.0")


class PredictRequest(BaseModel):
    text: str = Field(description="Основной текст отзыва")
    pros: str = Field(default="", description="Достоинства (необязательно)")
    cons: str = Field(default="", description="Недостатки (необязательно)")

    @field_validator("text")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Поле text не должно быть пустым")
        return value


class PredictResponse(BaseModel):
    sentiment: Literal["negative", "neutral", "positive"]
    model_version: str
    model_id: str


@lru_cache(maxsize=1)
def get_bundle() -> dict:
    path = Path(os.getenv("MODEL_PATH", "artifacts/final_model.joblib"))
    if not path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {path}")
    return joblib.load(path)


@app.get("/health")
def health() -> dict:
    try:
        bundle = get_bundle()
    except (FileNotFoundError, OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail=f"Модель не готова: {error}") from error
    return {"status": "ok", "model_version": bundle["model_version"], "model_ready": True}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        bundle = get_bundle()
    except (FileNotFoundError, OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail=f"Модель не готова: {error}") from error
    text = " ".join(part.strip() for part in (request.text, request.pros, request.cons) if part.strip())
    sentiment = bundle["model"].predict([text])[0]
    return PredictResponse(
        sentiment=sentiment,
        model_version=bundle["model_version"],
        model_id=bundle["model_name"],
    )
