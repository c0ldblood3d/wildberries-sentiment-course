"""Create a tiny deterministic model for Docker contract checks in CI."""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
samples = [
    "отличное качество очень нравится удобно", "прекрасная покупка все довольны",
    "ужасное качество сломалось не работает", "плохой товар разочарован",
    "обычный товар ничего особенного", "среднее качество без впечатлений",
]
labels = ["positive", "positive", "negative", "negative", "neutral", "neutral"]
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(random_state=42)),
]).fit(samples, labels)
out = ROOT / "artifacts/final_model.joblib"
out.parent.mkdir(exist_ok=True)
joblib.dump({"model": model, "model_version": "ci-smoke", "model_name": "ci-smoke",
             "classes": ["negative", "neutral", "positive"]}, out)
print(f"Created {out}")
