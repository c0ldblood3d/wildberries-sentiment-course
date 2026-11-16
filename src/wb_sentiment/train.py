"""Train and compare three reproducible TF-IDF sentiment models."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from wb_sentiment.data import CLASS_ORDER, grouped_split, sha256

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/versions/v2.csv"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
SEED = 42


def build_models() -> dict[str, tuple[Pipeline, dict[str, object]]]:
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)
    word_char = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3,
                                     max_features=120_000, sublinear_tf=True)),
        ]
    )
    configs = {
        "word_logistic_c1": (
            word,
            LogisticRegression(max_iter=1200, class_weight="balanced", C=1.0),
            {"vectorizer": "word_tfidf", "word_ngram_range": "1,2", "word_min_df": 2,
             "word_max_features": 100_000, "classifier": "LogisticRegression", "C": 1.0,
             "class_weight": "balanced", "max_iter": 1200},
        ),
        "word_logistic_c3": (
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True),
            LogisticRegression(max_iter=1200, class_weight="balanced", C=3.0),
            {"vectorizer": "word_tfidf", "word_ngram_range": "1,2", "word_min_df": 2,
             "word_max_features": 100_000, "classifier": "LogisticRegression", "C": 3.0,
             "class_weight": "balanced", "max_iter": 1200},
        ),
        "word_char_linear_svc": (
            word_char,
            LinearSVC(C=1.2, class_weight="balanced"),
            {"vectorizer": "word_char_tfidf", "word_ngram_range": "1,2",
             "word_min_df": 2, "word_max_features": 100_000,
             "char_ngram_range": "3,5", "char_min_df": 3, "char_max_features": 120_000,
             "classifier": "LinearSVC", "C": 1.2, "class_weight": "balanced"},
        ),
    }
    return {
        name: (Pipeline([("tfidf", vectorizer), ("classifier", classifier)]),
               {"model": name, "seed": SEED, "data_version": "v2", **params})
        for name, (vectorizer, classifier, params) in configs.items()
    }


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "working-tree"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Сначала выполните `poetry run python -m wb_sentiment.prepare`.")
    ARTIFACT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    frame = pd.read_csv(DATA_PATH)
    partitions = grouped_split(frame)
    x_train = partitions["train"]["review_text"].tolist()
    y_train = partitions["train"]["sentiment"].tolist()
    x_val = partitions["validation"]["review_text"].tolist()
    y_val = partitions["validation"]["sentiment"].tolist()
    x_fit = pd.concat([partitions["train"], partitions["validation"]])["review_text"].tolist()
    y_fit = pd.concat([partitions["train"], partitions["validation"]])["sentiment"].tolist()
    x_test = partitions["test"]["review_text"].tolist()
    y_test = partitions["test"]["sentiment"].tolist()

    default_tracking_uri = f"sqlite:///{ROOT / 'mlflow.db'}"
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", default_tracking_uri))
    mlflow.set_experiment("wildberries-sentiment")
    outcomes = []
    chosen = None
    for name, (model, params) in build_models().items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({**params, "split_seed": SEED, "split": "70/10/20 grouped by product_id"})
            mlflow.set_tags({"git_commit": git_revision(), "dataset_sha256": sha256(DATA_PATH)})
            model.fit(x_train, y_train)
            prediction = model.predict(x_val)
            scores = {
                "validation_macro_f1": f1_score(y_val, prediction, average="macro"),
                "validation_accuracy": accuracy_score(y_val, prediction),
                "validation_weighted_f1": f1_score(y_val, prediction, average="weighted"),
            }
            mlflow.log_metrics(scores)
            outcomes.append({"run_id": run.info.run_id, **params, **scores})
            if chosen is None or scores["validation_macro_f1"] > chosen["score"]:
                chosen = {"name": name, "params": params, "score": scores["validation_macro_f1"]}

    assert chosen is not None
    results = pd.DataFrame(outcomes).sort_values("validation_macro_f1", ascending=False)
    winner = results.iloc[0]
    chosen_model = dict(build_models())[chosen["name"]][0]
    chosen_model.fit(x_fit, y_fit)
    test_prediction = chosen_model.predict(x_test)
    test_metrics = {
        "test_macro_f1": f1_score(y_test, test_prediction, average="macro"),
        "test_accuracy": accuracy_score(y_test, test_prediction),
        "test_weighted_f1": f1_score(y_test, test_prediction, average="weighted"),
    }
    results["selected_for_final"] = results["model"] == chosen["name"]
    results.to_csv(REPORT_DIR / "experiments.csv", index=False)
    (REPORT_DIR / "test_report.json").write_text(
        json.dumps({"selected_model": chosen["name"], **test_metrics,
                    "classification_report": classification_report(y_test, test_prediction,
                                                                     labels=CLASS_ORDER,
                                                                     output_dict=True,
                                                                     zero_division=0)},
                   ensure_ascii=False, indent=2) + "\n"
    )
    bundle = {
        "model": chosen_model,
        "model_version": "1.0.0",
        "model_name": chosen["name"],
        "classes": list(CLASS_ORDER),
        "training_data_version": "v2",
        "training_data_sha256": sha256(DATA_PATH),
        "git_commit": git_revision(),
        "validation_macro_f1": float(winner["validation_macro_f1"]),
        **test_metrics,
    }
    model_path = ARTIFACT_DIR / "final_model.joblib"
    joblib.dump(bundle, model_path)
    with mlflow.start_run(run_name=f"final-{chosen['name']}"):
        mlflow.log_params({"selected_model": chosen["name"], "data_version": "v2", "seed": SEED})
        mlflow.log_metrics({**test_metrics, "validation_macro_f1": chosen["score"]})
        mlflow.set_tags({"git_commit": git_revision(), "dataset_sha256": sha256(DATA_PATH)})
        mlflow.log_artifact(str(model_path), artifact_path="model")
    print(results.to_string(index=False))
    print(json.dumps({"chosen": chosen["name"], **test_metrics}, indent=2))
    print(f"Модель: {model_path}")


if __name__ == "__main__":
    main()
