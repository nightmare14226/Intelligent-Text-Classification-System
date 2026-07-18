"""TF-IDF plus logistic-regression text classification baseline."""

import pickle
import time
from typing import Any

from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from evaluation import classification_metrics


def run_tfidf_experiment(
    train_data: Dataset,
    validation_data: Dataset,
    test_data: Dataset,
    max_features: int = 50_000,
) -> tuple[dict[str, Any], list[int]]:
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )

    train_started = time.perf_counter()
    pipeline.fit(train_data["text"], train_data["label"])
    training_seconds = time.perf_counter() - train_started

    validation_predictions = pipeline.predict(validation_data["text"])
    inference_started = time.perf_counter()
    test_predictions = pipeline.predict(test_data["text"])
    inference_seconds = time.perf_counter() - inference_started

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    positive_indices = classifier.coef_[0].argsort()[-10:][::-1]
    negative_indices = classifier.coef_[0].argsort()[:10]

    results = {
        "model": "TF-IDF + Logistic Regression",
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "milliseconds_per_sample": 1_000 * inference_seconds / len(test_data),
        "model_size_mb": len(pickle.dumps(pipeline)) / (1024**2),
        "vocabulary_size": len(feature_names),
        "validation": classification_metrics(
            validation_data["label"], validation_predictions
        ),
        "test": classification_metrics(test_data["label"], test_predictions),
        "most_positive_features": feature_names[positive_indices].tolist(),
        "most_negative_features": feature_names[negative_indices].tolist(),
    }
    return results, test_predictions.tolist()
