from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from atlasmind.retrieval import KnowledgeIndex, Passage, chunk_article
from atlasmind.storage import ExpansionStorage
from atlasmind.vector_store import PgVectorKnowledgeStore


@dataclass(frozen=True)
class TrainingSummary:
    articles: int
    passages: int
    labels: dict[str, int]
    ignored_labels: dict[str, int]
    classifier_metrics: dict[str, Any]
    classifier_path: str
    index_path: str
    vector_passages: int
    trained_at: str


def train_from_wikipedia(
    storage: ExpansionStorage,
    vector_store: PgVectorKnowledgeStore,
    random_seed: int = 42,
) -> TrainingSummary:
    """Train a topic classifier and fit a retrieval index from stored Wikipedia text."""
    rows = list(storage.read_jsonl("corpus/wikipedia_articles.jsonl"))
    candidate_examples = [row for row in rows if row.get("text") and row.get("labels")]
    candidate_counts = Counter(row["labels"][0] for row in candidate_examples)
    ignored_labels = {label: count for label, count in candidate_counts.items() if count < 2}
    examples = [row for row in candidate_examples if candidate_counts[row["labels"][0]] >= 2]
    if len(examples) < 12:
        counts = Counter(label for row in examples for label in row.get("labels", []))
        raise ValueError(
            "Training requires at least 12 labeled Wikipedia articles. "
            f"Found {len(examples)} with label counts {dict(sorted(counts.items()))}. "
            "Run the crawl command again before training."
        )

    texts = [row["text"] for row in examples]
    labels = [row["labels"][0] for row in examples]
    label_counts = Counter(labels)
    if len(label_counts) < 2:
        raise ValueError("Training requires at least two distinct topics")

    test_count = max(len(label_counts), ceil(len(examples) * 0.25))
    test_count = min(test_count, len(examples) - len(label_counts))
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=test_count,
        random_state=random_seed,
        stratify=labels,
    )
    classifier = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=75_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=random_seed,
                ),
            ),
        ]
    )
    classifier.fit(train_texts, train_labels)
    predictions = classifier.predict(test_texts)
    metrics = {
        "accuracy": round(float(accuracy_score(test_labels, predictions)), 6),
        "macro_f1": round(float(f1_score(test_labels, predictions, average="macro")), 6),
        "report": classification_report(
            test_labels,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    passages: list[Passage] = []
    for row in examples:
        passages.extend(chunk_article(row["title"], row["text"], row["url"]))
    index = KnowledgeIndex()
    index.fit(passages)

    model_dir = storage.prepare("models")
    index_dir = storage.prepare("indexes")
    classifier_path = _atomic_joblib_dump(classifier, model_dir / "topic_classifier.joblib")
    index_path = index_dir / "wikipedia_tfidf.joblib"
    index.save(index_path)
    vector_passages = vector_store.index(passages)
    return TrainingSummary(
        articles=len(examples),
        passages=len(passages),
        labels=dict(sorted(label_counts.items())),
        ignored_labels=dict(sorted(ignored_labels.items())),
        classifier_metrics=metrics,
        classifier_path=str(classifier_path),
        index_path=str(index_path),
        vector_passages=vector_passages,
        trained_at=datetime.now(timezone.utc).isoformat(),
    )


def _atomic_joblib_dump(value: Any, destination: Path) -> Path:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    joblib.dump(value, temporary)
    temporary.replace(destination)
    return destination
