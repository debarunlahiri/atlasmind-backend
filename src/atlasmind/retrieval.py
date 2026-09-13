from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Passage:
    passage_id: str
    title: str
    text: str
    url: str


@dataclass(frozen=True)
class RetrievalHit:
    title: str
    text: str
    url: str
    score: float


def chunk_article(
    title: str,
    text: str,
    url: str,
    chunk_words: int = 220,
    overlap_words: int = 40,
) -> list[Passage]:
    if chunk_words <= overlap_words:
        raise ValueError("chunk_words must be greater than overlap_words")
    words = text.split()
    step = chunk_words - overlap_words
    return [
        Passage(
            passage_id=f"{title}:{start}",
            title=title,
            text=" ".join(words[start : start + chunk_words]),
            url=url,
        )
        for start in range(0, len(words), step)
        if words[start : start + chunk_words]
    ]


class KnowledgeIndex:
    def __init__(self) -> None:
        self.passages: list[Passage] = []
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=100_000,
            sublinear_tf=True,
        )
        self.matrix: Any | None = None

    def fit(self, passages: list[Passage]) -> int:
        if not passages:
            raise ValueError("At least one Wikipedia passage is required")
        self.passages = passages
        self.matrix = self.vectorizer.fit_transform(passage.text for passage in passages)
        return len(passages)

    def search(self, query: str, limit: int = 5) -> list[RetrievalHit]:
        if self.matrix is None or not self.passages:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        indices = np.argsort(scores)[::-1][:limit]
        return [
            RetrievalHit(
                title=self.passages[index].title,
                text=self.passages[index].text,
                url=self.passages[index].url,
                score=round(float(scores[index]), 6),
            )
            for index in indices
            if scores[index] > 0
        ]

    def save(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        joblib.dump(self, temporary)
        temporary.replace(destination)

    @classmethod
    def load(cls, source: Path) -> KnowledgeIndex:
        index = joblib.load(source)
        if not isinstance(index, cls):
            raise TypeError("Artifact is not an AtlasMind knowledge index")
        return index
