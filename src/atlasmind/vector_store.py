from collections.abc import Sequence

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from sklearn.feature_extraction.text import HashingVectorizer

from atlasmind.retrieval import Passage

EMBEDDING_DIMENSIONS = 384

CREATE_EXTENSION_SQL = sql.SQL("CREATE EXTENSION IF NOT EXISTS vector;")

CREATE_PASSAGES_SQL = sql.SQL("""
CREATE TABLE IF NOT EXISTS knowledge_passages (
    passage_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    passage_text TEXT NOT NULL,
    source_url TEXT NOT NULL,
    embedding VECTOR({dimensions}) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
""").format(dimensions=sql.Literal(EMBEDDING_DIMENSIONS))

CREATE_VECTOR_INDEX_SQL = sql.SQL("""
CREATE INDEX IF NOT EXISTS knowledge_passages_embedding_idx
ON knowledge_passages
USING hnsw (embedding vector_cosine_ops);
""")

MIGRATE_WIKIPEDIA_PASSAGES_SQL = sql.SQL("""
DO $$
BEGIN
    IF to_regclass('public.wikipedia_passages') IS NOT NULL THEN
        INSERT INTO knowledge_passages (
            passage_id, title, passage_text, source_url, embedding, updated_at
        )
        SELECT passage_id, title, passage_text, source_url, embedding, updated_at
        FROM wikipedia_passages
        ON CONFLICT (passage_id) DO NOTHING;
    END IF;
END $$;
""")

UPSERT_PASSAGE_SQL = sql.SQL("""
INSERT INTO knowledge_passages (
    passage_id, title, passage_text, source_url, embedding
)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (passage_id) DO UPDATE SET
    title = EXCLUDED.title,
    passage_text = EXCLUDED.passage_text,
    source_url = EXCLUDED.source_url,
    embedding = EXCLUDED.embedding,
    updated_at = NOW();
""")


class LocalTextEmbedder:
    """Create deterministic local vectors without a model download or API call."""

    def __init__(self) -> None:
        self.vectorizer = HashingVectorizer(
            n_features=EMBEDDING_DIMENSIONS,
            alternate_sign=False,
            norm="l2",
            stop_words="english",
            ngram_range=(1, 2),
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(texts)
        return np.asarray(matrix.toarray(), dtype=np.float32)


class PgVectorKnowledgeStore:
    def __init__(self, dsn: str, embedder: LocalTextEmbedder) -> None:
        self.dsn = dsn
        self.embedder = embedder

    def initialize(self) -> None:
        with psycopg.connect(self.dsn) as connection:
            connection.execute(CREATE_EXTENSION_SQL)
            register_vector(connection)
            connection.execute(CREATE_PASSAGES_SQL)
            connection.execute(MIGRATE_WIKIPEDIA_PASSAGES_SQL)
            connection.execute(CREATE_VECTOR_INDEX_SQL)

    def index(self, passages: list[Passage]) -> int:
        if not passages:
            return 0
        embeddings = self.embedder.encode([passage.text for passage in passages])
        with psycopg.connect(self.dsn) as connection:
            connection.execute(CREATE_EXTENSION_SQL)
            register_vector(connection)
            connection.execute(CREATE_PASSAGES_SQL)
            connection.execute(MIGRATE_WIKIPEDIA_PASSAGES_SQL)
            with connection.cursor() as cursor:
                cursor.executemany(
                    UPSERT_PASSAGE_SQL,
                    [
                        (
                            passage.passage_id,
                            passage.title,
                            passage.text,
                            passage.url,
                            embedding,
                        )
                        for passage, embedding in zip(passages, embeddings)
                    ],
                )
            connection.execute(CREATE_VECTOR_INDEX_SQL)
        return len(passages)

    def search(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        query_vector = self.embedder.encode([query])[0]
        search_sql = sql.SQL("""
        SELECT
            passage_id,
            title,
            source_url,
            passage_text AS snippet,
            1 - (embedding <=> %s) AS score
        FROM knowledge_passages
        ORDER BY embedding <=> %s
        LIMIT %s;
        """)
        with psycopg.connect(self.dsn) as connection:
            register_vector(connection)
            with connection.cursor() as cursor:
                cursor.execute(search_sql, (query_vector, query_vector, limit))
                description = cursor.description
                if description is None:
                    return []
                columns = [column.name for column in description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]


def reciprocal_rank_fusion(
    keyword_results: list[dict[str, object]],
    vector_results: list[dict[str, object]],
    limit: int,
    rank_constant: int = 60,
) -> list[dict[str, object]]:
    """Combine lexical and semantic rankings without comparing incompatible scores."""
    combined: dict[str, dict[str, object]] = {}
    scores: dict[str, float] = {}
    for results in (keyword_results, vector_results):
        for rank, result in enumerate(results, start=1):
            key = str(result["source_url"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank_constant + rank)
            combined.setdefault(key, result)
    ranked_urls = sorted(scores, key=lambda url: scores[url], reverse=True)[:limit]
    return [{**combined[url], "hybrid_score": round(scores[url], 8)} for url in ranked_urls]
