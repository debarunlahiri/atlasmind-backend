import hashlib
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from io import BytesIO
from typing import Optional

import psycopg
from PIL import Image
from psycopg import sql
from psycopg.types.json import Jsonb

from atlasmind.crawler import CrawledArticle

CREATE_DOCUMENTS_SQL = sql.SQL("""
CREATE TABLE IF NOT EXISTS knowledge_documents (
    document_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    title TEXT NOT NULL,
    document_text TEXT NOT NULL,
    source_url TEXT NOT NULL UNIQUE,
    labels JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    published_at TIMESTAMPTZ,
    collected_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
""")

CREATE_DOCUMENT_INDEX_SQL = sql.SQL("""
CREATE INDEX IF NOT EXISTS knowledge_documents_search_idx
ON knowledge_documents
USING GIN (to_tsvector('english', title || ' ' || document_text));
""")

CREATE_IMAGES_SQL = sql.SQL("""
CREATE TABLE IF NOT EXISTS knowledge_images (
    image_id TEXT PRIMARY KEY,
    image_bytes BYTEA NOT NULL,
    mime_type TEXT NOT NULL,
    width INTEGER NOT NULL CHECK (width > 0),
    height INTEGER NOT NULL CHECK (height > 0),
    source_type TEXT NOT NULL,
    source_url TEXT,
    original_name TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
""")

CREATE_IMAGE_SOURCE_INDEX_SQL = sql.SQL("""
CREATE INDEX IF NOT EXISTS knowledge_images_source_url_idx
ON knowledge_images (source_url)
WHERE source_url IS NOT NULL;
""")

CREATE_CONVERSATIONS_SQL = sql.SQL("""
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    client_fingerprint TEXT NOT NULL,
    title TEXT NOT NULL,
    user_agent TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
""")

CREATE_CONVERSATION_CLIENT_INDEX_SQL = sql.SQL("""
CREATE INDEX IF NOT EXISTS conversations_client_fingerprint_idx
ON conversations (client_fingerprint, updated_at DESC);
""")

CREATE_CHAT_HISTORY_SQL = sql.SQL("""
CREATE TABLE IF NOT EXISTS chat_history (
    message_id BIGSERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
""")

CREATE_CHAT_HISTORY_INDEX_SQL = sql.SQL("""
CREATE INDEX IF NOT EXISTS chat_history_conversation_idx
ON chat_history (conversation_id, message_id);
""")

CREATE_APPLICATION_LOGS_SQL = sql.SQL("""
CREATE TABLE IF NOT EXISTS application_logs (
    log_id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL UNIQUE,
    level TEXT NOT NULL,
    event TEXT NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    query_string TEXT NOT NULL DEFAULT '',
    status_code INTEGER,
    duration_ms DOUBLE PRECISION NOT NULL,
    client_fingerprint TEXT NOT NULL,
    request_headers JSONB NOT NULL DEFAULT '{}'::jsonb,
    request_body TEXT NOT NULL DEFAULT '',
    response_headers JSONB NOT NULL DEFAULT '{}'::jsonb,
    response_body TEXT NOT NULL DEFAULT '',
    error_detail TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
""")

CREATE_APPLICATION_LOGS_INDEX_SQL = sql.SQL("""
CREATE INDEX IF NOT EXISTS application_logs_created_at_idx
ON application_logs (created_at DESC);
""")

MIGRATE_WIKIPEDIA_SQL = sql.SQL("""
DO $$
BEGIN
    IF to_regclass('public.wikipedia_articles') IS NOT NULL THEN
        INSERT INTO knowledge_documents (
            document_id,
            source_type,
            title,
            document_text,
            source_url,
            labels,
            metadata,
            collected_at,
            updated_at
        )
        SELECT
            document_id,
            'wikipedia',
            title,
            article_text,
            source_url,
            labels,
            '{}'::jsonb,
            collected_at,
            updated_at
        FROM wikipedia_articles
        ON CONFLICT (document_id) DO NOTHING;
    END IF;
END $$;
""")

UPSERT_DOCUMENT_SQL = sql.SQL("""
INSERT INTO knowledge_documents (
    document_id,
    source_type,
    title,
    document_text,
    source_url,
    labels,
    metadata,
    published_at,
    collected_at
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (document_id) DO UPDATE SET
    source_type = EXCLUDED.source_type,
    title = EXCLUDED.title,
    document_text = EXCLUDED.document_text,
    source_url = EXCLUDED.source_url,
    labels = EXCLUDED.labels,
    metadata = EXCLUDED.metadata,
    published_at = EXCLUDED.published_at,
    collected_at = EXCLUDED.collected_at,
    updated_at = NOW();
""")

UPSERT_IMAGE_SQL = sql.SQL("""
INSERT INTO knowledge_images (
    image_id,
    image_bytes,
    mime_type,
    width,
    height,
    source_type,
    source_url,
    original_name,
    metadata
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (image_id) DO UPDATE SET
    source_type = EXCLUDED.source_type,
    source_url = COALESCE(EXCLUDED.source_url, knowledge_images.source_url),
    original_name = COALESCE(EXCLUDED.original_name, knowledge_images.original_name),
    metadata = knowledge_images.metadata || EXCLUDED.metadata,
    updated_at = NOW();
""")

UPSERT_CONVERSATION_SQL = sql.SQL("""
INSERT INTO conversations (
    conversation_id, client_fingerprint, title, user_agent
)
VALUES (%s, %s, %s, %s)
ON CONFLICT (conversation_id) DO UPDATE SET
    client_fingerprint = EXCLUDED.client_fingerprint,
    user_agent = EXCLUDED.user_agent,
    updated_at = NOW();
""")

INSERT_CHAT_MESSAGE_SQL = sql.SQL("""
INSERT INTO chat_history (conversation_id, role, content, metadata)
VALUES (%s, %s, %s, %s);
""")

SELECT_CONVERSATION_SQL = sql.SQL("""
SELECT title
FROM conversations
WHERE conversation_id = %s AND client_fingerprint = %s;
""")

SELECT_CHAT_HISTORY_SQL = sql.SQL("""
SELECT role, content
FROM (
    SELECT message_id, role, content
    FROM chat_history
    WHERE conversation_id = %s
    ORDER BY message_id DESC
    LIMIT %s
) AS recent_messages
ORDER BY message_id ASC;
""")

INSERT_APPLICATION_LOG_SQL = sql.SQL("""
INSERT INTO application_logs (
    request_id,
    level,
    event,
    method,
    path,
    query_string,
    status_code,
    duration_ms,
    client_fingerprint,
    request_headers,
    request_body,
    response_headers,
    response_body,
    error_detail,
    metadata
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
""")


class PostgresKnowledgeRepository:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def initialize(self) -> None:
        with psycopg.connect(self.dsn) as connection:
            self._initialize(connection)

    def upsert(self, articles: Iterable[CrawledArticle]) -> int:
        rows = list(articles)
        now = datetime.now(timezone.utc).isoformat()
        values = [
            (
                article.document_id,
                "wikipedia",
                article.title,
                article.text,
                article.url,
                Jsonb(article.labels),
                Jsonb({}),
                None,
                article.collected_at or now,
            )
            for article in rows
        ]
        self._upsert_documents(values)
        return len(rows)

    def upsert_web_results(self, results: list[dict[str, object]], query: str) -> int:
        now = datetime.now(timezone.utc).isoformat()
        values = []
        for result in results:
            source_url = str(result.get("source_url", "")).strip()
            snippet = str(result.get("snippet", "")).strip()
            if not source_url or not snippet:
                continue
            values.append(
                (
                    hashlib.sha256(source_url.encode()).hexdigest(),
                    "web_html",
                    str(result.get("title", "Web result")),
                    snippet,
                    source_url,
                    Jsonb(["web", "latest"]),
                    Jsonb(
                        {
                            "query": query,
                            "image_id": result.get("image_id"),
                            "image_url": result.get("image_url"),
                            "retrieval": "direct_html",
                        }
                    ),
                    result.get("published_at"),
                    now,
                )
            )
        self._upsert_documents(values)
        return len(values)

    def save_image(
        self,
        image: Image.Image,
        original_name: Optional[str] = None,
        source_url: Optional[str] = None,
        source_type: str = "user_upload",
        metadata: Optional[dict[str, object]] = None,
    ) -> str:
        normalized = image.convert("RGB")
        output = BytesIO()
        normalized.save(output, format="PNG", optimize=True)
        image_bytes = output.getvalue()
        image_id = hashlib.sha256(image_bytes).hexdigest()
        with psycopg.connect(self.dsn) as connection:
            self._initialize(connection)
            connection.execute(
                UPSERT_IMAGE_SQL,
                (
                    image_id,
                    image_bytes,
                    "image/png",
                    normalized.width,
                    normalized.height,
                    source_type,
                    source_url,
                    original_name,
                    Jsonb(metadata or {}),
                ),
            )
        return image_id

    def save_web_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        source_url: str,
        document_url: str,
        title: str,
    ) -> Optional[str]:
        if len(image_bytes) > 10 * 1024 * 1024:
            return None
        try:
            image = Image.open(BytesIO(image_bytes))
            image.load()
        except OSError:
            return None
        if image.width > 4096 or image.height > 4096:
            return None
        return self.save_image(
            image,
            source_url=source_url,
            source_type="web_html",
            metadata={
                "document_url": document_url,
                "title": title,
                "original_mime_type": mime_type,
            },
        )

    def search(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        search_sql = sql.SQL("""
        SELECT
            document_id,
            source_type,
            title,
            source_url,
            labels,
            published_at,
            ts_rank(
                to_tsvector('english', title || ' ' || document_text),
                websearch_to_tsquery('english', %s)
            ) AS score,
            ts_headline(
                'english', document_text, websearch_to_tsquery('english', %s),
                'MaxWords=45, MinWords=20'
            ) AS snippet
        FROM knowledge_documents
        WHERE to_tsvector('english', title || ' ' || document_text)
              @@ websearch_to_tsquery('english', %s)
        ORDER BY score DESC, published_at DESC NULLS LAST
        LIMIT %s;
        """)
        with psycopg.connect(self.dsn) as connection:
            self._initialize(connection)
            with connection.cursor() as cursor:
                cursor.execute(search_sql, (query, query, query, limit))
                description = cursor.description
                if description is None:
                    return []
                columns = [column.name for column in description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def record_chat(
        self,
        conversation_id: str,
        client_fingerprint: str,
        title: str,
        user_agent: str,
        question: str,
        answer: str,
        sources: list[dict[str, object]],
        image_id: Optional[str],
    ) -> None:
        with psycopg.connect(self.dsn) as connection:
            self._initialize(connection)
            connection.execute(
                UPSERT_CONVERSATION_SQL,
                (conversation_id, client_fingerprint, title, user_agent),
            )
            with connection.cursor() as cursor:
                cursor.executemany(
                    INSERT_CHAT_MESSAGE_SQL,
                    [
                        (
                            conversation_id,
                            "user",
                            question,
                            Jsonb({"image_id": image_id}),
                        ),
                        (
                            conversation_id,
                            "assistant",
                            answer,
                            Jsonb({"sources": sources}),
                        ),
                    ],
                )

    def conversation_context(
        self,
        conversation_id: str,
        client_fingerprint: str,
        message_limit: int = 12,
    ) -> tuple[Optional[str], list[dict[str, str]]]:
        with psycopg.connect(self.dsn) as connection:
            self._initialize(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    SELECT_CONVERSATION_SQL,
                    (conversation_id, client_fingerprint),
                )
                conversation = cursor.fetchone()
                if conversation is None:
                    return None, []
                cursor.execute(
                    SELECT_CHAT_HISTORY_SQL,
                    (conversation_id, message_limit),
                )
                messages = [
                    {"role": str(role), "content": str(content)}
                    for role, content in cursor.fetchall()
                ]
                return str(conversation[0]), messages

    def record_application_log(self, entry: dict[str, object]) -> None:
        with psycopg.connect(self.dsn) as connection:
            connection.execute(
                INSERT_APPLICATION_LOG_SQL,
                (
                    entry["request_id"],
                    entry["level"],
                    entry["event"],
                    entry["method"],
                    entry["path"],
                    entry["query_string"],
                    entry["status_code"],
                    entry["duration_ms"],
                    entry["client_fingerprint"],
                    Jsonb(entry["request_headers"]),
                    entry["request_body"],
                    Jsonb(entry["response_headers"]),
                    entry["response_body"],
                    entry["error_detail"],
                    Jsonb(entry["metadata"]),
                ),
            )

    def _upsert_documents(self, values: list[tuple[object, ...]]) -> None:
        if not values:
            return
        with psycopg.connect(self.dsn) as connection:
            self._initialize(connection)
            with connection.cursor() as cursor:
                cursor.executemany(UPSERT_DOCUMENT_SQL, values)

    @staticmethod
    def _initialize(connection: psycopg.Connection) -> None:
        connection.execute(CREATE_DOCUMENTS_SQL)
        connection.execute(CREATE_DOCUMENT_INDEX_SQL)
        connection.execute(CREATE_IMAGES_SQL)
        connection.execute(CREATE_IMAGE_SOURCE_INDEX_SQL)
        connection.execute(CREATE_CONVERSATIONS_SQL)
        connection.execute(CREATE_CONVERSATION_CLIENT_INDEX_SQL)
        connection.execute(CREATE_CHAT_HISTORY_SQL)
        connection.execute(CREATE_CHAT_HISTORY_INDEX_SQL)
        connection.execute(CREATE_APPLICATION_LOGS_SQL)
        connection.execute(CREATE_APPLICATION_LOGS_INDEX_SQL)
        connection.execute(MIGRATE_WIKIPEDIA_SQL)


PostgresArticleRepository = PostgresKnowledgeRepository


def article_as_row(article: CrawledArticle) -> dict[str, object]:
    return asdict(article)
