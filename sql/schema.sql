CREATE EXTENSION IF NOT EXISTS vector;

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

CREATE INDEX IF NOT EXISTS knowledge_documents_search_idx
ON knowledge_documents
USING GIN (to_tsvector('english', title || ' ' || document_text));

CREATE TABLE IF NOT EXISTS knowledge_passages (
    passage_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    passage_text TEXT NOT NULL,
    source_url TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS knowledge_passages_embedding_idx
ON knowledge_passages
USING hnsw (embedding vector_cosine_ops);

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

CREATE INDEX IF NOT EXISTS knowledge_images_source_url_idx
ON knowledge_images (source_url)
WHERE source_url IS NOT NULL;

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    client_fingerprint TEXT NOT NULL,
    title TEXT NOT NULL,
    user_agent TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS conversations_client_fingerprint_idx
ON conversations (client_fingerprint, updated_at DESC);

CREATE TABLE IF NOT EXISTS chat_history (
    message_id BIGSERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chat_history_conversation_idx
ON chat_history (conversation_id, message_id);

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

CREATE INDEX IF NOT EXISTS application_logs_created_at_idx
ON application_logs (created_at DESC);

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

    IF to_regclass('public.wikipedia_passages') IS NOT NULL THEN
        INSERT INTO knowledge_passages (
            passage_id,
            title,
            passage_text,
            source_url,
            embedding,
            updated_at
        )
        SELECT
            passage_id,
            title,
            passage_text,
            source_url,
            embedding,
            updated_at
        FROM wikipedia_passages
        ON CONFLICT (passage_id) DO NOTHING;
    END IF;
END $$;
