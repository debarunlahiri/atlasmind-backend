# AtlasMind system design

## Data flow

```text
Wikipedia topics                 User question
          |                            |
          v                            v
robots-aware HTML crawler      HTML web discovery
          |
          v
clean + canonicalize + deduplicate
          |
          +--------------------+       |
          |                    |<------+
          v                    v
Expansion JSONL         PostgreSQL corpus + pgvector
          |                    |
          +----------+---------+
                     |
          +----------+----------+
          |                     |
          v                     v
 topic classifier       local knowledge index
          |                     |
          +----------+----------+
                     |
                     v
      Qwen2.5-VL text + image generation on Apple MPS
                     |
                     v
       JSON response or Server-Sent Events stream
                     |
              save_history=true
                     |
                     v
        conversations + chat_history
```

The crawler converts each topic into a Wikipedia article path without search,
then runs breadth-first and remains restricted to that Wikipedia host. It rejects
external sites, non-HTTPS URLs, and Wikipedia special namespaces. The source URL
is retained for attribution and traceability.

## Direct HTML web discovery

Fresh retrieval submits the question to DuckDuckGo's public HTML form, parses the
ordinary result markup, and follows a bounded number of public HTTP/HTTPS links.
It does not use a JSON search API or require an API key. Readable paragraphs and
page metadata are normalized into the same knowledge-document contract as the
Wikipedia corpus. Source pages are fetched concurrently with a bounded worker
pool. A page's representative Open Graph or Twitter image can be downloaded,
validated with Pillow, normalized to PNG, and stored in PostgreSQL when web-image
persistence is enabled. It is disabled by default to avoid an additional network
request and database write for every result.

If the freshness-filtered HTML search produces no parseable results, discovery
retries once with a compact keyword query and a broader time range, then tries
Bing News HTML for news-like questions and Bing's general public HTML result page
if DuckDuckGo remains blocked. Entity-role questions can also open a canonical
Wikipedia article directly, which helps correct premises such as treating a
building as a company with a CEO. Brave Web HTML is the final configured fallback
when Bing returns no relevant results. No search API or API key is used, and
CAPTCHA controls are not bypassed. Results must overlap meaningful query terms before
their source pages are fetched. When live retrieval still returns no evidence,
text-only generation is skipped and the API returns an insufficient-evidence
response. This prevents a small local model from fabricating current news while
still allowing ungrounded generation when the caller explicitly disables web
search.

The retrieval result carries an ordered `search_activity` trace into complete
JSON responses and SSE metadata. This makes provider challenges and fallback
decisions visible without relying on terminal logs. Every accepted web source is
also tagged with the HTML engine that discovered it.

Explicit URLs form a pre-search retrieval stage. Plain and Markdown URLs are
deduplicated and fetched directly; page and social-card metadata provide context
even when a social platform renders most content with JavaScript. That metadata
enriches the related-coverage query, while the original page remains an
attributed `Direct URL HTML` source.

## Persistence

Expansion JSON Lines is the reproducible Wikipedia training source. PostgreSQL
uses generalized `knowledge_documents`, `knowledge_passages`, and
`knowledge_images` tables for operational storage. The `conversations` and
`chat_history` tables hold backend conversation metadata and ordered user and
assistant messages. Cross-system writes cannot be one database transaction, so
repeated crawls use stable document IDs and idempotent upserts.

The FastAPI lifespan hook initializes both the document repository and pgvector
store whenever a server worker starts. Development reloads create a new worker,
so the same check runs after every source change. All schema statements use
`IF NOT EXISTS`; missing tables, indexes, and the `vector` extension are created
without replacing existing objects or deleting data. PostgreSQL and the target
database must already exist, and initialization errors stop worker startup.

## Conversation history

History saving is enabled by default and can be disabled per request with
`"save_history": false`. A request without a `conversation_id` receives a new
UUID and a short chat name derived from its first question. Supplying that UUID
on later requests appends messages to the same conversation. The original title
remains the stored conversation title.

The repository verifies that the UUID belongs to the request's pseudonymous
client fingerprint and loads up to 12 recent messages. A follow-up detector
applies them only to referential questions containing pronouns or continuation
phrases. It combines the last two user questions with that follow-up before web
search and sends up to eight messages to generation. Standalone questions ignore
old context, even inside the same conversation. Unknown or mismatched IDs expose
no context.

`conversations` stores the conversation UUID, generated title, pseudonymous
client fingerprint, User-Agent string, and timestamps. `chat_history` stores the
ordered user and assistant messages plus JSON metadata for sources and the
related image ID. This version intentionally provides no public endpoint for
listing or reading saved history.

For streaming responses, AtlasMind collects the emitted text and writes the two
messages only after generation finishes successfully. It does not persist a
partial assistant response after a streaming error. Opting out also prevents
that request's uploaded image, discovered web pages, search query, and chat
messages from being stored.

## Client identity and privacy

A normal HTTP server cannot obtain a remote device's MAC address because MAC
addresses do not travel across routed networks. AtlasMind instead hashes the
direct connection IP together with the User-Agent using SHA-256. The raw IP is
not stored. This fingerprint is only a best-effort grouping mechanism: it is not
authentication, can change as networks or browsers change, and may group users
behind the same proxy.

The application currently uses the socket peer address and does not trust
forwarded IP headers. A deployment behind a reverse proxy must configure a
trusted-proxy boundary before using forwarded addresses. Because the API allows
anonymous requests and has no account ownership checks, keep it bound to
`127.0.0.1` unless authentication, authorization, retention, and deletion
controls are added.

## Operational logging

An ASGI middleware captures every HTTP exchange without buffering delivery to
the client. It assigns an `X-Request-ID`, records request and response headers and
bodies, status, duration, route details, errors, transport metadata, and the same
pseudonymous client fingerprint in `application_logs`. SSE chunks are forwarded
immediately and accumulated only for the final log record. Sensitive credential
and cookie headers are redacted before terminal or database output.

Development mode mirrors each structured exchange to the server terminal. Other
environments keep database logging enabled but omit full terminal exchange logs.
Logging is operational and independent of conversation-history opt-out. A log
write failure is reported to the server logger but does not replace a response
that has already been delivered. Production deployments need restricted table
access and explicit retention and deletion schedules because bodies can contain
private user content.

## Local search

PostgreSQL tokenizes titles and article text with an English text-search
configuration. A GIN index accelerates `websearch_to_tsquery`, while `ts_rank`
orders matches and `ts_headline` returns concise snippets.

Each chunk also receives a deterministic 384-dimensional local hashing vector.
`pgvector` stores the vectors and an HNSW cosine index accelerates nearest-neighbor
retrieval. Reciprocal-rank fusion combines keyword and vector rankings without
pretending their raw scores have the same scale.

## Training

Seed labels provide weak supervision. Training uses TF-IDF text features, a balanced Logistic Regression classifier, a stratified held-out split, and macro F1. The separate retrieval index chunks full article text with overlap.

Prevent evaluation leakage by grouping redirects, near-duplicates, and closely linked variants into the same split. A production evaluation should use independently reviewed labels.

## Scaling

For a large crawl, replace in-memory accumulation with a durable frontier, content hashes, retry queues, and incremental batches. Use PostgreSQL row locking or a queue to coordinate multiple workers. Never increase concurrency or reduce crawl delay without checking Wikimedia's current policies.

The hashing vectors are a lightweight vector baseline, not a deep semantic model.
For stronger semantic retrieval, provision a transformer model locally on the
Expansion drive and keep its inference offline. The vector-store boundary can
accept those embeddings without adding a hosted API.

## Multimodal generation

The RAG layer sends retrieved Wikipedia passages, the user's question, and an
optional image to the Apache-2.0 Qwen2.5-VL 3B vision-language model. The model is
loaded lazily through Transformers, cached below the Expansion storage root, and
moved to PyTorch's `mps` device in float16 on Apple Silicon. CLI images are local
files; REST images are decoded from base64 and are never forwarded to a hosted
inference service.

The REST layer offers both a complete JSON response and a Server-Sent Events
stream. Streaming runs model generation in a background thread and forwards
decoded text chunks as `token` events, preceded by source metadata and followed
by a `done` or `error` event. Proxy buffering is explicitly disabled for the
streaming response.

## Safety and operations

- Identify the crawler with a real contact address.
- Respect `robots.txt`, response errors, backoff, and crawl delay.
- Cap page counts and restrict allowed domains.
- Treat scraped text as untrusted input.
- Retain source URLs and collection timestamps.
- Publish a clear retention policy and periodically remove expired conversations.
- Preserve the history opt-out across chat, image, and retrieval-result storage.
- Do not treat the client fingerprint as a user account or security credential.
- Monitor crawl failures, drive availability, database availability, corpus growth, duplicates, label balance, and search quality.
