# AtlasMind code reference

This document explains how the source code is organized and how the main components
collaborate. For installation and command examples, see the project
[README](../README.md). For HTTP payloads and examples, see the
[API reference](API.md). For architectural decisions and operational constraints,
see the [system design](SYSTEM_DESIGN.md).

## Package layout

```text
src/atlasmind/
├── api/
│   ├── app.py                 FastAPI application and request orchestration
│   ├── logging_middleware.py  Request/response capture and persistence
│   └── schemas.py             Pydantic HTTP contracts
├── cli.py                     Command-line entry point
├── config.py                  Environment-backed settings
├── corpus.py                  JSONL and PostgreSQL crawl persistence
├── crawler.py                 Wikipedia-only HTML crawler
├── database.py                PostgreSQL documents, images, history, and logs
├── dev_server.py              Reloading Uvicorn launcher
├── generation.py              Local multimodal model adapter
├── history.py                 Conversation identity and title helpers
├── rag.py                     Retrieval and generation orchestration
├── retrieval.py               Passage chunking and TF-IDF index
├── storage.py                 Expansion-drive artifact boundary
├── training.py                Classifier and retrieval-index training
├── vector_store.py            Local embeddings and pgvector search
└── web_search.py              Direct-HTML web discovery and page reading
```

Other important paths:

- `sql/schema.sql` is the canonical standalone PostgreSQL schema.
- `tests/` contains unit tests for pure logic and isolated service behavior.
- `notebooks/` is reserved for corpus exploration outside the application runtime.
- `.env.example` lists every supported environment variable.

## Runtime composition

The application uses constructor injection at the RAG boundary. This keeps retrieval,
web discovery, and generation replaceable and allows tests to supply lightweight fake
implementations.

```text
FastAPI or CLI
      |
      v
KnowledgeRagService
  |       |       |
  v       v       v
keyword  pgvector direct-HTML search
search   search    (optional)
  \       |       /
   reciprocal-rank fusion
            |
            v
   LocalMultimodalGenerator
```

`api.app` constructs process-wide instances with `functools.lru_cache`. The API
lifespan initializes document and vector tables before accepting requests. The CLI
constructs only the dependencies required by the selected subcommand.

## Configuration

`atlasmind.config.Settings` loads `.env` through `pydantic-settings`. All environment
variables use the `ATLASMIND_` prefix. Values are parsed and validated when settings
are created.

| Setting field | Purpose | Important constraint |
| --- | --- | --- |
| `environment` | Controls development terminal logging | Database logging remains enabled in every environment |
| `storage_root` | Root for corpus, indexes, and model cache | `ExpansionStorage` requires it to be under `/Volumes/Expansion` |
| `wikipedia_language` | Wikipedia subdomain used for crawl seeds | The crawler validates the language as a subdomain code |
| `wikipedia_user_agent` | Identity sent by the Wikipedia crawler | Configure a real contact before crawling |
| `request_timeout_seconds` | Wikipedia request timeout | Must be greater than 0 and at most 120 seconds |
| `crawl_delay_seconds` | Delay between Wikipedia page requests | Must be between 0.5 and 60 seconds |
| `postgres_dsn` | PostgreSQL connection string | PostgreSQL and the target database must already exist |
| `compute_device` | Local inference device | One of `mps`, `cpu`, or `auto` |
| `multimodal_model_id` | Hugging Face model identifier | Loaded locally by Transformers |
| `model_local_files_only` | Prevents model download attempts | Requires a complete local cache when enabled |
| `generator_max_context_characters` | Total retrieved excerpt budget | Must be between 250 and 50,000 characters |
| `web_search_timeout_seconds` | Timeout for HTML search and page reads | Must be greater than 0 and at most 30 seconds |
| `web_search_max_workers` | Concurrent result-page readers | Must be between 1 and 10 |
| `web_search_store_results` | Persists accepted web documents | Also controlled per request by history opt-out |
| `web_search_store_images` | Persists representative page images | Disabled by default |
| `web_search_freshness` | Default HTML-search time range | `pd`, `pw`, `pm`, or `py` |

`get_settings()` is cached. Tests or embedding applications that change environment
variables after the first call must clear that cache before requesting settings again.

## Command-line interface

`atlasmind.cli:main` backs both `python -m atlasmind.cli` and the installed
`atlasmind` console command.

| Command | Main collaborators | Result |
| --- | --- | --- |
| `init-db` | `PostgresArticleRepository`, `PgVectorKnowledgeStore` | Creates required tables, indexes, and the vector extension |
| `crawl` | `WikipediaCrawler`, `CorpusWriter` | Crawls topic pages and mirrors articles to JSONL and PostgreSQL |
| `train` | `train_from_wikipedia` | Writes the classifier and TF-IDF index, then indexes passages in pgvector |
| `search` | PostgreSQL repository and/or pgvector store | Returns keyword, vector, or fused local results |
| `classify` | Saved scikit-learn classifier | Returns labels ranked by probability |
| `generate` | Repository, vector store, optional web search, local generator | Returns a grounded text or image answer |

Before importing command-specific libraries, `missing_dependencies()` checks the
modules needed by the chosen command and reports the corresponding package names.

## Wikipedia collection

### `CrawlSeed`

`CrawlSeed.from_topic(topic, language)` normalizes a topic and maps common aliases
such as `ai`, `ml`, and `nlp` to canonical article titles. It constructs a direct
Wikipedia article URL; it does not call a search service.

### `WikipediaCrawler`

The crawler performs a breadth-first traversal for each seed:

1. Validate that the seed is an HTTPS Wikipedia article URL.
2. Load and cache the host's `robots.txt` rules.
3. Fetch a page with the configured User-Agent and timeout.
4. Remove navigation, tables, scripts, references, and other non-article elements.
5. Keep readable paragraphs and reject pages with fewer than 80 words.
6. Canonicalize the source URL and derive a stable SHA-256 document ID.
7. Queue valid article links from the same Wikipedia host.
8. Wait for the configured crawl delay before the next page.

`CrawlProgress` is delivered to the optional progress callback after a successful page
or an HTTP failure. A failure is recorded for progress reporting and the traversal
continues while queued pages remain.

### `CorpusWriter`

`CorpusWriter.save()` merges a crawl into
`corpus/wikipedia_articles.jsonl` by stable document ID, replaces the JSONL file
atomically, and upserts the same crawl into PostgreSQL. The JSONL corpus is the input
for training; PostgreSQL is the operational search store.

## Training and local retrieval

`train_from_wikipedia()` performs three related operations:

1. Read labeled articles from the JSONL corpus.
2. Train and evaluate a TF-IDF plus balanced Logistic Regression topic classifier.
3. Chunk article text, save a TF-IDF retrieval index, and index local vectors in
   PostgreSQL.

Labels with fewer than two examples are ignored. Training requires at least 12 usable
articles and at least two distinct labels. The held-out split is stratified and the
summary includes accuracy, macro F1, and a classification report.

`chunk_article()` defaults to 220-word passages with 40 words of overlap.
`KnowledgeIndex` is the serialized TF-IDF retrieval artifact. The production RAG path
currently searches PostgreSQL and pgvector directly; the saved index remains useful
for local experiments and evaluation.

`LocalTextEmbedder` uses scikit-learn's `HashingVectorizer` to create deterministic,
384-dimensional vectors. It downloads no embedding model. `PgVectorKnowledgeStore`
stores those vectors and performs cosine-distance search using pgvector.

`reciprocal_rank_fusion()` combines ordered result lists by source URL. It ranks
results by position instead of comparing incompatible keyword and vector scores.

## Live HTML discovery

`HtmlWebSearch` implements the `WebSearch` protocol used by the RAG service. Its public
entry point, `search_with_activity()`, returns both accepted source dictionaries and a
diagnostic activity list.

The discovery order is:

1. Fetch URLs explicitly included in the question.
2. Query DuckDuckGo's HTML result page.
3. Retry DuckDuckGo with a compact query if the first response is challenged or empty.
4. For current-event questions, try Bing News HTML.
5. For supported entity-role questions, try a direct Wikipedia article URL.
6. Try Bing Web HTML.
7. Try Brave Web HTML.

Result links must be public HTTP or HTTPS URLs. Localhost and literal private,
loopback, link-local, multicast, and reserved IP addresses are rejected. Accepted
pages are fetched with a bounded thread pool. Page scripts and style elements are
removed before readable paragraph text and publication metadata are extracted.

Search result persistence is best-effort. It occurs only when a repository is
configured, global persistence is enabled, and the current request permits it.
Representative images are separately gated by `web_search_store_images` and are
validated for media type, a 10 MB byte limit, public URL, and maximum dimensions.

## Grounded generation

`KnowledgeRagService` is the central application service. It depends on four small
interfaces from `rag.py`:

- `ArticleSearch` for PostgreSQL full-text results.
- `VectorSearch` for pgvector results.
- `WebSearch` for optional live HTML evidence and activity diagnostics.
- `MultimodalGenerator` for complete and streaming generation.

For a normal answer, the service:

1. Determines whether a question refers to earlier conversation turns.
2. Builds a contextual retrieval query only for a detected follow-up.
3. Retrieves keyword and vector candidates from local storage.
4. Optionally retrieves live web candidates.
5. Fuses and truncates sources to the configured character budget.
6. Builds a prompt that treats retrieved content as untrusted evidence.
7. Calls the local model with the prompt and optional image.

When web search is requested for a text-only question but no source is found,
generation is skipped and a fixed insufficient-evidence answer is returned. When web
search is explicitly disabled, the model may answer from its pretrained knowledge but
must not invent citations.

For image-oriented questions, `should_retrieve_sources()` avoids unrelated text search
when the wording points only to the supplied image. Current-information wording or an
explicit URL still triggers retrieval.

### Result contracts

`AnswerSource` is the normalized citation shape used by both JSON and streaming
responses. `GenerativeAnswer` contains the answer, configured model name, sources,
optional stored image ID, and web-search activity.

`answer()` returns a complete `GenerativeAnswer`. `stream_answer()` returns sources
and search activity immediately with an iterator that yields generated text chunks.

## Local multimodal generation

`LocalMultimodalGenerator` lazily loads the configured processor and
`AutoModelForImageTextToText` from Transformers. The model is cached below the
configured Expansion-drive model directory.

`resolve_compute_device()` enforces the configured device behavior:

- `mps` requires Apple's Metal backend and fails when it is unavailable.
- `auto` uses MPS when available and otherwise uses CPU.
- `cpu` always uses CPU.

MPS inference uses float16; CPU inference uses float32. Images are converted to RGB and
submitted with the text through the model's chat template. A temperature of zero uses
deterministic generation; a positive temperature enables sampling.

Streaming uses a `TextIteratorStreamer`. Model generation runs in a daemon thread while
the caller consumes decoded text chunks. An exception in that thread is re-raised as a
runtime error after the streamer closes.

## HTTP application

`atlasmind.api.app:app` exposes three request paths:

- `GET /health` returns process health without checking dependencies.
- `POST /v1/generate` returns JSON, or delegates to streaming when `stream` is true.
- `POST /v1/generate/stream` always returns Server-Sent Events.

Pydantic validates question length, retrieval limit, generation length, temperature,
image payload length, and the optional conversation UUID. Decoded images are limited
to 10 MB and 4096 pixels on either axis.

The streaming event order is `metadata`, zero or more `token` events, and then `done`.
Failures after streaming begins are represented by an `error` event because the HTTP
status and headers have already been sent.

### Conversation behavior

`history_context()` creates a conversation UUID and deterministic chat title. If a
known conversation ID belongs to the current pseudonymous client fingerprint, recent
messages are loaded. `KnowledgeRagService.uses_conversation_context()` applies them
only when the question looks like a referential follow-up.

History is enabled by default. With `save_history=false`, the request does not persist
chat messages, the uploaded image, or web-discovery results. Operational request logs
are independent of this preference and are still written.

### Request logging

`RequestResponseLoggingMiddleware` assigns an `X-Request-ID` and records method, path,
query string, status, duration, client fingerprint, headers, bodies, errors, and
transport metadata. Authorization and cookie headers are redacted. In development,
the structured record is also sent to the Uvicorn logger.

The middleware logs complete request and response bodies, including accumulated SSE
chunks. Deployments must therefore apply suitable database access, retention, and
deletion controls.

## PostgreSQL repositories

`PostgresKnowledgeRepository` owns document, image, conversation, and logging queries.
`PostgresArticleRepository` is a compatibility alias used throughout the application.
Every operation opens a short-lived psycopg connection; connection pooling is not
implemented in the current source.

| Table | Owner | Contents |
| --- | --- | --- |
| `knowledge_documents` | `database.py` | Wikipedia and web documents plus metadata |
| `knowledge_passages` | `vector_store.py` | Chunk text and 384-dimensional vectors |
| `knowledge_images` | `database.py` | Normalized uploads and optional web images |
| `conversations` | `database.py` | Conversation title and pseudonymous client ownership |
| `chat_history` | `database.py` | Ordered user and assistant messages |
| `application_logs` | `database.py` | Structured HTTP exchanges and failures |

Initialization is idempotent and includes migrations from the older
`wikipedia_articles` and `wikipedia_passages` tables when they exist. It creates
missing objects but does not create the PostgreSQL database itself.

## Storage boundary

`ExpansionStorage` rejects roots outside `/Volumes/Expansion` and prevents child paths
from escaping the configured root after resolution. Write operations also verify that
the Expansion volume is mounted. JSONL and joblib artifacts are written to a temporary
sibling path and atomically replaced.

The PostgreSQL DSN is configured separately; database files are managed by the local
PostgreSQL installation rather than `ExpansionStorage`.

## Extension points

The protocol boundaries in `rag.py` are the safest way to extend the system:

- Implement `ArticleSearch` to replace or augment lexical retrieval.
- Implement `VectorSearch` to use a different local embedding or vector backend.
- Implement `WebSearch` to change direct-HTML discovery while preserving activity
  reporting.
- Implement `MultimodalGenerator` to use another locally loaded model.

New persistence fields require coordinated changes to `sql/schema.sql`, the matching
SQL constants and repository methods in `database.py` or `vector_store.py`, response
schemas when exposed publicly, and the relevant tests and documentation.

## Test map

| Test module | Main coverage |
| --- | --- |
| `test_crawler.py` | Topic URL construction, domain restrictions, namespaces, and link parsing |
| `test_storage.py` | Expansion-drive boundary and safe child paths |
| `test_vector_store.py` | Embedding dimensions and reciprocal-rank fusion |
| `test_rag.py` | Grounding, image routing, web fallback, streaming, and follow-up context |
| `test_web_search.py` | Search parsing, relevance, challenge handling, URLs, and fallbacks |
| `test_api_stream.py` | Server-Sent Event encoding |
| `test_history.py` | Fingerprinting, conversation IDs, and generated titles |
| `test_logging_middleware.py` | Sensitive-header redaction |

The tests are primarily isolated unit tests. They do not prove that PostgreSQL,
pgvector, the Expansion drive, model download, Apple MPS, or third-party HTML layouts
work in a particular runtime environment.

## Maintenance checklist

When changing AtlasMind:

1. Keep configuration defaults synchronized across `config.py`, `.env.example`, and
   the README.
2. Keep the FastAPI version synchronized with `pyproject.toml` and the README version.
3. Keep database DDL synchronized between `sql/schema.sql` and the executable SQL in
   the repository modules.
4. Update `docs/API.md` when request fields, response fields, events, limits, or status
   codes change.
5. Update this reference and `docs/SYSTEM_DESIGN.md` when dependencies or component
   responsibilities change.
6. Add or update isolated tests for parsing, ranking, validation, and orchestration
   behavior.
