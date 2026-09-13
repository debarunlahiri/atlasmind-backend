# AtlasMind AI

Current source version: `0.11.0`

AtlasMind AI is a general knowledge and multimodal RAG application backed by
PostgreSQL full-text search and a local `pgvector` database. Wikipedia crawling
provides its reusable base corpus. At question time it can also search public web
result pages, read the linked HTML pages, save their text and representative images
in PostgreSQL, and combine that current evidence with local knowledge.

All discovery is direct HTML browsing. AtlasMind uses DuckDuckGo's non-JavaScript
HTML results page and then reads public source pages; it does not call Wikipedia,
DuckDuckGo, Brave, Google, Bing, or another search API. No API key is required.
The local hashing embedder and local model also require no hosted API.

## Features

- Text and image generation using a locally loaded open-source multimodal model.
- Apple Metal acceleration through PyTorch MPS.
- Direct Wikipedia and web-page HTML crawling without a search API.
- PostgreSQL full-text search and local `pgvector` retrieval.
- PostgreSQL persistence for knowledge documents and validated images.
- Complete JSON responses and incremental Server-Sent Events responses.
- Backend chat history with generated conversation names and per-request opt-out.
- A separate [API reference](docs/API.md) with URLs, bodies, headers, responses,
  and cURL examples.

## Multimodal model

Generation uses the open-source
[`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
vision-language model under the Apache-2.0 license. The 500M variant was selected
for this Apple Silicon Mac with 16 GB unified memory because it leaves substantially
more memory headroom than the 2.2B checkpoint while supporting both text and images.

The model runs inside this process through PyTorch and Hugging Face Transformers.
It does not send questions or images to an inference provider. The first generation
downloads the model files unless they have already been cached. Files are cached at:

```text
/Volumes/Expansion/aiml/atlasmind/models/huggingface/
```

After the model is cached, set this to prevent any model download attempt:

```text
ATLASMIND_MODEL_LOCAL_FILES_ONLY=true
```

## Apple Metal GPU

The default device is `mps`, PyTorch's Apple Metal backend:

```text
ATLASMIND_COMPUTE_DEVICE=mps
```

The model and inference tensors are moved to the MPS device and use float16 on
Apple Metal to reduce unified-memory usage. AtlasMind stops with a clear error if
MPS is requested but unavailable. Set the device to `cpu` only when Metal cannot
be used, or to `auto` when automatic local CPU fallback is wanted.

## Storage

Generated data is restricted to the Expansion drive:

```text
/Volumes/Expansion/aiml/atlasmind/
├── corpus/wikipedia_articles.jsonl
├── indexes/wikipedia_tfidf.joblib
└── models/
    ├── huggingface/
    └── topic_classifier.joblib
```

General knowledge documents, crawled Wikipedia pages, web-page text, downloaded
representative images, and 384-dimensional passage vectors are stored in local
PostgreSQL:

```text
postgresql://debarunlahiri@localhost:5432/atlasmind
```

## Environment setup

The source supports Python 3.9 and newer. `requirements.txt` now contains every
runtime and development dependency with the same version ranges declared in
`pyproject.toml`. Create the virtual environment and install them with the project
interpreter:

```bash
/opt/homebrew/bin/python3.14 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
```

The manifest includes:

- Crawling: `beautifulsoup4` and `httpx`.
- API and streaming: `fastapi` and `uvicorn[standard]`.
- PostgreSQL and vectors: `psycopg[binary]` and `pgvector`.
- Machine learning: `numpy`, `scikit-learn`, and `joblib`.
- Multimodal generation: `torch`, `torchvision`, `transformers`, `pillow`, and
  `num2words`.
- Development validation: `pytest`, `ruff`, and `types-beautifulsoup4`.

Installing the requirements does not download the configured model immediately.
The model is downloaded on the first generation request unless it already exists
in the configured cache.

If the API reports that it cannot import or load `SmolVLMProcessor`, synchronize
the environment and restart the automatically reloading server. The processor
requires both `torchvision` and `num2words`:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

`torch` and `torchvision` must be compatible builds. Both are declared together in
the dependency manifests so pip can resolve a matching pair.

## Configuration

Create `.env` from `.env.example` and review these settings:

```text
ATLASMIND_STORAGE_ROOT=/Volumes/Expansion/aiml/atlasmind
ATLASMIND_WIKIPEDIA_USER_AGENT=AtlasMindCrawler/0.3 (your-contact@example.com)
ATLASMIND_REQUEST_TIMEOUT_SECONDS=20
ATLASMIND_CRAWL_DELAY_SECONDS=1
ATLASMIND_POSTGRES_DSN=postgresql://debarunlahiri@localhost:5432/atlasmind
ATLASMIND_COMPUTE_DEVICE=mps
ATLASMIND_MULTIMODAL_MODEL_ID=HuggingFaceTB/SmolVLM2-500M-Video-Instruct
ATLASMIND_MODEL_LOCAL_FILES_ONLY=false
ATLASMIND_GENERATOR_MAX_CONTEXT_CHARACTERS=4000
ATLASMIND_WEB_SEARCH_USER_AGENT=AtlasMindWebCrawler/0.7 (your-contact@example.com)
ATLASMIND_WEB_SEARCH_TIMEOUT_SECONDS=8
ATLASMIND_WEB_SEARCH_MAX_WORKERS=5
ATLASMIND_WEB_SEARCH_STORE_RESULTS=true
ATLASMIND_WEB_SEARCH_STORE_IMAGES=false
ATLASMIND_WEB_SEARCH_FRESHNESS=pm
```

Never commit credentials in `.env`.

## Initialize PostgreSQL

PostgreSQL must have the `pgvector` extension available:

```bash
createdb -U debarunlahiri atlasmind
PYTHONPATH=src .venv/bin/python -m atlasmind.cli init-db
```

The schema is in `sql/schema.sql`.

## Crawl Wikipedia

Use at least two labels if you want to train the classifier:

```bash
PYTHONPATH=src .venv/bin/python -m atlasmind.cli crawl \
  --topics "artificial intelligence" "computer science" "world history" \
  --max-pages-per-seed 30
```

The crawler remains on the selected Wikipedia host, checks `robots.txt`, waits
between requests, extracts readable paragraphs, follows eligible article links,
deduplicates pages, and prints live crawl progress.

## Train retrieval and classification

```bash
PYTHONPATH=src .venv/bin/python -m atlasmind.cli train
```

This trains the topic classifier, prepares the local retrieval index, and stores
local hashing vectors in PostgreSQL. The open-source multimodal generator is already
pretrained, so there is no `train-generator` step.

## Search

```bash
PYTHONPATH=src .venv/bin/python -m atlasmind.cli search \
  "How do neural networks learn?" --mode hybrid --limit 5
```

Available modes are `keyword`, `vector`, and `hybrid`.

This command searches data already stored in PostgreSQL. Live HTML web discovery
runs during `generate` unless `--no-web` is passed.

## Fresh web search without an API

```bash
PYTHONPATH=src .venv/bin/python -m atlasmind.cli generate \
  "What changed in Apple machine learning this week?" \
  --freshness pw \
  --limit 5
```

Freshness values are `pd` (day), `pw` (week), `pm` (month), and `py` (year).
AtlasMind requests the HTML search page, extracts result links, and reads the
linked HTML documents concurrently with a bounded worker pool. It uses readable
page text as grounded context and records the query, URL, title, text, publication
date when exposed, and collection timestamp. Representative
`og:image`/`twitter:image` downloading is disabled by default because it adds a
network request and database write per result. Enable it with
`ATLASMIND_WEB_SEARCH_STORE_IMAGES=true` only when those images are required.

DuckDuckGo can return an automated-traffic challenge instead of result markup.
AtlasMind detects the resulting empty search, retries with a compact query, and
then uses Bing News HTML for current-news questions before trying Bing's general
public HTML results page. This remains direct HTML browsing: none of these paths
uses a search API or API key. A relevance filter rejects fallback results that do
not match enough meaningful terms from the question. AtlasMind does not attempt
to defeat CAPTCHA controls; it switches providers when one is challenged.

The API response exposes this fallback process in `search_activity`, including
each attempted engine, its status, relevant-result count, and diagnostic detail.
Web entries in `sources` include the engine that discovered them. The configured
chain currently consists of DuckDuckGo HTML, Bing News HTML for current-news
queries, direct Wikipedia article HTML for recognizable entity-role questions,
Bing Web HTML, and Brave Web HTML. It stops when a source supplies relevant
results and continues when the current source is challenged or unsuccessful.

Questions can contain plain or Markdown-formatted HTTP/HTTPS URLs. AtlasMind
deduplicates them, fetches each page's ordinary HTML, extracts its title, Open
Graph or Twitter description, readable paragraphs, and publication metadata, and
returns it as a `Direct URL HTML` source. The discovered text also enriches the
related-coverage search. A failed URL fetch appears in `search_activity` and does
not prevent the normal fallback chain.

Web access is best-effort: sites may block automated requests, omit dates or images,
require JavaScript, or change their HTML. Respect each site's robots rules, terms,
copyright, and rate limits. Disable live browsing with `--no-web` or by setting
`web_search` to `false` in the REST request.

Live web operations use an eight-second timeout by default and open at most five
result pages concurrently. Tune `ATLASMIND_WEB_SEARCH_TIMEOUT_SECONDS` and
`ATLASMIND_WEB_SEARCH_MAX_WORKERS` for the local network. The first generation
request can still be slower while Transformers downloads or loads the model. Use
`"stream": true` to see tokens as they are generated; with `"stream": false`,
the client remains blank until retrieval and token generation both finish.

If live web search is requested but both PostgreSQL and HTML retrieval return no
sources, AtlasMind returns an explicit insufficient-evidence response instead of
letting the local model invent current facts. When web search is intentionally
disabled, the local model can still answer from model knowledge without claiming
retrieval citations. Image questions can still use the supplied image as evidence.

## Generate from text

The first call downloads the configured open-source model to the Expansion drive.
Later calls use the cached local files:

```bash
PYTHONPATH=src .venv/bin/python -m atlasmind.cli generate \
  "How do neural networks learn representations?" --limit 5
```

## Generate from text and an image

Pass a local JPEG, PNG, WebP, or other Pillow-supported image:

```bash
PYTHONPATH=src .venv/bin/python -m atlasmind.cli generate \
  "Explain what this diagram shows" \
  --image /absolute/path/to/diagram.png \
  --limit 5
```

The image is processed locally. It is not uploaded to Hugging Face or another
inference service.

## REST API

The complete endpoint, header, body, cURL, JSON-response, streaming-response, and
error reference is in [docs/API.md](docs/API.md).

Authentication is not enabled yet, so anonymous clients can currently call the
API. Bind it only to `127.0.0.1`; do not expose it publicly until authentication,
TLS, and rate limiting are implemented.

### Backend chat history and privacy

Chat history is saved to PostgreSQL by default. AtlasMind creates a conversation
UUID, generates a short title from the first question, and stores the user and
assistant messages in `conversations` and `chat_history`. History remains backend
only; this version intentionally provides no endpoint for listing it.

Supplying the returned `conversation_id` on a later request loads recent messages
for the same pseudonymous client. AtlasMind applies those messages only when the
new question looks like a referential follow-up, such as “What are they going to
do?” or “Tell me more.” Standalone questions search independently even inside the
same conversation, preventing an earlier topic from polluting retrieval. The
response returns `conversation_context_used: true` only when stored context was
actually applied and preserves the original chat name.

HTTP does not expose a remote device's MAC address. AtlasMind therefore derives a
pseudonymous client fingerprint from the connection IP and User-Agent/device string
and does not store the raw IP address. This identifier is not authentication and
can change when the client's network or browser details change.

Clients can disable conversation-history and retrieval-result persistence with:

```json
{
  "save_history": false
}
```

For an opted-out request, AtlasMind does not store chat messages, the uploaded
image, or live-web result/query records. Retrieval and generation still work.

Operational API logs are separate from chat history. Every HTTP exchange is
recorded in PostgreSQL's `application_logs` table for diagnostics, including its
request ID, method, path, query string, status, duration, pseudonymous client
fingerprint, headers, request body, response headers, response body, errors, and
transport metadata. Authorization and cookie header values are replaced with
`[REDACTED]`. The response includes an `X-Request-ID` header that matches the
database record. Because these diagnostic logs contain request and response
bodies, configure a retention and deletion policy before any public deployment.

When `ATLASMIND_ENVIRONMENT=development`, the same structured API exchange is
printed to the server terminal after the response finishes. For an SSE response,
the saved and printed response body contains the complete emitted event stream.
Production environments continue writing database logs but do not print full
exchange bodies to the terminal.

### Development server with automatic reload

Start the development API after PostgreSQL itself is running and the configured
database exists:

```bash
PYTHONPATH=src .venv/bin/python -m atlasmind.dev_server
```

The development server watches `src/`. When a Python source file changes, Uvicorn
restarts the worker automatically, similar to Nodemon in Node.js. The default URL
is `http://127.0.0.1:8000`; override it with `--host` or `--port` when needed:

```bash
PYTHONPATH=src .venv/bin/python -m atlasmind.dev_server \
  --host 127.0.0.1 \
  --port 8000
```

After installing the package, the shorter command is:

```bash
atlasmind-dev --host 127.0.0.1 --port 8000
```

On every worker startup—including the first launch and each automatic reload—
AtlasMind runs idempotent `CREATE EXTENSION IF NOT EXISTS`, `CREATE TABLE IF NOT
EXISTS`, and `CREATE INDEX IF NOT EXISTS` statements. Missing application tables
and indexes are created automatically; existing data is left intact. PostgreSQL
must already be reachable, the target database must already exist, and its role
must have permission to create the `vector` extension and schema objects. A
startup initialization failure prevents the API from accepting requests instead
of leaving it partially initialized.

### Server without automatic reload

Use the normal Uvicorn command for a stable non-reloading process:

```bash
PYTHONPATH=src .venv/bin/python -m uvicorn atlasmind.api.app:app \
  --host 127.0.0.1 --port 8000
```

Do not enable automatic reload in production.

Text-only request:

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is artificial intelligence?",
    "limit": 5,
    "max_new_tokens": 256,
    "temperature": 0
  }'
```

For image input, include the image bytes as base64, either raw or with a data URL
prefix, in `image_base64`:

```json
{
  "question": "What does this image show?",
  "image_base64": "data:image/png;base64,iVBORw0KGgo...",
  "limit": 5,
  "max_new_tokens": 256,
  "temperature": 0
}
```

Decoded images are limited to 10 MB and 4096 pixels on either dimension. The model
loads lazily on the first generation request rather than during API startup.

### Streaming responses

Use the Server-Sent Events endpoint for ChatGPT-style incremental output:

```bash
curl -N -X POST http://127.0.0.1:8000/v1/generate \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "question": "Explain how neural networks learn",
    "limit": 5,
    "max_new_tokens": 256,
    "temperature": 0,
    "web_search": true,
    "stream": true
  }'
```

When `stream` is `true`, the endpoint uses `text/event-stream` and emits these events:

```text
event: metadata
data: {"model":"...","sources":[...],"image_id":null}

event: token
data: {"text":"Neural"}

event: token
data: {"text":" networks learn..."}

event: done
data: {"finish_reason":"stop"}
```

If generation fails after response headers have been sent, the final event is
`error` with a JSON `message`. Omit `stream` or set it to `false` for one complete
JSON response. `/v1/generate/stream` is also available as an explicit streaming
endpoint for clients that prefer separate URLs.

## Online and offline behavior

| Operation | Offline after setup? | External access |
| --- | --- | --- |
| Crawl new Wikipedia pages | No | Direct Wikipedia HTML requests |
| Fresh web search | No | Direct search-result and source-page HTML requests |
| Train classifier and indexes | Yes | None |
| First model load | No | Downloads open-source model files |
| Later text or image generation | Yes | None with local-only mode enabled |
| Search PostgreSQL corpus | Yes | None when PostgreSQL is local |
| Serve the REST API | Yes | None when bound locally and model is cached |

## Important limitations

- SmolVLM2 is a pretrained third-party open-source model; AtlasMind does not own its
  architecture, tokenizer, training data, or base weights.
- The model can produce inaccurate answers. Wikipedia retrieval and source links help
  grounding but do not guarantee correctness. Fresh web content is also untrusted and
  may be incomplete, malicious, or wrong.
- A 500M model prioritizes memory usage and local speed over the quality of larger
  multimodal models.
- Image understanding, MPS execution, generation quality, dependency compatibility,
  PostgreSQL connectivity, and model download were not runtime-tested during this
  source-only change.
- Crawling must follow Wikimedia's current robots rules, terms, rate guidance,
  attribution requirements, and content licenses. The same obligation applies to
  every site reached through live HTML search.
- Keep the API bound to `127.0.0.1` because anonymous access is currently enabled.
  Add authentication, TLS, rate limiting, and audit logging before exposing it on
  a network.
