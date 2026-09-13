# AtlasMind API reference

## Server startup

Every API worker startup checks the PostgreSQL schema before accepting requests.
This includes the initial launch and every Uvicorn development reload. AtlasMind
automatically creates missing application tables, indexes, and the `vector`
extension with idempotent `IF NOT EXISTS` statements, while preserving existing
data. PostgreSQL and the configured database must already exist and be reachable.
If schema initialization fails, worker startup fails rather than serving against
an incomplete database.

## Request and response logs

Every HTTP request receives an `X-Request-ID` response header. AtlasMind writes a
matching row to PostgreSQL's `application_logs` table after the response ends.
The row contains method, path, query string, status code, duration, pseudonymous
client fingerprint, request and response headers, complete request and response
bodies, error details, and HTTP transport metadata. Streaming responses are
stored as the complete Server-Sent Events payload after streaming finishes.

Authorization, cookie, proxy-authorization, and set-cookie values are always
stored as `[REDACTED]`. Operational logging is independent of the chat-history
`save_history` option. This makes request diagnostics available even when chat
memory is disabled, so deployments must define access, retention, and deletion
rules for `application_logs`.

In `ATLASMIND_ENVIRONMENT=development`, the structured exchange is also printed
to the running server's terminal. Other environments retain database logging but
do not print full exchange bodies.

## Base URL

```text
http://127.0.0.1:8000
```

The URLs below are complete local-development URLs. Replace the base URL with the
HTTPS address of the deployed service when AtlasMind is hosted elsewhere.

## Start the development server

Run the API with automatic source-file reload, similar to Nodemon:

```bash
PYTHONPATH=src .venv/bin/python -m atlasmind.dev_server \
  --host 127.0.0.1 \
  --port 8000
```

Uvicorn watches the `src/` directory and restarts the API worker whenever Python
source changes. Use this only during development. Do not enable reload in
production.

## Authentication

Authentication is not enabled in the current version. Anonymous clients can use
every endpoint without an API key or Authorization header. For that reason, bind
the server to `127.0.0.1` only and do not expose it publicly until authentication,
TLS, and rate limiting are implemented.

## Health check

### Request

```http
GET http://127.0.0.1:8000/health
Accept: application/json
```

### cURL

```bash
curl --request GET \
  --url http://127.0.0.1:8000/health \
  --header "Accept: application/json"
```

### Full response

```http
HTTP/1.1 200 OK
content-type: application/json

{"status":"ok"}
```

## Chat history behavior

History saving is enabled by default. A new request without `conversation_id`
receives a generated UUID and a short `chat_name`. Send that UUID in later requests
to group messages into the same backend conversation.

AtlasMind stores the question, answer, sources, generated title, User-Agent/device
string, and a pseudonymous SHA-256 fingerprint derived from the connection IP and
User-Agent. It does not store the raw IP. HTTP servers cannot obtain a remote
device's MAC address, so AtlasMind does not claim to collect one.

History is backend-only in this version; there is no history-listing API. To opt
out for a request, send:

```json
{
  "question": "Do not retain this conversation",
  "save_history": false
}
```

Opting out prevents chat-message, uploaded-image, and web-result/query persistence
for that request. The response still contains a transient conversation ID and title,
with `history_saved` set to `false`.

## Generate a complete JSON response

This mode returns only after retrieval and local token generation finish. Live
result pages are fetched concurrently with an eight-second timeout by default,
and optional web-image downloads are disabled by default. The first request can
remain slower while Transformers downloads or loads the local model. Use the
streaming mode below when the client should display tokens immediately.

Live discovery first reads DuckDuckGo HTML, retries a compact query when the
result page is empty or challenged, and then falls back to Bing's public HTML
result page. These are HTML requests rather than search APIs. Irrelevant fallback
results are removed before generation. If Bing Web returns no relevant results,
AtlasMind tries Brave Web HTML. Entity-role questions may also use a direct
Wikipedia article request. CAPTCHA controls are reported and skipped rather than
bypassed.

Every response includes `search_activity`, in attempted order. Each item reports
the HTML engine, `success`, `no_results`, `challenged`, or `error`, the relevant
result count, and an optional detail. AtlasMind stops after the first configured
engine returns relevant results; otherwise it continues through the fallback
chain. Each web source also identifies its `search_engine`.

If `question` contains a plain URL or Markdown link, AtlasMind first fetches that
URL and reports `Direct URL HTML` in `search_activity`. Its title, Open Graph or
Twitter description, readable paragraphs, and publication metadata become a
source and enrich the subsequent related-coverage search. Duplicate Markdown
display and target URLs are fetched only once.

### URL

```text
POST http://127.0.0.1:8000/v1/generate
```

### Headers

```http
Content-Type: application/json
Accept: application/json
```

### Request body

```json
{
  "question": "What is artificial intelligence?",
  "limit": 5,
  "max_new_tokens": 256,
  "temperature": 0,
  "image_base64": null,
  "web_search": true,
  "stream": false,
  "save_history": true,
  "conversation_id": null
}
```

If live web search is requested but retrieval finds no local or live-web sources,
AtlasMind does not ask the local model to invent a current factual answer. It
returns an explicit insufficient-evidence message with an empty `sources` array:

```json
{
  "answer": "I could not find reliable local or live-web sources for this current question, so I cannot give a factual answer without risking invented information. Try a shorter query that includes the full names of the people and the specific event.",
  "model": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
  "sources": [],
  "image_id": null,
  "conversation_id": "2dfbc42d-67a2-40fd-bbe4-cf95335a804e",
  "chat_name": "Artificial Intelligence",
  "history_saved": true,
  "conversation_context_used": false,
  "search_activity": [
    {
      "engine": "DuckDuckGo HTML",
      "status": "challenged",
      "result_count": 0,
      "detail": "CAPTCHA detected"
    },
    {
      "engine": "Bing News HTML",
      "status": "no_results",
      "result_count": 0,
      "detail": null
    }
  ]
}
```

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `question` | string | Yes | — | Question between 2 and 2,000 characters. |
| `limit` | integer | No | `5` | Number of retrieved sources, from 1 to 20. |
| `max_new_tokens` | integer | No | `256` | Generation limit, from 32 to 1,024 tokens. |
| `temperature` | number | No | `0` | Sampling temperature from 0 to 2. |
| `image_base64` | string or null | No | `null` | Raw base64 or a base64 data URL containing an image. |
| `web_search` | boolean | No | `true` | Enables direct HTML web discovery. No search API is used. |
| `stream` | boolean | No | `false` | Set to `true` for Server-Sent Events. |
| `save_history` | boolean | No | `true` | Save this chat and associated request data. Set to `false` to opt out. |
| `conversation_id` | UUID or null | No | `null` | Continue a conversation, or omit it to create a new UUID. |

When `conversation_id` belongs to the same pseudonymous client, AtlasMind loads
up to 12 recent messages. For referential follow-ups containing pronouns or
phrases such as “What are they going to do?” and “Tell me more,” it adds the last
two user questions to retrieval and includes up to eight messages in the prompt.
A standalone question starts an independent retrieval query even when it uses the
same conversation ID. `conversation_context_used` reports whether history was
actually applied. Unknown or mismatched IDs do not expose stored messages.

### cURL

```bash
curl --request POST \
  --url http://127.0.0.1:8000/v1/generate \
  --header "Content-Type: application/json" \
  --header "Accept: application/json" \
  --data '{
    "question": "What is artificial intelligence?",
    "limit": 5,
    "max_new_tokens": 256,
    "temperature": 0,
    "image_base64": null,
    "web_search": true,
    "stream": false,
    "save_history": true,
    "conversation_id": null
  }'
```

### Full response

```http
HTTP/1.1 200 OK
content-type: application/json

{
  "answer": "Artificial intelligence is the field of building systems that perform tasks associated with human intelligence [1].",
  "model": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
  "sources": [
    {
      "number": 1,
      "title": "Artificial intelligence",
      "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
      "excerpt": "Artificial intelligence is intelligence demonstrated by machines...",
      "source_type": "wikipedia",
      "published_at": null
    }
  ],
  "image_id": null,
  "conversation_id": "2dfbc42d-67a2-40fd-bbe4-cf95335a804e",
  "chat_name": "Artificial Intelligence",
  "history_saved": true,
  "conversation_context_used": false,
  "search_activity": []
}
```

## Generate a streaming response

The main generation URL streams when `stream` is `true`. The explicit
`/v1/generate/stream` URL provides the same stream and is useful for clients that
prefer a dedicated endpoint.

### URL

```text
POST http://127.0.0.1:8000/v1/generate
```

Alternative URL:

```text
POST http://127.0.0.1:8000/v1/generate/stream
```

### Headers

```http
Content-Type: application/json
Accept: text/event-stream
```

### Request body

```json
{
  "question": "Explain how neural networks learn",
  "limit": 5,
  "max_new_tokens": 256,
  "temperature": 0,
  "image_base64": null,
  "web_search": true,
  "stream": true,
  "save_history": true,
  "conversation_id": null
}
```

### cURL

The `--no-buffer` option prints each event immediately:

```bash
curl --no-buffer --request POST \
  --url http://127.0.0.1:8000/v1/generate \
  --header "Content-Type: application/json" \
  --header "Accept: text/event-stream" \
  --data '{
    "question": "Explain how neural networks learn",
    "limit": 5,
    "max_new_tokens": 256,
    "temperature": 0,
    "image_base64": null,
    "web_search": true,
    "stream": true,
    "save_history": true,
    "conversation_id": null
  }'
```

### Full response stream

```http
HTTP/1.1 200 OK
content-type: text/event-stream; charset=utf-8
cache-control: no-cache
connection: keep-alive
x-accel-buffering: no

event: metadata
data: {"model":"HuggingFaceTB/SmolVLM2-500M-Video-Instruct","sources":[{"number":1,"title":"Artificial neural network","url":"https://en.wikipedia.org/wiki/Neural_network_(machine_learning)","excerpt":"A neural network is a machine learning model...","source_type":"wikipedia","published_at":null,"search_engine":null}],"image_id":null,"conversation_id":"2dfbc42d-67a2-40fd-bbe4-cf95335a804e","chat_name":"Neural Networks Learn","history_saved":true,"conversation_context_used":false,"search_activity":[]}

event: token
data: {"text":"Neural networks "}

event: token
data: {"text":"learn by adjusting internal weights "}

event: token
data: {"text":"to reduce prediction error [1]."}

event: done
data: {"finish_reason":"stop"}

```

An error after streaming has started is returned as the final SSE event:

```text
event: error
data: {"message":"Local model streaming failed: example error"}

```

## Generate from an image

Encode an image and send it in the request:

```bash
IMAGE_BASE64="$(base64 < /absolute/path/to/diagram.png | tr -d '\n')"

curl --request POST \
  --url http://127.0.0.1:8000/v1/generate \
  --header "Content-Type: application/json" \
  --header "Accept: application/json" \
  --data "{\"question\":\"Explain this diagram\",\"image_base64\":\"${IMAGE_BASE64}\",\"web_search\":false,\"stream\":false}"
```

Images are limited to 10 MB after base64 decoding and 4,096 pixels on either
dimension. A successful response includes the PostgreSQL `image_id`.

## Error responses

### Invalid request body

```http
HTTP/1.1 422 Unprocessable Entity
content-type: application/json

{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "question"],
      "msg": "String should have at least 2 characters",
      "input": "?",
      "ctx": {"min_length": 2}
    }
  ]
}
```

### Runtime dependency failure

```http
HTTP/1.1 503 Service Unavailable
content-type: application/json

{"detail":"The local model or PostgreSQL service is unavailable"}
```

If the detail mentions `SmolVLMProcessor`, install the updated requirements and
restart the server. SmolVLM's processor requires `num2words` and the vision stack
provided by a `torchvision` build compatible with the installed `torch` version:

```bash
.venv/bin/python -m pip install -r requirements.txt
```
