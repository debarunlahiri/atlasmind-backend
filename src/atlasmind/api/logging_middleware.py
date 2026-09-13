import json
import logging
from collections.abc import Callable
from time import perf_counter
from typing import Any
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from atlasmind.database import PostgresArticleRepository
from atlasmind.history import client_fingerprint

LOGGER = logging.getLogger("uvicorn.error")
SENSITIVE_HEADERS = {"authorization", "cookie", "proxy-authorization", "set-cookie"}


def decoded_headers(raw_headers: list[tuple[bytes, bytes]]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for raw_name, raw_value in raw_headers:
        name = raw_name.decode("latin-1").lower()
        value = raw_value.decode("latin-1")
        headers[name] = "[REDACTED]" if name in SENSITIVE_HEADERS else value
    return headers


class RequestResponseLoggingMiddleware:
    """Persist complete HTTP exchanges and print them in development."""

    def __init__(
        self,
        app: ASGIApp,
        repository_factory: Callable[[], PostgresArticleRepository],
        environment: str,
    ) -> None:
        self.app = app
        self.repository_factory = repository_factory
        self.development = environment.lower() == "development"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started_at = perf_counter()
        request_id = str(uuid4())
        request_parts: list[bytes] = []
        response_parts: list[bytes] = []
        status_code: int | None = None
        response_headers: dict[str, str] = {}
        error_detail: str | None = None

        received_messages: list[Message] = []
        while True:
            message = await receive()
            received_messages.append(message)
            if message["type"] != "http.request" or not message.get("more_body", False):
                break
        request_parts.extend(
            message.get("body", b"")
            for message in received_messages
            if message["type"] == "http.request"
        )
        replay_index = 0

        async def capture_receive() -> Message:
            nonlocal replay_index
            if replay_index < len(received_messages):
                replayed_message = received_messages[replay_index]
                replay_index += 1
                return replayed_message
            return await receive()

        async def capture_send(message: Message) -> None:
            nonlocal status_code, response_headers
            if message["type"] == "http.response.start":
                status_code = message["status"]
                mutable_headers = list(message.get("headers", []))
                mutable_headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = mutable_headers
                response_headers = decoded_headers(mutable_headers)
            elif message["type"] == "http.response.body":
                response_parts.append(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, capture_receive, capture_send)
        except Exception as error:
            error_detail = f"{type(error).__name__}: {error}"
            raise
        finally:
            headers = decoded_headers(scope.get("headers", []))
            client = scope.get("client")
            client_ip = str(client[0]) if client else "unknown"
            entry: dict[str, Any] = {
                "request_id": request_id,
                "level": "ERROR" if error_detail or (status_code or 500) >= 500 else "INFO",
                "event": (
                    "http_request_completed" if error_detail is None else "http_request_failed"
                ),
                "method": scope.get("method", ""),
                "path": scope.get("path", ""),
                "query_string": scope.get("query_string", b"").decode("utf-8", "replace"),
                "status_code": status_code,
                "duration_ms": round((perf_counter() - started_at) * 1000, 3),
                "client_fingerprint": client_fingerprint(
                    client_ip,
                    headers.get("user-agent", "unknown"),
                ),
                "request_headers": headers,
                "request_body": b"".join(request_parts).decode("utf-8", "replace"),
                "response_headers": response_headers,
                "response_body": b"".join(response_parts).decode("utf-8", "replace"),
                "error_detail": error_detail,
                "metadata": {
                    "http_version": scope.get("http_version"),
                    "scheme": scope.get("scheme"),
                    "server": scope.get("server"),
                },
            }
            if self.development:
                LOGGER.info("API exchange: %s", json.dumps(entry, ensure_ascii=False))
            try:
                self.repository_factory().record_application_log(entry)
            except Exception:
                LOGGER.exception("Could not persist API log %s", request_id)
