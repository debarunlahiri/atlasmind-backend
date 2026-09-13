import base64
import binascii
import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import lru_cache
from io import BytesIO

from fastapi import FastAPI, HTTPException, Request
from PIL import Image, UnidentifiedImageError
from starlette.responses import StreamingResponse

from atlasmind.api.logging_middleware import RequestResponseLoggingMiddleware
from atlasmind.api.schemas import GenerateRequest, GenerateResponse
from atlasmind.config import get_settings
from atlasmind.database import PostgresArticleRepository
from atlasmind.generation import LocalMultimodalGenerator
from atlasmind.history import (
    client_fingerprint,
    generate_chat_name,
    normalize_conversation_id,
)
from atlasmind.rag import KnowledgeRagService
from atlasmind.vector_store import LocalTextEmbedder, PgVectorKnowledgeStore
from atlasmind.web_search import HtmlWebSearch


@lru_cache
def knowledge_repository() -> PostgresArticleRepository:
    settings = get_settings()
    return PostgresArticleRepository(settings.postgres_dsn)


@lru_cache
def vector_knowledge_store() -> PgVectorKnowledgeStore:
    settings = get_settings()
    return PgVectorKnowledgeStore(settings.postgres_dsn, LocalTextEmbedder())


@lru_cache
def rag_service() -> KnowledgeRagService:
    settings = get_settings()
    articles = knowledge_repository()
    vectors = vector_knowledge_store()
    generator = LocalMultimodalGenerator(
        settings.multimodal_model_id,
        settings.huggingface_cache_dir,
        settings.compute_device,
        settings.model_local_files_only,
    )
    web_search = HtmlWebSearch(
        settings.web_search_user_agent,
        settings.web_search_timeout_seconds,
        articles,
        settings.web_search_store_results,
        settings.web_search_store_images,
        settings.web_search_max_workers,
    )
    return KnowledgeRagService(
        articles,
        vectors,
        generator,
        settings.generator_max_context_characters,
        web_search,
        settings.web_search_freshness,
    )


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Idempotently ensure every required table exists on each worker start."""
    knowledge_repository().initialize()
    vector_knowledge_store().initialize()
    yield


app = FastAPI(
    title="AtlasMind Generative API",
    version="0.11.0",
    description="Grounded text and image answers from local knowledge and fresh web pages.",
    lifespan=lifespan,
)
app.add_middleware(
    RequestResponseLoggingMiddleware,
    repository_factory=knowledge_repository,
    environment=get_settings().environment,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def decode_image(image_base64: str) -> Image.Image:
    encoded = image_base64.split(",", 1)[-1]
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("image_base64 is not valid base64 data") from error
    if len(image_bytes) > 10 * 1024 * 1024:
        raise ValueError("The decoded image must not exceed 10 MB")
    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError("image_base64 does not contain a supported image") from error
    if image.width > 4096 or image.height > 4096:
        raise ValueError("Image width and height must not exceed 4096 pixels")
    return image.convert("RGB")


def history_context(
    request: GenerateRequest,
    http_request: Request,
) -> tuple[str, str, str, str, list[dict[str, str]]]:
    conversation_id = normalize_conversation_id(request.conversation_id)
    chat_name = generate_chat_name(request.question)
    user_agent = http_request.headers.get("user-agent", "unknown")
    client_ip = http_request.client.host if http_request.client is not None else "unknown"
    fingerprint = client_fingerprint(client_ip, user_agent)
    conversation_history: list[dict[str, str]] = []
    if request.conversation_id is not None:
        stored_title, conversation_history = knowledge_repository().conversation_context(
            conversation_id,
            fingerprint,
        )
        if stored_title is not None:
            chat_name = stored_title
    return conversation_id, chat_name, fingerprint, user_agent, conversation_history


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest, http_request: Request) -> object:
    if request.stream:
        return generate_stream(request, http_request)
    try:
        conversation_id, chat_name, fingerprint, user_agent, conversation_history = (
            history_context(request, http_request)
        )
        context_used = rag_service().uses_conversation_context(
            request.question,
            conversation_history,
        )
        effective_history = conversation_history if context_used else []
        image = decode_image(request.image_base64) if request.image_base64 else None
        image_id = (
            knowledge_repository().save_image(image)
            if image is not None and request.save_history
            else None
        )
        answer = rag_service().answer(
            request.question,
            request.limit,
            request.max_new_tokens,
            request.temperature,
            image,
            request.web_search,
            image_id,
            request.save_history,
            effective_history,
        )
        if request.save_history:
            knowledge_repository().record_chat(
                conversation_id,
                fingerprint,
                chat_name,
                user_agent,
                request.question,
                answer.answer,
                [asdict(source) for source in answer.sources],
                image_id,
            )
        response = answer.as_dict()
        response.update(
            {
                "conversation_id": conversation_id,
                "chat_name": chat_name,
                "history_saved": request.save_history,
                "conversation_context_used": context_used,
            }
        )
        return response
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def server_sent_event(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/v1/generate/stream")
def generate_stream(request: GenerateRequest, http_request: Request) -> StreamingResponse:
    try:
        conversation_id, chat_name, fingerprint, user_agent, conversation_history = (
            history_context(request, http_request)
        )
        context_used = rag_service().uses_conversation_context(
            request.question,
            conversation_history,
        )
        effective_history = conversation_history if context_used else []
        image = decode_image(request.image_base64) if request.image_base64 else None
        image_id = (
            knowledge_repository().save_image(image)
            if image is not None and request.save_history
            else None
        )
        sources, chunks, search_activity = rag_service().stream_answer(
            request.question,
            request.limit,
            request.max_new_tokens,
            request.temperature,
            image,
            request.web_search,
            request.save_history,
            effective_history,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    def events() -> Iterator[str]:
        yield server_sent_event(
            "metadata",
            {
                "model": rag_service().generator.model_name,
                "sources": [asdict(source) for source in sources],
                "image_id": image_id,
                "conversation_id": conversation_id,
                "chat_name": chat_name,
                "history_saved": request.save_history,
                "conversation_context_used": context_used,
                "search_activity": search_activity,
            },
        )
        answer_parts: list[str] = []
        try:
            for chunk in chunks:
                answer_parts.append(chunk)
                yield server_sent_event("token", {"text": chunk})
            if request.save_history:
                knowledge_repository().record_chat(
                    conversation_id,
                    fingerprint,
                    chat_name,
                    user_agent,
                    request.question,
                    "".join(answer_parts),
                    [asdict(source) for source in sources],
                    image_id,
                )
            yield server_sent_event("done", {"finish_reason": "stop"})
        except Exception as error:
            yield server_sent_event("error", {"message": str(error)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
