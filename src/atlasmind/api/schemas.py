from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)
    max_new_tokens: int = Field(default=256, ge=32, le=1024)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    image_base64: Optional[str] = Field(
        default=None,
        max_length=14_000_000,
        description="Optional base64-encoded image, with or without a data URL prefix",
    )
    web_search: bool = True
    stream: bool = False
    save_history: bool = True
    conversation_id: Optional[UUID] = None


class SourceResponse(BaseModel):
    number: int
    title: str
    url: str
    excerpt: str
    source_type: str
    published_at: Optional[str] = None
    search_engine: Optional[str] = None


class SearchActivityResponse(BaseModel):
    engine: str
    status: str
    result_count: int
    detail: Optional[str] = None


class GenerateResponse(BaseModel):
    answer: str
    model: str
    sources: list[SourceResponse]
    image_id: Optional[str] = None
    conversation_id: str
    chat_name: str
    history_saved: bool
    conversation_context_used: bool
    search_activity: list[SearchActivityResponse]
