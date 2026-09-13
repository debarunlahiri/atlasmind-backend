import hashlib
import re
from typing import Optional
from uuid import UUID, uuid4

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "how",
    "is",
    "me",
    "of",
    "the",
    "to",
    "what",
    "when",
    "where",
    "why",
}


def client_fingerprint(ip_address: str, user_agent: str) -> str:
    """Create a stable pseudonymous identifier without storing the raw IP."""
    identity = f"{ip_address.strip()}\n{user_agent.strip()}".encode()
    return hashlib.sha256(identity).hexdigest()


def normalize_conversation_id(value: Optional[UUID]) -> str:
    return str(value) if value is not None else str(uuid4())


def generate_chat_name(question: str, maximum_words: int = 6) -> str:
    """Create a short deterministic title immediately from the first question."""
    words = re.findall(r"[\w'-]+", question, flags=re.UNICODE)
    meaningful = [word for word in words if word.casefold() not in STOP_WORDS]
    selected = (meaningful or words)[:maximum_words]
    title = " ".join(selected).strip().title()
    return title or "New Conversation"
