from uuid import UUID

from atlasmind.history import client_fingerprint, generate_chat_name, normalize_conversation_id


def test_client_fingerprint_is_stable_and_does_not_contain_raw_values() -> None:
    fingerprint = client_fingerprint("203.0.113.10", "Example Browser")

    assert fingerprint == client_fingerprint("203.0.113.10", "Example Browser")
    assert "203.0.113.10" not in fingerprint
    assert "Example Browser" not in fingerprint


def test_generate_chat_name_uses_meaningful_question_words() -> None:
    assert generate_chat_name("What is artificial intelligence?") == "Artificial Intelligence"


def test_normalize_conversation_id_creates_uuid() -> None:
    assert UUID(normalize_conversation_id(None))
