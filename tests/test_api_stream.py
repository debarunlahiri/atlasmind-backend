import json

from atlasmind.api.app import server_sent_event


def test_server_sent_event_encodes_unicode_json() -> None:
    event = server_sent_event("token", {"text": "Hello 🌍"})

    assert event.startswith("event: token\ndata: ")
    assert event.endswith("\n\n")
    payload = json.loads(event.split("data: ", 1)[1])
    assert payload == {"text": "Hello 🌍"}
