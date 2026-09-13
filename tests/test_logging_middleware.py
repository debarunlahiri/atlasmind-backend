from atlasmind.api.logging_middleware import decoded_headers


def test_decoded_headers_redacts_credentials_and_cookies() -> None:
    headers = decoded_headers(
        [
            (b"authorization", b"Bearer secret"),
            (b"cookie", b"session=secret"),
            (b"content-type", b"application/json"),
        ]
    )

    assert headers == {
        "authorization": "[REDACTED]",
        "cookie": "[REDACTED]",
        "content-type": "application/json",
    }
