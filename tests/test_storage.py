from pathlib import Path

import pytest

from atlasmind.storage import ExpansionStorage


def test_storage_rejects_internal_drive() -> None:
    with pytest.raises(ValueError, match="Expansion"):
        ExpansionStorage(Path("/tmp/atlasmind"))


def test_storage_builds_paths_below_expansion() -> None:
    storage = ExpansionStorage(Path("/Volumes/Expansion/aiml/atlasmind"))
    assert storage.path("corpus", "articles.jsonl") == Path(
        "/Volumes/Expansion/aiml/atlasmind/corpus/articles.jsonl"
    )
