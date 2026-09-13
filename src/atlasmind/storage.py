import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class ExpansionStorage:
    """Restrict all generated artifacts to one configured external-drive root."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        expansion_root = Path("/Volumes/Expansion").resolve()
        if self.root != expansion_root and expansion_root not in self.root.parents:
            raise ValueError("Generated data must be stored on /Volumes/Expansion")

    def path(self, *parts: str) -> Path:
        candidate = self.root.joinpath(*parts).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("Storage path must remain inside the configured root")
        return candidate

    def require_available(self) -> None:
        volume = Path("/Volumes/Expansion")
        if not volume.is_dir():
            raise RuntimeError(
                "Expansion drive is unavailable at /Volumes/Expansion. "
                "Connect it before collecting or training."
            )

    def prepare(self, *parts: str) -> Path:
        self.require_available()
        directory = self.path(*parts)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def write_jsonl(self, relative_path: str, rows: Iterable[Any]) -> Path:
        destination = self.path(relative_path)
        self.prepare(*Path(relative_path).parent.parts)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                if is_dataclass(row):
                    if isinstance(row, type):
                        raise TypeError("A dataclass class cannot be written as a JSONL row")
                    payload = asdict(row)
                else:
                    payload = row
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        temporary.replace(destination)
        return destination

    def read_jsonl(self, relative_path: str) -> Iterator[dict[str, Any]]:
        with self.path(relative_path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    def read_jsonl_if_exists(self, relative_path: str) -> Iterator[dict[str, Any]]:
        source = self.path(relative_path)
        if source.is_file():
            yield from self.read_jsonl(relative_path)
