from collections.abc import Iterator
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from atlasmind.generation import GeneratedText
from atlasmind.rag import NO_WEB_EVIDENCE_ANSWER, KnowledgeRagService


class FakeArticles:
    def search(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        return [
            {
                "title": "Artificial intelligence",
                "source_url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
                "snippet": "Artificial intelligence enables machines to perform tasks.",
            }
        ]


class FakeVectors:
    def search(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        return []


@dataclass
class FakeGenerator:
    model_name: str = "local-test-model"
    prompt: str = ""
    image: Optional[Image.Image] = None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        image: Optional[Image.Image] = None,
    ) -> GeneratedText:
        self.prompt = prompt
        self.image = image
        return GeneratedText("Machines can perform intelligent tasks [1].", self.model_name)

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        image: Optional[Image.Image] = None,
    ) -> Iterator[str]:
        self.prompt = prompt
        self.image = image
        return iter(("Machines can ", "perform intelligent tasks [1]."))


def test_rag_answer_is_grounded_and_returns_sources() -> None:
    generator = FakeGenerator()
    service = KnowledgeRagService(FakeArticles(), FakeVectors(), generator)

    answer = service.answer("What is artificial intelligence?")

    assert answer.model == "local-test-model"
    assert answer.sources[0].number == 1
    assert answer.sources[0].url.endswith("Artificial_intelligence")
    assert "Use the retrieved knowledge" in generator.prompt
    assert "[1] Artificial intelligence" in generator.prompt


class EmptySearch:
    def search(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        return []


def test_rag_sends_an_image_to_the_multimodal_model_without_sources() -> None:
    generator = FakeGenerator()
    service = KnowledgeRagService(EmptySearch(), EmptySearch(), generator)
    image = Image.new("RGB", (4, 4))

    answer = service.answer("What is shown?", image=image)

    assert answer.sources == []
    assert generator.image is image
    assert "Use the attached image as visual evidence" in generator.prompt


def test_rag_uses_model_knowledge_when_retrieval_is_empty() -> None:
    generator = FakeGenerator()
    service = KnowledgeRagService(EmptySearch(), EmptySearch(), generator)

    answer = service.answer("What is artificial intelligence?", use_web_search=False)

    assert answer.answer == "Machines can perform intelligent tasks [1]."
    assert answer.sources == []
    assert "No relevant source was retrieved" in generator.prompt


def test_rag_refuses_current_answer_when_live_retrieval_is_empty() -> None:
    generator = FakeGenerator()
    service = KnowledgeRagService(EmptySearch(), EmptySearch(), generator)

    answer = service.answer("What is the latest AI news?", use_web_search=True)

    assert answer.answer == NO_WEB_EVIDENCE_ANSWER
    assert answer.sources == []
    assert generator.prompt == ""


def test_rag_streams_chunks_with_sources() -> None:
    generator = FakeGenerator()
    service = KnowledgeRagService(FakeArticles(), FakeVectors(), generator)

    sources, chunks, search_activity = service.stream_answer("What is artificial intelligence?")

    assert sources[0].number == 1
    assert list(chunks) == ["Machines can ", "perform intelligent tasks [1]."]
    assert search_activity == []


def test_rag_includes_previous_conversation_in_prompt() -> None:
    generator = FakeGenerator()
    service = KnowledgeRagService(FakeArticles(), FakeVectors(), generator)
    history = [
        {"role": "user", "content": "What did the AI leaders agree about?"},
        {"role": "assistant", "content": "They discussed slowing AI development [1]."},
    ]

    service.answer("What are they going to do?", conversation_history=history)

    assert "Previous conversation:" in generator.prompt
    assert "What did the AI leaders agree about?" in generator.prompt
    assert "What are they going to do?" in generator.prompt


def test_rag_does_not_mix_history_into_a_new_standalone_topic() -> None:
    generator = FakeGenerator()
    service = KnowledgeRagService(FakeArticles(), FakeVectors(), generator)
    history = [
        {"role": "user", "content": "What did Elon Musk say about AI?"},
        {"role": "assistant", "content": "He discussed AI development [1]."},
    ]

    service.answer("Who is the CEO of Trump Tower?", conversation_history=history)

    assert "Previous conversation:" not in generator.prompt
    assert "What did Elon Musk say about AI?" not in generator.prompt


def test_context_detection_requires_a_referential_follow_up() -> None:
    history = [{"role": "user", "content": "Tell me about Dario Amodei."}]

    assert KnowledgeRagService.uses_conversation_context(
        "What are they going to do?",
        history,
    )
    assert not KnowledgeRagService.uses_conversation_context(
        "Who is the CEO of Trump Tower?",
        history,
    )
