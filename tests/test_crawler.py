import pytest
from bs4 import BeautifulSoup

from atlasmind.crawler import CrawlSeed, WikipediaCrawler


def test_topic_creates_seed_without_external_search() -> None:
    seed = CrawlSeed.from_topic("ai")
    assert seed.label == "ai"
    assert seed.url.endswith("/Artificial_intelligence")


def test_multiword_topic_is_encoded_as_article_path() -> None:
    seed = CrawlSeed.from_topic("computer science")
    assert seed.url == "https://en.wikipedia.org/wiki/Computer_science"


def test_crawler_rejects_non_wikipedia_urls() -> None:
    with pytest.raises(ValueError, match="Wikipedia"):
        WikipediaCrawler._validate_article_url("https://example.com/wiki/AI")


def test_crawler_rejects_special_namespaces() -> None:
    with pytest.raises(ValueError, match="special namespaces"):
        WikipediaCrawler._validate_article_url("https://en.wikipedia.org/wiki/Special:Random")


def test_article_links_accept_current_absolute_wikipedia_markup() -> None:
    html = """
    <div>
        <a href="https://en.wikipedia.org/wiki/Machine_learning">ML</a>
        <a href="/wiki/Computer_vision">Vision</a>
        <a href="https://en.wikipedia.org/wiki/Category:Computing">Category</a>
        <a href="#History">Section</a>
        <a href="https://example.com/wiki/External">External</a>
    </div>
    """
    content = BeautifulSoup(html, "html.parser").div
    assert content is not None

    links = WikipediaCrawler("AtlasMind test")._article_links(
        content,
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
    )

    assert links == [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Computer_vision",
    ]
