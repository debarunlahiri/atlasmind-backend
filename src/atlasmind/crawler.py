from __future__ import annotations

import hashlib
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import quote, urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup, Tag


@dataclass(frozen=True)
class CrawlSeed:
    label: str
    url: str

    @classmethod
    def from_topic(cls, topic: str, language: str = "en") -> CrawlSeed:
        normalized_topic = " ".join(topic.split()).strip()
        if not normalized_topic:
            raise ValueError("Wikipedia topics cannot be empty")
        if not re.fullmatch(r"[a-z][a-z0-9-]{1,11}", language):
            raise ValueError("Wikipedia language must be a valid subdomain code")
        aliases = {
            "ai": "Artificial intelligence",
            "ml": "Machine learning",
            "nlp": "Natural language processing",
        }
        article_title = aliases.get(normalized_topic.lower(), normalized_topic)
        article_title = article_title[0].upper() + article_title[1:]
        article_slug = quote(article_title.replace(" ", "_"), safe="()_,-")
        return cls(
            label=normalized_topic.lower(),
            url=f"https://{language}.wikipedia.org/wiki/{article_slug}",
        )


@dataclass(frozen=True)
class CrawledArticle:
    document_id: str
    title: str
    text: str
    url: str
    labels: list[str]
    collected_at: str


@dataclass(frozen=True)
class CrawlProgress:
    topic: str
    visited: int
    collected: int
    queued: int
    current_url: str
    error: str | None = None


class WikipediaCrawler:
    """Polite HTML crawler restricted to Wikipedia article pages."""

    def __init__(
        self,
        user_agent: str,
        request_timeout_seconds: float = 20.0,
        crawl_delay_seconds: float = 1.0,
        on_progress: Callable[[CrawlProgress], None] | None = None,
    ) -> None:
        self.headers = {"User-Agent": user_agent, "Accept": "text/html"}
        self.timeout = request_timeout_seconds
        self.delay = crawl_delay_seconds
        self.on_progress = on_progress
        self._robots: dict[str, RobotFileParser] = {}

    def crawl(self, seeds: list[CrawlSeed], max_pages_per_seed: int = 25) -> list[CrawledArticle]:
        if not seeds:
            raise ValueError("At least one Wikipedia topic is required")
        collected: dict[str, CrawledArticle] = {}
        labels_by_url: dict[str, set[str]] = {}
        with httpx.Client(
            headers=self.headers,
            timeout=self.timeout,
            follow_redirects=True,
        ) as client:
            for seed in seeds:
                for article in self._crawl_seed(client, seed, max_pages_per_seed):
                    collected[article.url] = article
                    labels_by_url.setdefault(article.url, set()).add(seed.label)
        return [
            CrawledArticle(
                document_id=article.document_id,
                title=article.title,
                text=article.text,
                url=article.url,
                labels=sorted(labels_by_url[url]),
                collected_at=article.collected_at,
            )
            for url, article in collected.items()
        ]

    def _crawl_seed(
        self,
        client: httpx.Client,
        seed: CrawlSeed,
        limit: int,
    ) -> list[CrawledArticle]:
        seed_url = self._validate_article_url(seed.url)
        host = urlparse(seed_url).netloc
        pending = deque([seed_url])
        queued = {seed_url}
        visited: set[str] = set()
        articles: list[CrawledArticle] = []
        while pending and len(articles) < limit:
            url = pending.popleft()
            queued.discard(url)
            if url in visited:
                continue
            visited.add(url)
            try:
                if not self._allowed(client, url):
                    continue
                response = client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as error:
                if self.on_progress:
                    self.on_progress(
                        CrawlProgress(
                            topic=seed.label,
                            visited=len(visited),
                            collected=len(articles),
                            queued=len(pending),
                            current_url=url,
                            error=str(error),
                        )
                    )
                if pending:
                    time.sleep(self.delay)
                continue
            article, links = self._parse_article(response.text, str(response.url), seed.label)
            if article:
                articles.append(article)
            for link in links:
                if urlparse(link).netloc == host and link not in visited and link not in queued:
                    pending.append(link)
                    queued.add(link)
            if self.on_progress:
                self.on_progress(
                    CrawlProgress(
                        topic=seed.label,
                        visited=len(visited),
                        collected=len(articles),
                        queued=len(pending),
                        current_url=str(response.url),
                    )
                )
            if pending and len(articles) < limit:
                time.sleep(self.delay)
        return articles

    def _allowed(self, client: httpx.Client, url: str) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            robots_url = f"{origin}/robots.txt"
            response = client.get(robots_url)
            response.raise_for_status()
            parser = RobotFileParser()
            parser.set_url(robots_url)
            parser.parse(response.text.splitlines())
            self._robots[origin] = parser
        return self._robots[origin].can_fetch(self.headers["User-Agent"], url)

    def _parse_article(
        self,
        html: str,
        source_url: str,
        label: str,
    ) -> tuple[CrawledArticle | None, list[str]]:
        soup = BeautifulSoup(html, "html.parser")
        content = soup.select_one("#mw-content-text .mw-parser-output")
        heading = soup.select_one("#firstHeading")
        if content is None or heading is None:
            return None, []
        for unwanted in content.select(
            "table, style, script, sup.reference, .mw-editsection, .navbox, .infobox"
        ):
            unwanted.decompose()
        paragraphs = [node.get_text(" ", strip=True) for node in content.select("p")]
        text = "\n\n".join(paragraph for paragraph in paragraphs if len(paragraph) >= 40)
        canonical = soup.select_one('link[rel="canonical"]')
        url = canonical.get("href", source_url) if canonical else source_url
        normalized_url = self._validate_article_url(str(url))
        links = self._article_links(content, normalized_url)
        if len(text.split()) < 80:
            return None, links
        article = CrawledArticle(
            document_id=hashlib.sha256(normalized_url.encode()).hexdigest(),
            title=heading.get_text(" ", strip=True),
            text=text,
            url=normalized_url,
            labels=[label],
            collected_at=datetime.now(timezone.utc).isoformat(),
        )
        return article, links

    def _article_links(self, content: Tag, base_url: str) -> list[str]:
        links: list[str] = []
        for anchor in content.select("a[href]"):
            href = anchor.get("href")
            if not isinstance(href, str) or href.startswith("#"):
                continue
            candidate = urldefrag(urljoin(base_url, href)).url
            try:
                links.append(self._validate_article_url(candidate))
            except ValueError:
                continue
        return list(dict.fromkeys(links))

    @staticmethod
    def _is_special_namespace(path: str) -> bool:
        return ":" in path.removeprefix("/wiki/")

    @staticmethod
    def _validate_article_url(url: str) -> str:
        normalized = urldefrag(url).url
        parsed = urlparse(normalized)
        is_wikipedia = parsed.netloc.endswith(".wikipedia.org")
        if parsed.scheme != "https" or not is_wikipedia or not parsed.path.startswith("/wiki/"):
            raise ValueError(f"Only HTTPS Wikipedia article URLs can be crawled: {url}")
        if WikipediaCrawler._is_special_namespace(parsed.path):
            raise ValueError(f"Wikipedia special namespaces are not crawled: {url}")
        return normalized
