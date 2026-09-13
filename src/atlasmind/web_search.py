import re
from base64 import urlsafe_b64decode
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from html import unescape
from ipaddress import ip_address
from typing import Optional, Protocol
from urllib.parse import parse_qs, quote, unquote, urlparse

import httpx
from bs4 import BeautifulSoup

DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
BING_HTML_URL = "https://www.bing.com/search"
BING_NEWS_HTML_URL = "https://www.bing.com/news/search"
BRAVE_HTML_URL = "https://search.brave.com/search"
VALID_FRESHNESS = {"pd", "pw", "pm", "py"}
FRESHNESS_PARAMETERS = {"pd": "d", "pw": "w", "pm": "m", "py": "y"}
MAX_PAGE_CHARACTERS = 12_000
SEARCH_STOP_WORDS = {
    "about",
    "agreeing",
    "and",
    "are",
    "did",
    "for",
    "from",
    "have",
    "is",
    "need",
    "of",
    "s",
    "that",
    "the",
    "them",
    "this",
    "to",
    "what",
    "which",
    "with",
}


class WebResultStore(Protocol):
    def upsert_web_results(self, results: list[dict[str, object]], query: str) -> int: ...

    def save_web_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        source_url: str,
        document_url: str,
        title: str,
    ) -> Optional[str]: ...


class HtmlWebSearch:
    """Search and read public web pages through HTML requests, without an API."""

    def __init__(
        self,
        user_agent: str,
        timeout_seconds: float = 20.0,
        result_store: Optional[WebResultStore] = None,
        store_results: bool = True,
        store_images: bool = False,
        max_workers: int = 5,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.result_store = result_store
        self.store_results = store_results
        self.store_images = store_images
        self.max_workers = max_workers
        self.headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,image/avif,image/webp,image/*;q=0.8",
        }

    def search(
        self,
        query: str,
        limit: int = 5,
        freshness: str = "pm",
        persist: bool = True,
    ) -> list[dict[str, object]]:
        results, _ = self.search_with_activity(query, limit, freshness, persist)
        return results

    def search_with_activity(
        self,
        query: str,
        limit: int = 5,
        freshness: str = "pm",
        persist: bool = True,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        query = query.strip()
        if not query:
            return [], []
        if freshness not in VALID_FRESHNESS:
            raise ValueError("Freshness must be one of: pd, pw, pm, py")

        result_limit = min(max(limit, 1), 10)
        explicit_urls = self._extract_urls(query)
        direct_results = [
            result
            for url in explicit_urls
            if (result := self._read_explicit_url(url)) is not None
        ]
        search_query = self._related_search_query(query, direct_results)
        results: list[dict[str, object]] = []
        activity: list[dict[str, object]] = (
            [
                self._activity(
                    "Direct URL HTML",
                    "success" if direct_results else "no_results",
                    len(direct_results),
                    None if direct_results else "URL returned no readable HTML metadata",
                )
            ]
            if explicit_urls
            else []
        )
        attempts = (
            (search_query, FRESHNESS_PARAMETERS[freshness]),
            (self._compact_query(search_query), None),
        )
        for attempted_query, freshness_parameter in attempts:
            form_data = {"q": attempted_query}
            if freshness_parameter is not None:
                form_data["df"] = freshness_parameter
            try:
                response = httpx.post(
                    DUCKDUCKGO_HTML_URL,
                    headers=self.headers,
                    data=form_data,
                    follow_redirects=True,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
            except httpx.HTTPError as error:
                activity.append(
                    self._activity("DuckDuckGo HTML", "error", 0, str(error))
                )
                continue
            if self._is_search_challenge(response):
                activity.append(
                    self._activity("DuckDuckGo HTML", "challenged", 0, "CAPTCHA detected")
                )
                break
            results = self._parse_results(response.text, result_limit)
            results = [
                result for result in results if self._is_relevant_result(search_query, result)
            ]
            activity.append(
                self._activity(
                    "DuckDuckGo HTML",
                    "success" if results else "no_results",
                    len(results),
                )
            )
            if results:
                self._tag_results(results, "DuckDuckGo HTML")
                break
        if not results:
            compact_query = self._compact_query(search_query)
            if self._looks_current(search_query):
                results = self._search_bing_news_html(
                    self._news_query(search_query, compact_query),
                    result_limit,
                )
                activity.append(
                    self._activity(
                        "Bing News HTML",
                        "success" if results else "no_results",
                        len(results),
                    )
                )
                self._tag_results(results, "Bing News HTML")
            if not results:
                entity = self._entity_role_target(search_query)
                results = self._search_wikipedia_entity_html(entity) if entity else []
                if entity:
                    activity.append(
                        self._activity(
                            "Wikipedia direct HTML",
                            "success" if results else "no_results",
                            len(results),
                        )
                    )
                if results:
                    self._tag_results(results, "Wikipedia direct HTML")
            if not results:
                results = self._search_bing_html(compact_query, result_limit)
                activity.append(
                    self._activity(
                        "Bing Web HTML",
                        "success" if results else "no_results",
                        len(results),
                    )
                )
                self._tag_results(results, "Bing Web HTML")
            if not results:
                results = self._search_brave_html(search_query, result_limit)
                activity.append(
                    self._activity(
                        "Brave Web HTML",
                        "success" if results else "no_results",
                        len(results),
                    )
                )
                self._tag_results(results, "Brave Web HTML")
        if results:
            with ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(results)),
                thread_name_prefix="atlasmind-web",
            ) as executor:
                enriched = list(
                    executor.map(
                        lambda result: self._read_result(result, persist),
                        results,
                    )
                )
        else:
            enriched = []
        combined_results = [*direct_results, *enriched]
        if persist and self.store_results and self.result_store is not None:
            self.result_store.upsert_web_results(combined_results, query)
        return combined_results, activity

    @staticmethod
    def _activity(
        engine: str,
        status: str,
        result_count: int,
        detail: Optional[str] = None,
    ) -> dict[str, object]:
        return {
            "engine": engine,
            "status": status,
            "result_count": result_count,
            "detail": detail,
        }

    @staticmethod
    def _tag_results(results: list[dict[str, object]], engine: str) -> None:
        for result in results:
            result["search_engine"] = engine

    @staticmethod
    def _extract_urls(text: str) -> list[str]:
        urls = [url.rstrip(".,;:!?'") for url in re.findall(r"https?://[^\s\])>]+", text)]
        return list(dict.fromkeys(urls))

    @staticmethod
    def _related_search_query(
        question: str,
        direct_results: list[dict[str, object]],
    ) -> str:
        without_markdown_links = re.sub(r"\[https?://[^\]]+\]\(https?://[^)]+\)", " ", question)
        without_urls = re.sub(r"https?://\S+", " ", without_markdown_links)
        user_text = " ".join(without_urls.split())
        discovered_text = " ".join(
            f"{result.get('title', '')} {result.get('snippet', '')}"
            for result in direct_results
        )
        return " ".join(f"{user_text} {discovered_text}".split())[:2000] or question

    def _read_explicit_url(self, url: str) -> Optional[dict[str, object]]:
        if not self._is_public_web_url(url):
            return None
        try:
            response = httpx.get(
                url,
                headers=self.headers,
                follow_redirects=True,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            return None
        if not self._is_public_web_url(str(response.url)):
            return None
        if "html" not in response.headers.get("content-type", "").lower():
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        title_node = soup.select_one('meta[property="og:title"], meta[name="twitter:title"]')
        description_node = soup.select_one(
            'meta[property="og:description"], meta[name="twitter:description"], '
            'meta[name="description"]'
        )
        title = unescape(str(title_node.get("content", "")).strip()) if title_node else ""
        if not title and soup.title is not None:
            title = soup.title.get_text(" ", strip=True)
        description = (
            unescape(str(description_node.get("content", "")).strip())
            if description_node
            else ""
        )
        paragraphs = [
            " ".join(node.get_text(" ", strip=True).split())
            for node in soup.select("article p, main p")
        ]
        readable_text = "\n".join(text for text in paragraphs if len(text) >= 40)
        snippet = "\n".join(value for value in (description, readable_text) if value)
        if not title or not snippet:
            return None
        return {
            "document_id": "",
            "source_type": "direct_url",
            "title": " ".join(title.split()),
            "source_url": str(response.url),
            "snippet": snippet[:MAX_PAGE_CHARACTERS],
            "published_at": self._published_at(soup),
            "score": 1.0,
            "search_engine": "Direct URL HTML",
        }

    @staticmethod
    def _compact_query(query: str) -> str:
        words = re.findall(r"[A-Za-z0-9]+", query)
        useful_words = [word for word in words if word.lower() not in SEARCH_STOP_WORDS]
        return " ".join(useful_words[:12]) or query

    @staticmethod
    def _is_search_challenge(response: httpx.Response) -> bool:
        lowered = response.text.lower()
        return response.status_code == 202 or any(
            marker in lowered
            for marker in ("anomaly-modal", "challenge-form", "unusual traffic", "captcha")
        )

    @staticmethod
    def _looks_current(query: str) -> bool:
        words = set(re.findall(r"[a-z]+", query.lower()))
        return bool(words & {"current", "latest", "news", "recent", "recently", "today"})

    @staticmethod
    def _news_query(query: str, compact_query: str) -> str:
        names: list[str] = []
        question_openers = {
            "Can",
            "Could",
            "Did",
            "Do",
            "Does",
            "How",
            "Is",
            "Was",
            "What",
            "When",
            "Who",
            "Why",
        }
        for match in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", query):
            words = match.split()
            if words[0] in question_openers:
                words = words[1:]
            names.extend(
                " ".join(words[index : index + 2])
                for index in range(0, len(words) - 1, 2)
            )
        quoted_names = " ".join(f'"{name}"' for name in names[:4])
        return f"{quoted_names} {compact_query}".strip()

    def _search_bing_news_html(self, query: str, limit: int) -> list[dict[str, object]]:
        try:
            response = httpx.get(
                BING_NEWS_HTML_URL,
                headers={
                    **self.headers,
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                },
                params={"q": query, "qft": 'sortbydate="1"'},
                follow_redirects=True,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            return []
        return [
            result
            for result in self._parse_bing_news_results(response.text, limit)
            if self._is_relevant_result(query, result)
        ]

    @staticmethod
    def _entity_role_target(query: str) -> Optional[str]:
        match = re.search(
            r"\b(?:ceo|owner|founder|president)\s+of\s+(.+?)[?.!]*$",
            query,
            flags=re.IGNORECASE,
        )
        if match is None:
            return None
        entity = " ".join(match.group(1).split()).strip()
        if not entity or len(entity) > 120:
            return None
        return entity

    def _search_wikipedia_entity_html(self, entity: str) -> list[dict[str, object]]:
        if not entity:
            return []
        article_slug = quote(entity.replace(" ", "_"), safe="_()'-")
        article_url = f"https://en.wikipedia.org/wiki/{article_slug}"
        try:
            response = httpx.get(
                article_url,
                headers=self.headers,
                follow_redirects=True,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            return []
        if "html" not in response.headers.get("content-type", "").lower():
            return []
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [
            " ".join(node.get_text(" ", strip=True).split())
            for node in soup.select(".mw-parser-output > p, main p")
        ]
        article_text = "\n".join(text for text in paragraphs if len(text) >= 40)
        if not article_text:
            return []
        heading = soup.select_one("h1")
        title = heading.get_text(" ", strip=True) if heading is not None else entity
        return [
            {
                "document_id": "",
                "source_type": "wikipedia",
                "title": title,
                "source_url": str(response.url),
                "snippet": article_text[:MAX_PAGE_CHARACTERS],
                "published_at": None,
                "score": 1.0,
            }
        ]

    @classmethod
    def _parse_bing_news_results(cls, html: str, limit: int) -> list[dict[str, object]]:
        soup = BeautifulSoup(html, "html.parser")
        results: list[dict[str, object]] = []
        for node in soup.select(".news-card"):
            link = node.select_one("a.title")
            snippet = node.select_one(".snippet")
            source_url = str(
                node.get("data-url")
                or node.get("url")
                or (link.get("href") if link is not None else "")
            )
            title = str(
                node.get("data-title")
                or node.get("title")
                or (link.get_text(" ", strip=True) if link is not None else "")
            )
            snippet_text = " ".join(
                snippet.get_text(" ", strip=True).split() if snippet is not None else []
            )
            source_url = cls._unwrap_url(source_url)
            if not title or not snippet_text or not cls._is_public_web_url(source_url):
                continue
            results.append(
                {
                    "document_id": "",
                    "source_type": "web_html",
                    "title": " ".join(title.split()),
                    "source_url": source_url,
                    "snippet": snippet_text,
                    "published_at": None,
                    "score": 1.0,
                }
            )
            if len(results) >= limit:
                break
        return results

    def _search_bing_html(self, query: str, limit: int) -> list[dict[str, object]]:
        try:
            response = httpx.get(
                BING_HTML_URL,
                headers={
                    **self.headers,
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                },
                params={"q": query, "count": limit},
                follow_redirects=True,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            return []
        return [
            result
            for result in self._parse_bing_results(response.text, limit)
            if self._is_relevant_result(query, result)
        ]

    def _search_brave_html(self, query: str, limit: int) -> list[dict[str, object]]:
        try:
            response = httpx.get(
                BRAVE_HTML_URL,
                headers={
                    **self.headers,
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                },
                params={"q": query, "source": "web"},
                follow_redirects=True,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            return []
        return [
            result
            for result in self._parse_brave_results(response.text, limit)
            if self._is_relevant_result(query, result)
        ]

    @classmethod
    def _parse_brave_results(cls, html: str, limit: int) -> list[dict[str, object]]:
        soup = BeautifulSoup(html, "html.parser")
        results: list[dict[str, object]] = []
        for node in soup.select('.snippet[data-type="web"] .result-content'):
            link = node.select_one("a[href]")
            title = node.select_one(".title")
            snippet = node.select_one(".generic-snippet .content")
            if link is None or title is None or snippet is None:
                continue
            source_url = str(link.get("href", ""))
            if not cls._is_public_web_url(source_url):
                continue
            snippet_text = " ".join(snippet.get_text(" ", strip=True).split())
            results.append(
                {
                    "document_id": "",
                    "source_type": "web_html",
                    "title": " ".join(title.get_text(" ", strip=True).split()),
                    "source_url": source_url,
                    "snippet": snippet_text,
                    "published_at": cls._date_from_text(snippet_text),
                    "score": 1.0,
                }
            )
            if len(results) >= limit:
                break
        return results

    @classmethod
    def _parse_bing_results(cls, html: str, limit: int) -> list[dict[str, object]]:
        soup = BeautifulSoup(html, "html.parser")
        results: list[dict[str, object]] = []
        for node in soup.select("li.b_algo"):
            link = node.select_one("h2 a")
            snippet = node.select_one(".b_caption p")
            if link is None or snippet is None:
                continue
            source_url = cls._unwrap_url(str(link.get("href", "")))
            if not cls._is_public_web_url(source_url):
                continue
            snippet_text = " ".join(snippet.get_text(" ", strip=True).split())
            results.append(
                {
                    "document_id": "",
                    "source_type": "web_html",
                    "title": " ".join(link.get_text(" ", strip=True).split()),
                    "source_url": source_url,
                    "snippet": snippet_text,
                    "published_at": cls._date_from_text(snippet_text),
                    "score": 1.0,
                }
            )
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _is_relevant_result(query: str, result: dict[str, object]) -> bool:
        query_words = {
            word.lower()
            for word in re.findall(r"[A-Za-z0-9]+", query)
            if len(word) >= 3 and word.lower() not in SEARCH_STOP_WORDS
        }
        if not query_words:
            return True
        result_words = set(
            re.findall(
                r"[a-z0-9]+",
                f"{result.get('title', '')} {result.get('snippet', '')}".lower(),
            )
        )
        required_matches = min(2, len(query_words))
        return len(query_words & result_words) >= required_matches

    def _read_result(
        self,
        result: dict[str, object],
        persist: bool,
    ) -> dict[str, object]:
        source_url = str(result["source_url"])
        try:
            response = httpx.get(
                source_url,
                headers=self.headers,
                follow_redirects=True,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            if not self._is_public_web_url(str(response.url)):
                return result
            if "html" not in response.headers.get("content-type", "").lower():
                return result
            result["source_url"] = str(response.url)
            soup = BeautifulSoup(response.text, "html.parser")
            for element in soup(["script", "style", "noscript", "template", "svg"]):
                element.decompose()
            paragraphs = [
                " ".join(node.get_text(" ", strip=True).split())
                for node in soup.select("article p, main p, p")
            ]
            page_text = "\n".join(text for text in paragraphs if len(text) >= 40)
            if page_text:
                result["snippet"] = page_text[:MAX_PAGE_CHARACTERS]
            result["published_at"] = self._published_at(soup) or result.get("published_at")
            image_url = self._image_url(soup, str(response.url))
            if image_url:
                result["image_url"] = image_url
                image_id = self._store_image(
                    image_url,
                    str(response.url),
                    str(result["title"]),
                    persist,
                )
                if image_id:
                    result["image_id"] = image_id
        except (httpx.HTTPError, ValueError):
            return result
        return result

    def _store_image(
        self,
        image_url: str,
        document_url: str,
        title: str,
        persist: bool,
    ) -> Optional[str]:
        if (
            not persist
            or not self.store_images
            or self.result_store is None
            or not self._is_public_web_url(image_url)
        ):
            return None
        try:
            response = httpx.get(
                image_url,
                headers=self.headers,
                follow_redirects=True,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            if not self._is_public_web_url(str(response.url)):
                return None
            mime_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            if not mime_type.startswith("image/") or len(response.content) > 10 * 1024 * 1024:
                return None
            return self.result_store.save_web_image(
                response.content,
                mime_type,
                image_url,
                document_url,
                title,
            )
        except (httpx.HTTPError, OSError, ValueError):
            return None

    @classmethod
    def _parse_results(cls, html: str, limit: int) -> list[dict[str, object]]:
        soup = BeautifulSoup(html, "html.parser")
        results: list[dict[str, object]] = []
        for node in soup.select(".result"):
            link = node.select_one("a.result__a")
            snippet = node.select_one(".result__snippet")
            if link is None or snippet is None:
                continue
            source_url = cls._unwrap_url(str(link.get("href", "")))
            if not cls._is_public_web_url(source_url):
                continue
            results.append(
                {
                    "document_id": "",
                    "source_type": "web_html",
                    "title": " ".join(link.get_text(" ", strip=True).split()),
                    "source_url": source_url,
                    "snippet": " ".join(snippet.get_text(" ", strip=True).split()),
                    "published_at": cls._date_from_text(snippet.get_text(" ", strip=True)),
                    "score": 1.0,
                }
            )
            if len(results) >= limit:
                break
        if results:
            return results

        for link in soup.select("a.result-link"):
            snippet = link.find_next(class_="result-snippet")
            if snippet is None:
                continue
            source_url = cls._unwrap_url(str(link.get("href", "")))
            if not cls._is_public_web_url(source_url):
                continue
            snippet_text = " ".join(snippet.get_text(" ", strip=True).split())
            results.append(
                {
                    "document_id": "",
                    "source_type": "web_html",
                    "title": " ".join(link.get_text(" ", strip=True).split()),
                    "source_url": source_url,
                    "snippet": snippet_text,
                    "published_at": cls._date_from_text(snippet_text),
                    "score": 1.0,
                }
            )
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _unwrap_url(url: str) -> str:
        parsed = urlparse(url)
        if parsed.netloc.endswith("duckduckgo.com"):
            redirect = parse_qs(parsed.query).get("uddg", [])
            if redirect:
                return unquote(redirect[0])
        if parsed.netloc.endswith("bing.com"):
            encoded_url = parse_qs(parsed.query).get("u", [])
            if encoded_url and encoded_url[0].startswith("a1"):
                encoded = encoded_url[0][2:]
                try:
                    padding = "=" * (-len(encoded) % 4)
                    return urlsafe_b64decode(encoded + padding).decode("utf-8")
                except (ValueError, UnicodeDecodeError):
                    return url
        return url

    @staticmethod
    def _is_public_web_url(url: str) -> bool:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if parsed.scheme not in {"http", "https"} or not hostname or hostname == "localhost":
            return False
        try:
            address = ip_address(hostname)
        except ValueError:
            return True
        return not (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
        )

    @staticmethod
    def _date_from_text(text: str) -> Optional[str]:
        match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
        return match.group(1) if match else None

    @classmethod
    def _published_at(cls, soup: BeautifulSoup) -> Optional[str]:
        selectors = (
            'meta[property="article:published_time"]',
            'meta[name="date"]',
            'meta[name="pubdate"]',
            "time[datetime]",
        )
        for selector in selectors:
            node = soup.select_one(selector)
            if node is None:
                continue
            value = str(node.get("content") or node.get("datetime") or "").strip()
            if not value:
                continue
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
            except ValueError:
                parsed = cls._date_from_text(value)
                if parsed:
                    return parsed
        return None

    @staticmethod
    def _image_url(soup: BeautifulSoup, page_url: str) -> Optional[str]:
        node = soup.select_one('meta[property="og:image"], meta[name="twitter:image"]')
        value = str(node.get("content", "")).strip() if node is not None else ""
        if not value:
            return None
        return str(httpx.URL(page_url).join(value))
