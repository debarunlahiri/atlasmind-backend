import httpx

from atlasmind.web_search import HtmlWebSearch


def test_parse_html_search_results_unwraps_redirects() -> None:
    html = """
    <div class="result">
      <a class="result__a"
         href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fnews">
        Example result
      </a>
      <a class="result__snippet">2026-09-13 Fresh information about the topic.</a>
    </div>
    """

    assert HtmlWebSearch._parse_results(html, 5) == [
        {
            "document_id": "",
            "source_type": "web_html",
            "title": "Example result",
            "source_url": "https://example.com/news",
            "snippet": "2026-09-13 Fresh information about the topic.",
            "published_at": "2026-09-13",
            "score": 1.0,
        }
    ]


def test_parse_html_search_results_rejects_local_urls() -> None:
    html = """
    <div class="result">
      <a class="result__a" href="http://localhost/private">Private</a>
      <a class="result__snippet">This must not be fetched.</a>
    </div>
    """

    assert HtmlWebSearch._parse_results(html, 5) == []


def test_parse_lite_html_search_results() -> None:
    html = """
    <a class="result-link" href="https://example.com/article">Example article</a>
    <td class="result-snippet">2026-09-13 A current article summary.</td>
    """

    results = HtmlWebSearch._parse_results(html, 5)

    assert results[0]["title"] == "Example article"
    assert results[0]["source_url"] == "https://example.com/article"


def test_compact_query_preserves_names_and_subject() -> None:
    query = "What's the latest Dario news about AI slowing down with Elon Musk and Sam Altman?"

    compact = HtmlWebSearch._compact_query(query)

    assert compact == "latest Dario news about AI slowing down Elon Musk Sam Altman"


def test_parse_bing_html_results_and_unwrap_redirect() -> None:
    html = """
    <li class="b_algo">
      <h2>
        <a href="https://www.bing.com/ck/a?u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS9uZXdz">
          Dario Amodei discusses AI
        </a>
      </h2>
      <div class="b_caption"><p>Dario Amodei commented on AI development.</p></div>
    </li>
    """

    results = HtmlWebSearch._parse_bing_results(html, 5)

    assert results[0]["source_url"] == "https://example.com/news"
    assert results[0]["title"] == "Dario Amodei discusses AI"


def test_relevance_filter_rejects_unrelated_search_result() -> None:
    result = {
        "title": "Apache Flink task slots",
        "snippet": "Configure distributed processing resources.",
    }

    assert not HtmlWebSearch._is_relevant_result(
        "Dario Amodei Elon Musk Sam Altman AI",
        result,
    )


def test_search_challenge_is_detected() -> None:
    response = httpx.Response(202, text='<form id="challenge-form"></form>')

    assert HtmlWebSearch._is_search_challenge(response)


def test_parse_bing_news_results() -> None:
    html = """
    <div class="news-card" data-url="https://example.com/ai-news"
         data-title="Elon Musk and Sam Altman agree with Dario Amodei">
      <a class="title" href="https://example.com/ai-news">AI leaders respond</a>
      <div class="snippet">Amodei called for slowing powerful AI development.</div>
    </div>
    """

    results = HtmlWebSearch._parse_bing_news_results(html, 5)

    assert results[0]["source_url"] == "https://example.com/ai-news"
    assert "Dario Amodei" in str(results[0]["title"])


def test_news_query_quotes_full_names() -> None:
    query = "Did Dario Amodei, Elon Musk, and Sam Altman discuss the latest AI news?"

    news_query = HtmlWebSearch._news_query(query, HtmlWebSearch._compact_query(query))

    assert '"Dario Amodei"' in news_query
    assert '"Elon Musk"' in news_query
    assert '"Sam Altman"' in news_query


def test_news_query_splits_adjacent_full_names_after_question_opener() -> None:
    query = "Did Dario Amodei Elon Musk Sam Altman discuss AI safety?"

    news_query = HtmlWebSearch._news_query(query, HtmlWebSearch._compact_query(query))

    assert news_query.startswith('"Dario Amodei" "Elon Musk" "Sam Altman"')


def test_entity_role_question_extracts_direct_wikipedia_candidate() -> None:
    target = HtmlWebSearch._entity_role_target("Who is the CEO of Trump Tower?")

    assert target == "Trump Tower"


def test_parse_brave_html_results() -> None:
    html = """
    <div class="snippet" data-type="web">
      <div class="result-content">
        <a href="https://docs.netapp.com/ontap/storage-disk-show.html">
          <div class="title">storage disk show</div>
        </a>
        <div class="generic-snippet">
          <div class="content">Display broken physical disks in ONTAP 9.</div>
        </div>
      </div>
    </div>
    """

    results = HtmlWebSearch._parse_brave_results(html, 5)

    assert results[0]["title"] == "storage disk show"
    assert results[0]["source_url"].startswith("https://docs.netapp.com/")


def test_extract_urls_deduplicates_markdown_link_target() -> None:
    text = "[https://example.com/news/1](https://example.com/news/1) give me context"

    assert HtmlWebSearch._extract_urls(text) == ["https://example.com/news/1"]


def test_related_search_query_uses_direct_page_metadata() -> None:
    direct_results = [
        {
            "title": "Polymarket on X",
            "snippet": "David Sacks comments on slowing AI development.",
        }
    ]

    query = HtmlWebSearch._related_search_query(
        "https://x.com/example/status/1 give me context of this news",
        direct_results,
    )

    assert "x.com" not in query
    assert "David Sacks" in query
