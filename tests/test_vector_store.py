from atlasmind.vector_store import LocalTextEmbedder, reciprocal_rank_fusion


def test_local_embedder_has_expected_dimensions() -> None:
    vectors = LocalTextEmbedder().encode(["artificial intelligence", "biology"])
    assert vectors.shape == (2, 384)


def test_reciprocal_rank_fusion_rewards_results_in_both_lists() -> None:
    keyword = [
        {"source_url": "https://en.wikipedia.org/wiki/A", "title": "A"},
        {"source_url": "https://en.wikipedia.org/wiki/B", "title": "B"},
    ]
    vector = [
        {"source_url": "https://en.wikipedia.org/wiki/B", "title": "B"},
        {"source_url": "https://en.wikipedia.org/wiki/C", "title": "C"},
    ]

    results = reciprocal_rank_fusion(keyword, vector, limit=3)

    assert results[0]["source_url"] == "https://en.wikipedia.org/wiki/B"
