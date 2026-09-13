from atlasmind.crawler import CrawledArticle
from atlasmind.database import PostgresArticleRepository, article_as_row
from atlasmind.storage import ExpansionStorage


class CorpusWriter:
    """Mirror one crawl into Expansion JSON Lines and PostgreSQL."""

    relative_path = "corpus/wikipedia_articles.jsonl"

    def __init__(
        self,
        storage: ExpansionStorage,
        database: PostgresArticleRepository,
    ) -> None:
        self.storage = storage
        self.database = database

    def save(self, articles: list[CrawledArticle]) -> dict[str, object]:
        existing = {
            row["document_id"]: row for row in self.storage.read_jsonl_if_exists(self.relative_path)
        }
        existing.update({article.document_id: article_as_row(article) for article in articles})
        destination = self.storage.write_jsonl(self.relative_path, existing.values())
        database_rows = self.database.upsert(articles)
        return {
            "crawled_articles": len(articles),
            "corpus_articles": len(existing),
            "postgres_rows": database_rows,
            "corpus_path": str(destination),
        }
