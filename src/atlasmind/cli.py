import argparse
import importlib.util
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atlasmind.crawler import CrawlProgress


COMMAND_DEPENDENCIES = {
    "crawl": {
        "bs4": "beautifulsoup4",
        "httpx": "httpx",
        "psycopg": "psycopg",
        "pydantic": "pydantic",
        "pydantic_settings": "pydantic-settings",
    },
    "init-db": {
        "numpy": "numpy",
        "pgvector": "pgvector",
        "psycopg": "psycopg",
        "sklearn": "scikit-learn",
    },
    "train": {
        "joblib": "joblib",
        "numpy": "numpy",
        "pgvector": "pgvector",
        "psycopg": "psycopg",
        "sklearn": "scikit-learn",
    },
    "search": {
        "numpy": "numpy",
        "pgvector": "pgvector",
        "psycopg": "psycopg",
        "sklearn": "scikit-learn",
    },
    "classify": {
        "joblib": "joblib",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
    },
    "generate": {
        "numpy": "numpy",
        "num2words": "num2words",
        "PIL": "pillow",
        "pgvector": "pgvector",
        "psycopg": "psycopg",
        "sklearn": "scikit-learn",
        "torch": "torch",
        "torchvision": "torchvision",
        "transformers": "transformers",
    },
}

COMMON_DEPENDENCIES = {
    "pydantic": "pydantic",
    "pydantic_settings": "pydantic-settings",
}


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(description="AtlasMind crawler and trainer")
    subcommands = command_parser.add_subparsers(dest="command", required=True)

    crawl = subcommands.add_parser("crawl", help="Crawl Wikipedia automatically by topic")
    crawl.add_argument(
        "--topics",
        nargs="+",
        required=True,
        help='Topic names, for example --topics ai biology "world history"',
    )
    crawl.add_argument("--language", default=None, help="Wikipedia language code")
    crawl.add_argument("--max-pages-per-seed", type=positive_int, default=25)

    subcommands.add_parser("init-db", help="Create PostgreSQL tables and indexes")
    subcommands.add_parser("train", help="Train indexes from the local knowledge corpus")
    search = subcommands.add_parser("search", help="Search locally stored knowledge")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument(
        "--mode",
        choices=["keyword", "vector", "hybrid"],
        default="hybrid",
    )

    classify = subcommands.add_parser("classify", help="Classify text by learned topic")
    classify.add_argument("text")

    generate = subcommands.add_parser(
        "generate",
        help="Generate a grounded answer from local knowledge and fresh web pages",
    )
    generate.add_argument("question")
    generate.add_argument("--limit", type=positive_int, default=5)
    generate.add_argument("--max-new-tokens", type=positive_int, default=256)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument(
        "--image",
        type=Path,
        help="Optional local image path to include with the question",
    )
    generate.add_argument(
        "--freshness",
        choices=["pd", "pw", "pm", "py"],
        help="Web freshness: past day, week, month, or year",
    )
    generate.add_argument(
        "--no-web",
        action="store_true",
        help="Generate using only knowledge already stored in PostgreSQL",
    )
    return command_parser


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return number


def print_crawl_progress(progress: "CrawlProgress") -> None:
    error_text = f" error={progress.error!r}" if progress.error else ""
    print(
        f"topic={progress.topic!r} "
        f"collected={progress.collected} "
        f"visited={progress.visited} "
        f"queued={progress.queued} "
        f"url={progress.current_url}"
        f"{error_text}",
        flush=True,
    )


def missing_dependencies(command: str) -> list[str]:
    dependencies = {**COMMON_DEPENDENCIES, **COMMAND_DEPENDENCIES.get(command, {})}
    return [
        package
        for module, package in dependencies.items()
        if importlib.util.find_spec(module) is None
    ]


def main() -> None:
    command_parser = parser()
    arguments = command_parser.parse_args()
    missing = missing_dependencies(arguments.command)
    if missing:
        command_parser.error(
            "Missing dependencies: "
            f"{', '.join(missing)}. Install them with "
            f"{'.venv/bin/python -m pip install -r requirements.txt'!r}."
        )

    from atlasmind.config import get_settings
    from atlasmind.storage import ExpansionStorage

    settings = get_settings()
    storage = ExpansionStorage(settings.storage_root)

    if arguments.command == "crawl":
        from atlasmind.corpus import CorpusWriter
        from atlasmind.crawler import CrawlSeed, WikipediaCrawler
        from atlasmind.database import PostgresArticleRepository

        database = PostgresArticleRepository(settings.postgres_dsn)
        try:
            language = arguments.language or settings.wikipedia_language
            seeds = [CrawlSeed.from_topic(topic, language) for topic in arguments.topics]
        except ValueError as error:
            command_parser.error(str(error))
        crawler = WikipediaCrawler(
            settings.wikipedia_user_agent,
            settings.request_timeout_seconds,
            settings.crawl_delay_seconds,
            on_progress=print_crawl_progress,
        )
        articles = crawler.crawl(seeds, arguments.max_pages_per_seed)
        print(CorpusWriter(storage, database).save(articles))
    elif arguments.command == "init-db":
        from atlasmind.database import PostgresArticleRepository
        from atlasmind.vector_store import LocalTextEmbedder, PgVectorKnowledgeStore

        database = PostgresArticleRepository(settings.postgres_dsn)
        vector_store = PgVectorKnowledgeStore(settings.postgres_dsn, LocalTextEmbedder())
        database.initialize()
        vector_store.initialize()
        print({"status": "initialized", "database_role": "debarunlahiri"})
    elif arguments.command == "train":
        from psycopg import Error as PostgresError

        from atlasmind.training import train_from_wikipedia
        from atlasmind.vector_store import LocalTextEmbedder, PgVectorKnowledgeStore

        vector_store = PgVectorKnowledgeStore(settings.postgres_dsn, LocalTextEmbedder())
        try:
            summary = train_from_wikipedia(storage, vector_store, settings.random_seed)
        except (FileNotFoundError, PostgresError, RuntimeError, ValueError) as error:
            command_parser.error(str(error))
        print(asdict(summary))
    elif arguments.command == "search":
        from atlasmind.database import PostgresArticleRepository
        from atlasmind.vector_store import (
            LocalTextEmbedder,
            PgVectorKnowledgeStore,
            reciprocal_rank_fusion,
        )

        database = PostgresArticleRepository(settings.postgres_dsn)
        vector_store = PgVectorKnowledgeStore(settings.postgres_dsn, LocalTextEmbedder())
        if arguments.mode == "keyword":
            results = database.search(arguments.query, arguments.limit)
        elif arguments.mode == "vector":
            results = vector_store.search(arguments.query, arguments.limit)
        else:
            candidate_limit = arguments.limit * 2
            keyword_results = database.search(arguments.query, candidate_limit)
            vector_results = vector_store.search(arguments.query, candidate_limit)
            results = reciprocal_rank_fusion(
                keyword_results,
                vector_results,
                arguments.limit,
            )
        print(results)
    elif arguments.command == "classify":
        import joblib

        classifier_path = storage.path("models", "topic_classifier.joblib")
        if not classifier_path.is_file():
            command_parser.error(
                "The topic classifier has not been trained. Run "
                f"{'PYTHONPATH=src .venv/bin/python -m atlasmind.cli train'!r} first. "
                f"Expected artifact: {classifier_path}"
            )
        classifier = joblib.load(classifier_path)
        probabilities = classifier.predict_proba([arguments.text])[0]
        labels = classifier.classes_
        ranked = sorted(zip(labels, probabilities), key=lambda item: -item[1])
        print([{"label": label, "probability": float(score)} for label, score in ranked])
    elif arguments.command == "generate":
        from PIL import Image, UnidentifiedImageError

        from atlasmind.database import PostgresArticleRepository
        from atlasmind.generation import LocalMultimodalGenerator
        from atlasmind.rag import KnowledgeRagService
        from atlasmind.vector_store import LocalTextEmbedder, PgVectorKnowledgeStore
        from atlasmind.web_search import HtmlWebSearch

        if not 0 <= arguments.temperature <= 2:
            command_parser.error("--temperature must be between 0 and 2")
        database = PostgresArticleRepository(settings.postgres_dsn)
        vector_store = PgVectorKnowledgeStore(settings.postgres_dsn, LocalTextEmbedder())
        image = None
        if arguments.image is not None:
            if not arguments.image.is_file():
                command_parser.error(f"Image file not found: {arguments.image}")
            try:
                image = Image.open(arguments.image)
                image.load()
            except (OSError, UnidentifiedImageError) as error:
                command_parser.error(f"Could not read image: {error}")
        image_id = (
            database.save_image(image, original_name=arguments.image.name)
            if image is not None and arguments.image is not None
            else None
        )
        web_search = (
            HtmlWebSearch(
                settings.web_search_user_agent,
                settings.web_search_timeout_seconds,
                database,
                settings.web_search_store_results,
                settings.web_search_store_images,
                settings.web_search_max_workers,
            )
            if not arguments.no_web
            else None
        )
        generator = LocalMultimodalGenerator(
            settings.multimodal_model_id,
            settings.huggingface_cache_dir,
            settings.compute_device,
            settings.model_local_files_only,
        )
        rag = KnowledgeRagService(
            database,
            vector_store,
            generator,
            settings.generator_max_context_characters,
            web_search,
            arguments.freshness or settings.web_search_freshness,
        )
        try:
            answer = rag.answer(
                arguments.question,
                arguments.limit,
                arguments.max_new_tokens,
                arguments.temperature,
                image,
                not arguments.no_web,
                image_id,
            )
        except Exception as error:
            command_parser.error(str(error))
        print(answer.as_dict())


if __name__ == "__main__":
    main()
