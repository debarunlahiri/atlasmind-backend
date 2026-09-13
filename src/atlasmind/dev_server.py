import argparse

import uvicorn


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(
        description="Run the AtlasMind development API with automatic reload",
    )
    command_parser.add_argument("--host", default="127.0.0.1")
    command_parser.add_argument("--port", type=int, default=8000)
    return command_parser


def main() -> None:
    arguments = parser().parse_args()
    uvicorn.run(
        "atlasmind.api.app:app",
        host=arguments.host,
        port=arguments.port,
        reload=True,
        reload_dirs=["src"],
    )


if __name__ == "__main__":
    main()
