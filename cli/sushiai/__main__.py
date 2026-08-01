"""Enable `python3 -m sushiai` as an entry point."""

from .cli import app

if __name__ == "__main__":
    app()
