# facts_cli.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Answers where a fact lives and who cites it, and rewrites the quotes of a fact from its home.

Usage: python tools/documentation/facts_cli.py [--root ROOT] list | where ID | cited-by ID | apply
The map is computed from the markers on every run and is never stored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation import facts  # noqa: E402
from documentation.check_docs_facts import Manual, load  # noqa: E402

K_EXIT_DONE = 0
K_EXIT_UNKNOWN = 1


def citations(manual: Manual, fact_id: str) -> list[str]:
    """Returns `path:line: quote` or `path:line: follows` for each place that cites a fact."""
    found = []
    for page in manual.pages:
        found += [f"{page.path}:{block.first}: quote" for block in page.quotes if block.id == fact_id]
        found += [f"{page.path}:{line}: follows" for cited, line in page.follows if cited == fact_id]
    return found


def apply(manual: Manual) -> list[str]:
    """Rewrites every out-of-date quote from its home and returns the pages it wrote."""
    homes = {fact_id: home[1].body for fact_id, home in manual.homes().items()}
    written = []
    for page in manual.pages:
        path = manual.root / page.path
        text = path.read_text(encoding="utf-8")
        updated = facts.regenerate(text, homes)
        if updated != text:
            path.write_text(updated, encoding="utf-8", newline="")
            written.append(page.path)
    return written


def main(argv: list[str] | None = None) -> int:
    """Runs one subcommand and returns its exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="every fact with its home and the number of places that cite it")
    commands.add_parser("where", help="the home of a fact").add_argument("id")
    commands.add_parser("cited-by", help="every place that quotes or follows a fact").add_argument("id")
    commands.add_parser("apply", help="rewrite every quote from its home")
    arguments = parser.parse_args(argv)
    manual = load(arguments.root)
    homes = manual.homes()
    if arguments.command == "list":
        for fact_id, (page, block) in sorted(homes.items()):
            print(f"{fact_id}: {page.path}:{block.first}, cited {len(citations(manual, fact_id))}")
    elif arguments.command == "apply":
        for path in apply(manual):
            print(f"{path}: written")
    elif arguments.id not in homes:
        print(f"no fact {arguments.id}", file=sys.stderr)
        return K_EXIT_UNKNOWN
    elif arguments.command == "where":
        page, block = homes[arguments.id]
        print(f"{page.path}:{block.first}")
    else:
        print("\n".join(citations(manual, arguments.id)))
    return K_EXIT_DONE


if __name__ == "__main__":
    sys.exit(main())
