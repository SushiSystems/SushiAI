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
        found += [
            f"{page.path}:{block.first}: quote" for block in page.quotes if block.id == fact_id
        ]
        found += [
            f"{page.path}:{line}: follows" for cited, line in page.follows if cited == fact_id
        ]
    return found


def plan_apply(manual: Manual) -> list[tuple[str, str]]:
    """Returns the page and the new text of each page whose quotes differ from their homes."""
    homes = {fact_id: home[1].body for fact_id, home in manual.homes().items()}
    planned: list[tuple[str, str]] = []
    for source in manual.sources:
        updated = facts.regenerate(source.text, homes)
        if updated != source.text:
            planned.append((source.path.relative_to(manual.root).as_posix(), updated))
    return planned


def apply(manual: Manual) -> list[str]:
    """Writes every planned rewrite to its page and returns the pages it wrote."""
    planned = plan_apply(manual)
    for page, _ in planned:
        # read_files replaces undecodable bytes; a rewrite must refuse them, not write them.
        (manual.root / page).read_bytes().decode("utf-8")
    for page, updated in planned:
        (manual.root / page).write_text(updated, encoding="utf-8", newline="")
    return [page for page, _ in planned]


def main(argv: list[str] | None = None) -> int:
    """Runs one subcommand and returns its exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="every fact, its home and the number of places citing it")
    commands.add_parser("where", help="the home of a fact").add_argument("id")
    cited_by = commands.add_parser("cited-by", help="every place that quotes or follows a fact")
    cited_by.add_argument("id")
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
