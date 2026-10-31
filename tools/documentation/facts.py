# facts.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Reads the facts a manual keeps in one home and the places that quote or follow them.

A fact is the text between `<!-- fact:begin ID -->` and `<!-- fact:end ID -->`. A quote is a block
between `<!-- quote:begin ID -->` and `<!-- quote:end ID -->` that holds the fact's text exactly. A
line `<!-- follows: ID -->` marks a paraphrase of the fact. The id is the identity; no position is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

K_ID = r"[a-z0-9][a-z0-9-]*"
K_FACT_BEGIN = re.compile(rf"^<!-- fact:begin ({K_ID}) -->$")
K_FACT_END = re.compile(rf"^<!-- fact:end ({K_ID}) -->$")
K_QUOTE_BEGIN = re.compile(rf"^<!-- quote:begin ({K_ID}) -->$")
K_QUOTE_END = re.compile(rf"^<!-- quote:end ({K_ID}) -->$")
K_FOLLOWS = re.compile(rf"^<!-- follows: ({K_ID}) -->$")


@dataclass(frozen=True, slots=True)
class Block:
    """Holds one fact or one quote: its id, the lines of its markers and the text between them."""

    id: str
    first: int
    last: int
    body: str


@dataclass(slots=True)
class Page:
    """Holds the facts, quotes and follow marks of one page, and the faults in how they are marked."""

    path: str
    facts: list[Block] = field(default_factory=list)
    quotes: list[Block] = field(default_factory=list)
    follows: list[tuple[str, int]] = field(default_factory=list)
    faults: list[tuple[int, str]] = field(default_factory=list)


def parse(path: str, text: str) -> Page:
    """Returns the facts, quotes and follow marks of a page, with the faults of its markers."""
    page = Page(path)
    lines = text.replace("\r\n", "\n").split("\n")
    open_kind = ""
    open_id = ""
    open_line = 0
    inner: list[str] = []
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        begin = K_FACT_BEGIN.match(line) or K_QUOTE_BEGIN.match(line)
        end = K_FACT_END.match(line) or K_QUOTE_END.match(line)
        kind = "fact" if "fact:" in line else "quote"
        if begin:
            if open_kind:
                page.faults.append((number, f"{kind} {begin.group(1)} begins inside {open_kind} {open_id}"))
            else:
                open_kind, open_id, open_line, inner = kind, begin.group(1), number, []
        elif end:
            if not open_kind or end.group(1) != open_id or kind != open_kind:
                page.faults.append((number, f"{kind} {end.group(1)} ends without its beginning"))
            else:
                block = Block(open_id, open_line, number, "\n".join(inner))
                (page.facts if open_kind == "fact" else page.quotes).append(block)
                open_kind = ""
        elif open_kind:
            inner.append(raw)
        else:
            follows = K_FOLLOWS.match(line)
            if follows:
                page.follows.append((follows.group(1), number))
    if open_kind:
        page.faults.append((open_line, f"{open_kind} {open_id} is never closed"))
    return page


def regenerate(text: str, homes: dict[str, str]) -> str:
    """Returns the text with the body of each quote replaced by the body of its fact, where the fact exists."""
    page = parse("", text)
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.replace("\r\n", "\n").split("\n")
    for quote in sorted(page.quotes, key=lambda block: -block.first):
        if quote.id in homes:
            lines[quote.first:quote.last - 1] = homes[quote.id].split("\n") if homes[quote.id] != "" else []
    return newline.join(lines)
