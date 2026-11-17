# source_blanking.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Blanks the comments and string literals of C-family source, and finds the brace that closes a block."""

from __future__ import annotations

import re

K_RAW_STRING = re.compile(r'R"([^()\\ ]*)\(')


def _ends_word(text: str, index: int) -> bool:
    """Returns whether the character before *index* belongs to a name, so a `R` there is not a prefix."""
    return index > 0 and (text[index - 1].isalnum() or text[index - 1] == "_")


def blank(text: str, *, backtick: bool, raw: bool) -> str:
    """Returns the text with comments and the insides of string literals turned into spaces.

    Offsets and newlines stay, so a position in the result is a position in the text.
    """
    out = list(text)
    length = len(text)

    def erase(start: int, end: int) -> None:
        for index in range(start, min(end, length)):
            if out[index] != "\n":
                out[index] = " "

    index = 0
    while index < length:
        char = text[index]
        following = text[index + 1] if index + 1 < length else ""
        if char == "/" and following == "/":
            end = text.find("\n", index)
            end = length if end == -1 else end
            erase(index, end)
            index = end
        elif char == "/" and following == "*":
            end = text.find("*/", index + 2)
            end = length if end == -1 else end + 2
            erase(index, end)
            index = end
        elif raw and char == "R" and following == '"' and not _ends_word(text, index):
            opening = K_RAW_STRING.match(text, index)
            if opening is None:
                index += 1
                continue
            closing = ")" + opening.group(1) + '"'
            end = text.find(closing, opening.end())
            end = length if end == -1 else end
            erase(opening.end(), end)
            index = end + len(closing)
        elif char in "\"'" or (backtick and char == "`"):
            end = index + 1
            while end < length and text[end] != char:
                end += 2 if text[end] == "\\" else 1
            erase(index + 1, end)
            index = end + 1
        else:
            index += 1
    return "".join(out)


def matching_brace(code: str, opening: int) -> int | None:
    """Returns the offset of the brace that closes the one at *opening*, in text with strings blanked."""
    depth = 0
    for index in range(opening, len(code)):
        if code[index] == "{":
            depth += 1
        elif code[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return None
