# case_reader.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Reads the test cases of a source file and what each one asserts, compares to a literal and sleeps on.

Supports C++ (GoogleTest), Python (pytest and unittest) and TypeScript (describe, it and test).
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

K_SIZES = ("small", "medium", "large")
K_BENCHMARK = "benchmark"
K_CPP_SUFFIXES = frozenset({".cpp", ".cc", ".cxx"})
K_TS_SUFFIXES = (".test.ts", ".test.tsx")
K_CPP_CASE = re.compile(
    r"^[ \t]*(?:TEST|TEST_F|TEST_P|TYPED_TEST|TYPED_TEST_P)[ \t]*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*\{",
    re.MULTILINE)
K_CPP_ASSERT = re.compile(
    r"\b(?:[A-Za-z0-9]+_)*(?:EXPECT|ASSERT)\w*\s*\(|\bstatic_assert\s*\("
    r"|\b(?:FAIL|SUCCEED|ADD_FAILURE|GTEST_FAIL|GTEST_SUCCEED)\s*\("
    r"|\b(?:expect|assert)_\w+\s*(?:<[^;{}()]*>)?\s*\(")
K_CPP_SLEEP = re.compile(r"\b(?:sleep_for|sleep_until|usleep|Sleep)\s*\(")
K_CPP_LITERAL = re.compile(
    r"\b(?:GTEST_)?SUCCEED\s*\(\s*\)|\b(?:EXPECT|ASSERT)_TRUE\s*\(\s*(?:true|1)\s*\)|\b(?:EXPECT|ASSERT)_FALSE\s*\(\s*(?:false|0)\s*\)"
    r"|\b(?:EXPECT|ASSERT)_EQ\s*\(\s*(-?\d[\w.]*|true|false|\"[^\"]*\")\s*,\s*\1\s*\)")
K_CPP_WEAK = re.compile(
    r"\bstatic_assert\s*\(|\b(?:GTEST_)?SUCCEED\s*\(|\b(?:[A-Za-z0-9]+_)*(?:EXPECT|ASSERT)_NO_THROW\s*\(")
K_NUMBER_LITERAL = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?[fFlL]?"
K_CPP_TOLERANCE = re.compile(
    rf"\b(?:EXPECT|ASSERT)_NEAR\s*\((?:[^;()]|\([^;()]*\))*,\s*{K_NUMBER_LITERAL}\s*\)")
K_TS_DESCRIBE = re.compile(r"^describe\s*\(\s*([\"'`])(.*?)\1", re.MULTILINE)
K_TS_CASE = re.compile(
    r"\b(?:it|test)(?:\.\w+)?\s*\(\s*(?P<q>[\"'`])(?P<name>[^\"'`]*)(?P=q)\s*,\s*(?:async\s*)?"
    r"(?:function\s*\w*\s*\([^)]*\)|\([^)]*\)\s*=>|\w+\s*=>)\s*\{")
K_TS_ASSERT = re.compile(r"\b(?:expect|assert)\w*\s*\(|\bfail\s*\(")
K_TS_SLEEP = re.compile(r"\bsetTimeout\s*\(|\bsleep\s*\(")
K_TS_LITERAL = re.compile(
    r"\bexpect\(\s*(true|false|\d+)\s*\)\s*\.\s*(?:toBe|toEqual|toStrictEqual)\(\s*\1\s*\)"
    r"|\bexpect\(\s*true\s*\)\s*\.\s*toBeTruthy\(")
K_RAW_STRING = re.compile(r'R"([^()\\ ]*)\(')
K_TESTCASE_BASES = frozenset({"TestCase", "unittest.TestCase"})
K_PYTHON_ASSERTING = frozenset({"raises", "warns", "approx", "fail"})
K_PYTHON_TOLERANCE_KEYWORDS = frozenset({"abs", "rel", "delta", "atol", "rtol"})


@dataclass(frozen=True, slots=True)
class Case:
    """Holds one test case: where it is, its size and what its body does."""

    path: Path
    name: str
    first: int
    last: int
    size: str | None
    assertions: int
    literal_lines: tuple[int, ...]
    sleep_lines: tuple[int, ...]
    digest: str = ""
    weak_only: bool = False
    tolerance_lines: tuple[int, ...] = ()


def is_test_file(path: Path) -> bool:
    """Returns whether a file under a test folder declares test cases, by its name."""
    if path.suffix in K_CPP_SUFFIXES or path.suffix == ".py":
        return path.name.startswith("test_")
    return path.name.endswith(K_TS_SUFFIXES)


def cases_of(path: Path, text: str) -> list[Case]:
    """Returns the cases a test file declares, or none for a file of another kind."""
    if path.suffix in K_CPP_SUFFIXES:
        return list(_cpp_cases(path, text))
    if path.suffix == ".py":
        return list(_python_cases(path, text))
    if path.name.endswith(K_TS_SUFFIXES):
        return list(_typescript_cases(path, text))
    return []


def _digest(body: str) -> str:
    """Returns a short digest of a case body with its whitespace collapsed, so a renamed or moved case keeps it."""
    return hashlib.sha256(" ".join(body.split()).encode("utf-8")).hexdigest()[:16]


def _line(text: str, offset: int) -> int:
    """Returns the line number of a character offset."""
    return text.count("\n", 0, offset) + 1


def _ends_word(text: str, index: int) -> bool:
    """Returns whether the character before *index* belongs to a name, so a `R` there is not a prefix."""
    return index > 0 and (text[index - 1].isalnum() or text[index - 1] == "_")


def _blank(text: str, *, backtick: bool, raw: bool) -> str:
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


def _matching_brace(code: str, opening: int) -> int | None:
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


def _lines_of(pattern: re.Pattern[str], text: str, code: str, start: int, end: int) -> tuple[int, ...]:
    """Returns the line of each match of *pattern* in the text between two offsets that is not blanked."""
    return tuple(_line(text, match.start()) for match in pattern.finditer(text, start, end)
                 if code[match.start()] == text[match.start()])


def _cpp_size(suite: str) -> str | None:
    """Returns the size a C++ suite name opens with, after any `Characterization_`; `benchmark` for a benchmark, or None."""
    if suite.startswith("Benchmark_"):
        return K_BENCHMARK
    suite = suite.removeprefix("Characterization_")
    for size in K_SIZES:
        if suite.startswith(f"{size.capitalize()}_"):
            return size
    return None


def _cpp_cases(path: Path, text: str) -> Iterator[Case]:
    """Yields the GoogleTest cases of a C++ file."""
    code = _blank(text, backtick=False, raw=True)
    for match in K_CPP_CASE.finditer(code):
        opening = match.end() - 1
        closing = _matching_brace(code, opening)
        if closing is None:
            continue
        body = code[opening + 1:closing]
        yield Case(
            path=path,
            name=f"{match.group(1)}.{match.group(2)}",
            first=_line(code, match.start()),
            last=_line(code, closing),
            size=_cpp_size(match.group(1)),
            assertions=len(K_CPP_ASSERT.findall(body)),
            literal_lines=_lines_of(K_CPP_LITERAL, text, code, opening + 1, closing),
            sleep_lines=_lines_of(K_CPP_SLEEP, text, code, opening + 1, closing),
            digest=_digest(text[opening + 1:closing]),
            weak_only=0 < len(K_CPP_ASSERT.findall(body)) == len(K_CPP_WEAK.findall(body)),
            tolerance_lines=_lines_of(K_CPP_TOLERANCE, text, code, opening + 1, closing),
        )


def _marks(node: ast.expr) -> set[str]:
    """Returns the last name of each pytest mark an expression names, such as `small` of `pytest.mark.small`."""
    items = node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node]
    return {ast.unparse(item).rsplit(".", 1)[-1] for item in items}


def _size_among(names: set[str]) -> str | None:
    """Returns the one size a set of names holds, or None."""
    found = [size for size in K_SIZES if size in names]
    return found[0] if found else None


def _file_marks(module: ast.Module) -> set[str]:
    """Returns the marks a file sets for all its tests with `pytestmark` or a `K_TEST_SIZE` string."""
    names: set[str] = set()
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if "pytestmark" in targets:
            names |= _marks(node.value)
        elif "K_TEST_SIZE" in targets and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            names.add(node.value.value)
    return names


def is_test_class(node: ast.ClassDef) -> bool:
    """Returns whether a class is collected as a test class: named `Test...` or derived from `TestCase`."""
    return node.name.startswith("Test") or any(
        ast.unparse(base) in K_TESTCASE_BASES for base in node.bases)


def class_size(node: ast.ClassDef) -> str | None:
    """Returns the size a test class sets for its tests, by a size mark or a `K_TEST_SIZE` class attribute."""
    names = {name for item in node.decorator_list for name in _marks(item)}
    for member in node.body:
        if isinstance(member, ast.Assign):
            targets = [target for target in member.targets if isinstance(target, ast.Name)]
            value = member.value
        elif isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
            targets, value = [member.target], member.value
        else:
            continue
        if any(target.id == "K_TEST_SIZE" for target in targets) and isinstance(value, ast.Constant)                 and isinstance(value.value, str):
            names.add(value.value)
    return _size_among(names)


def _python_cases(path: Path, text: str) -> Iterator[Case]:
    """Yields the pytest and unittest test functions of a Python file."""
    try:
        module = ast.parse(text)
    except SyntaxError:
        return
    file_size = _size_among(_file_marks(module))
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            yield _python_case(path, text, node, file_size)
        elif isinstance(node, ast.ClassDef) and is_test_class(node):
            outer_size = class_size(node) or file_size
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and member.name.startswith("test_"):
                    yield _python_case(path, text, member, outer_size)


def _call_name(call: ast.Call) -> str:
    """Returns the name a call invokes, such as `sleep` for `time.sleep(1)`."""
    function = call.func
    if isinstance(function, ast.Attribute):
        return function.attr
    return function.id if isinstance(function, ast.Name) else ""


def _python_case(path: Path, text: str, node: ast.FunctionDef | ast.AsyncFunctionDef, outer_size: str | None) -> Case:
    """Returns the case a Python test function declares; its own mark beats the size of its class or file."""
    own_marks = {name for item in node.decorator_list for name in _marks(item)}
    assertions = 0
    literals: list[int] = []
    sleeps: list[int] = []
    tolerances: list[int] = []
    for inner in ast.walk(node):
        if isinstance(inner, ast.Assert):
            assertions += 1
            if isinstance(inner.test, ast.Constant) and inner.test.value:
                literals.append(inner.lineno)
        elif isinstance(inner, ast.Call):
            name = _call_name(inner)
            if name.lower().startswith(("assert", "expect")) or name in K_PYTHON_ASSERTING:
                assertions += 1
            if name == "sleep":
                sleeps.append(inner.lineno)
            if any(keyword.arg in K_PYTHON_TOLERANCE_KEYWORDS and isinstance(keyword.value, ast.Constant)
                   and isinstance(keyword.value.value, float) for keyword in inner.keywords):
                tolerances.append(inner.lineno)
    return Case(path, node.name, node.lineno, node.end_lineno or node.lineno,
                _size_among(own_marks) or outer_size,
                assertions, tuple(literals), tuple(sleeps),
                _digest(" ".join(ast.get_source_segment(text, statement) or "" for statement in node.body)),
                tolerance_lines=tuple(tolerances))


def _typescript_size(name: str) -> str | None:
    """Returns the size a describe block's name opens with, or None."""
    for size in K_SIZES:
        if name.startswith(f"{size}:"):
            return size
    return None


def _typescript_cases(path: Path, text: str) -> Iterator[Case]:
    """Yields the `it` and `test` cases of a TypeScript file, each sized by the describe above it."""
    code = _blank(text, backtick=True, raw=False)
    describes = [(match.start(), _typescript_size(match.group(2))) for match in K_TS_DESCRIBE.finditer(text)]
    for match in K_TS_CASE.finditer(code):
        opening = match.end() - 1
        closing = _matching_brace(code, opening)
        if closing is None:
            continue
        size = next((found for start, found in reversed(describes) if start < match.start()), None)
        body = code[opening + 1:closing]
        yield Case(
            path=path,
            name=text[match.start("name"):match.end("name")],
            first=_line(code, match.start()),
            last=_line(code, closing),
            size=size,
            assertions=len(K_TS_ASSERT.findall(body)),
            literal_lines=_lines_of(K_TS_LITERAL, text, code, opening + 1, closing),
            sleep_lines=_lines_of(K_TS_SLEEP, text, code, opening + 1, closing),
            digest=_digest(text[opening + 1:closing]),
        )
