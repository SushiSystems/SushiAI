# build_rename_map.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Lists every name the naming ruling changes and writes tools/naming/rename_map.tsv.

Usage: python tools/naming/build_rename_map.py [repository root] [--output PATH]
The design is docs/agent/2026_10_07_ERROR_CONTRACT_AND_NAMING/SPEC.md, section 2.13.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naming.apply_rename import (  # noqa: E402
    K_CODE_ROOTS,
    K_COLUMNS,
    K_SIBLING_NAMESPACES,
    K_RENAME_MAP,
    K_STATUS_RENAME,
    RenameEntry,
    RenameNames,
    code_only,
    read_source,
    rewrite,
    source_files,
    tokenize,
    write_rename_map,
)

K_DOCUMENT_ROOTS = (*K_CODE_ROOTS, "docs")
K_JSON_ROOTS = ("examples", "tests")
K_CLI_ROOT = "cli"
K_SIBLING_ENUMS = ("DataType", "Layout", "ExecutionMode")
K_SIBLING_MACRO = re.compile(r"\b(?:SUSHIBLAS|SUSHIRUNTIME|SB|SR)_[A-Z0-9_]+\b")
K_PREFIXED_MACROS = ("SA_", "SUSHIAI_")
K_ENUMERATOR_EXCEPTIONS = {"NUL": "Null"}
K_STATUS_BY_HAND = "by_hand"
K_STATUS_EXCLUDED = "excluded"
K_STATUS_SIBLING = "sibling"
K_UPPER_NAME = re.compile(r"[A-Z][A-Z0-9_]*")
K_CONSTANT_NAME = re.compile(r"K_[A-Z0-9_]+")
K_ENUM_DECLARATION = re.compile(r"\benum\s+(?:(class|struct)\s+)?(\w+)\s*(:[^{;]*)?\{([^}]*)\}")
K_CONSTANT_KEYWORD = re.compile(r"\bconstexpr\b|\bstatic\s+const\b(?!expr)")
K_DECLARED_NAME = re.compile(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)*$")
K_INTERFACE = re.compile(r"\b(?:class|struct)\s+(I[A-Z]\w*)\b(?!\s*;)")
K_TYPE_DECLARATION = re.compile(r"\b(?:class|struct)\s+([A-Z][A-Z0-9_]*)\b(?!\s*;)")
K_MACRO = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)", re.M)
K_ALIAS = re.compile(r"\busing\s+(\w+)\s*=\s*(?:\w+::)*(\w+)\s*;")
K_QUALIFIED_UPPER = re.compile(r"\b((?:\w+::)*)(\w+)::([A-Z][A-Z0-9_]*)\b")
K_JSON_STRING = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
K_CLI_FLAG = re.compile(r"--[a-z][a-z0-9-]*")
K_DECLARATION_REACH = 400
K_SHORTEST_SAFE_NAME = 3
K_SIBLING_FILE = "(sibling)"
K_REWRITE_COUNTED_KINDS = ("constant", "enumerator", "interface", "macro")
K_PAIR_COUNTED_KINDS = ("sibling_enumerator",)
K_PER_FILE_KINDS = ("unscoped_enum",)


@dataclass(frozen=True, slots=True)
class Declaration:
    """Holds one name a scanner found, before its occurrences are counted."""

    kind: str
    old_name: str
    new_name: str
    file: str
    status: str
    reason: str


def pascal_case(name: str) -> str:
    """Returns an enumerator's PascalCase spelling by the mapping rule of SPEC 2.13."""
    if name in K_ENUMERATOR_EXCEPTIONS:
        return K_ENUMERATOR_EXCEPTIONS[name]
    return "".join(part[:1].upper() + part[1:].lower() for part in name.split("_"))


def is_pascal_case(name: str) -> bool:
    """Returns whether a name already reads as PascalCase."""
    has_lower = any(character.islower() for character in name)
    return "_" not in name and name[:1].isupper() and (has_lower or len(name) == 1)


def upper_snake(name: str) -> str:
    """Returns a name in UPPER_SNAKE, split after each lower-case letter an upper-case one follows.

    A leading k is dropped. A name already in upper case comes back as it is.
    """
    plain = name[1:] if re.match(r"k[A-Z]", name) else name
    return re.sub(r"(?<=[a-z])(?=[A-Z])", "_", plain).upper()


class Tree:
    """Holds the texts of one repository that the scanners read."""

    def __init__(self, root: Path) -> None:
        """Remembers the root; files are read when a scanner first asks."""
        self.root = root

    @cached_property
    def sources(self) -> dict[str, str]:
        """Returns the text of every C++ file under the code roots, by relative path."""
        return {
            path.relative_to(self.root).as_posix(): read_source(path)
            for path in source_files(self.root)
        }

    @cached_property
    def code(self) -> dict[str, str]:
        """Returns each source with its comments and literals blanked."""
        return {file: code_only(text) for file, text in self.sources.items()}

    @cached_property
    def identifiers(self) -> Counter[str]:
        """Returns how often each identifier occurs as a code token."""
        return Counter(
            word for code in self.code.values() for word in re.findall(r"[A-Za-z_]\w*", code)
        )

    @cached_property
    def pairs(self) -> Counter[tuple[str, str]]:
        """Returns how often each `Qualifier::name` pair occurs in code."""
        pattern = re.compile(r"(?=\b(\w+)\s*::\s*(\w+)\b)")
        return Counter(pair for code in self.code.values() for pair in pattern.findall(code))

    @cached_property
    def spellings(self) -> dict[str, tuple[str, int]]:
        """Returns, per lower-case string, the first file that spells it and how often it occurs.

        A string is a whole C++ literal, a JSON string or a command line flag of the CLI.
        """
        seen: dict[str, tuple[str, int]] = {}

        def add(text: str, file: str) -> None:
            key = text.lower().lstrip("-").replace("-", "_")
            seen[key] = (seen.get(key, (file, 0))[0], seen.get(key, (file, 0))[1] + 1)

        for file, text in self.sources.items():
            for token in tokenize(text):
                is_plain = token.text.endswith('"') and "R" not in token.text[:3]
                if token.kind == "literal" and is_plain:
                    add(token.text[token.text.index('"') + 1:-1], file)
        for top, suffix, pattern in (*((top, ".json", K_JSON_STRING) for top in K_JSON_ROOTS),
                                     (K_CLI_ROOT, ".py", K_CLI_FLAG)):
            for path in sorted((self.root / top).rglob(f"*{suffix}")):
                shown = path.relative_to(self.root).as_posix()
                for match in pattern.finditer(path.read_text(encoding="utf-8", errors="replace")):
                    add(match.group(match.lastindex or 0), shown)
        return seen

    @cached_property
    def documents(self) -> tuple[Counter[str], Counter[tuple[str, str]]]:
        """Returns the word counts and the `Qualifier::name` counts of every Markdown file."""
        words: Counter[str] = Counter()
        pairs: Counter[tuple[str, str]] = Counter()
        for top in K_DOCUMENT_ROOTS:
            for path in sorted((self.root / top).rglob("*.md")):
                text = path.read_text(encoding="utf-8", errors="replace")
                words.update(re.findall(r"[A-Za-z_]\w*", text))
                pairs.update(re.findall(r"(?=\b(\w+)::(\w+)\b)", text))
        return words, pairs


def _enumerators(body: str) -> list[str]:
    """Returns the enumerator names of an enum body whose comments are already blanked."""
    names = [re.match(r"\s*([A-Za-z_]\w*)", item) for item in body.split(",")]
    return [match.group(1) for match in names if match]


def find_enumerators(tree: Tree) -> Iterator[Declaration]:
    """Yields the enumerators not in PascalCase, the unsized enums and the unscoped enums."""
    for file, code in tree.code.items():
        for match in K_ENUM_DECLARATION.finditer(code):
            scoped, enum, underlying, body = match.groups()
            names = _enumerators(body)
            if not scoped:
                held = ", ".join(names)
                reason = f"SPEC 2.13: an unscoped enum becomes K_ constants by hand: {held}"
                yield Declaration("unscoped_enum", enum, "", file, K_STATUS_BY_HAND, reason)
                continue
            if not underlying:
                reason = f"SPEC 2.13: the type is added by hand; {len(names)} enumerators"
                yield Declaration(
                    "enum_underlying",
                    enum,
                    f"{enum} : std::uint8_t",
                    file,
                    K_STATUS_BY_HAND,
                    reason,
                )
            targets = Counter(pascal_case(name) for name in names)
            for name in names:
                new = pascal_case(name)
                if is_pascal_case(name):
                    continue
                taken = targets[new] > 1 or tree.pairs[(enum, new)] > 0
                status = K_STATUS_BY_HAND if taken else K_STATUS_RENAME
                reason = f"{enum}::{new} is already taken" if taken else ""
                yield Declaration(
                    "enumerator", f"{enum}::{name}", f"{enum}::{new}", file, status, reason
                )


def _declared_constant(code: str, keyword_end: int) -> str | None:
    """Returns the name a constexpr or static const declares, or None for a function."""
    depth = 0
    for index in range(keyword_end, min(len(code), keyword_end + K_DECLARATION_REACH)):
        character = code[index]
        if character == "<":
            depth += 1
        elif character == ">" and depth > 0:
            depth -= 1
        elif depth == 0 and character in "=;{(":
            declared = code[keyword_end:index].rstrip()
            name = K_DECLARED_NAME.search(declared)
            is_function = character == "(" or re.search(r"\boperator\b", declared) is not None
            return name.group(1) if name and not is_function else None
    return None


def _is_template(code: str, keyword_start: int) -> bool:
    """Returns whether the declaration that holds a keyword opens with `template`."""
    opening = max(code.rfind(mark, 0, keyword_start) for mark in ";{}")
    return re.search(r"\btemplate\b", code[opening + 1:keyword_start]) is not None


def find_constants(tree: Tree) -> Iterator[Declaration]:
    """Yields the constexpr and static const names not spelled K_UPPER_SNAKE."""
    unscoped = {
        name
        for code in tree.code.values()
        for match in K_ENUM_DECLARATION.finditer(code)
        if not match.group(1)
        for name in _enumerators(match.group(4))
    }
    for file, code in tree.code.items():
        for match in K_CONSTANT_KEYWORD.finditer(code):
            name = _declared_constant(code, match.end())
            if name is None or K_CONSTANT_NAME.fullmatch(name):
                continue
            new = "K_" + upper_snake(name)
            if _is_template(code, match.start()) and is_pascal_case(name):
                reason = "D12: a trait variable template keeps its type-like name"
                yield Declaration("constant", name, name, file, K_STATUS_EXCLUDED, reason)
            elif not K_UPPER_NAME.fullmatch(name):
                reason = "lower-case today, so the same word may name a variable; renamed by hand"
                yield Declaration("constant", name, new, file, K_STATUS_BY_HAND, reason)
            elif len(name) < K_SHORTEST_SAFE_NAME:
                reason = "too short for a whole-word replacement; renamed by hand"
                yield Declaration("constant", name, new, file, K_STATUS_BY_HAND, reason)
            elif name in unscoped or tree.identifiers[new] > 0:
                taken = f"{new} is already a token"
                why = "an unscoped enum also declares it" if name in unscoped else taken
                yield Declaration("constant", name, new, file, K_STATUS_BY_HAND, why)
            else:
                yield Declaration("constant", name, new, file, K_STATUS_RENAME, "")


def find_interfaces(tree: Tree) -> Iterator[Declaration]:
    """Yields the classes and structs named I followed by an upper-case letter."""
    for file, code in tree.code.items():
        for match in K_INTERFACE.finditer(code):
            old = match.group(1)
            new = old[1:]
            headers = [path for path in tree.sources if Path(path).stem == old.lower()]
            moves = f"{headers[0]} moves by hand with its type" if headers else ""
            if tree.identifiers[new] > 0:
                reason = f"{new} is already a token {tree.identifiers[new]} times"
                yield Declaration(
                    "interface",
                    old,
                    new,
                    file,
                    K_STATUS_BY_HAND,
                    "; ".join(filter(None, (reason, moves))),
                )
            else:
                yield Declaration("interface", old, new, file, K_STATUS_RENAME, moves)


def find_macros(tree: Tree) -> Iterator[Declaration]:
    """Yields every macro the tree defines; a prefixed one is listed as excluded."""
    for file, text in tree.sources.items():
        blanked = "".join(
            " " * len(token.text) if token.kind == "comment" else token.text
            for token in tokenize(text)
        )
        for match in K_MACRO.finditer(blanked):
            name = match.group(1)
            if name.startswith(K_PREFIXED_MACROS):
                reason = "already prefixed; D12 keeps SUSHIAI_VERSION_*"
                yield Declaration("macro", name, name, file, K_STATUS_EXCLUDED, reason)
            else:
                yield Declaration("macro", name, f"SA_{name}", file, K_STATUS_RENAME, "")


def find_sibling_names(tree: Tree) -> Iterator[Declaration]:
    """Yields the upper-case names SushiBLAS or SushiRuntime declare and this tree uses."""
    sibling_enums = set(K_SIBLING_ENUMS)
    local_enums = set()
    for code in tree.code.values():
        sibling_enums.update(
            alias for alias, target in K_ALIAS.findall(code) if target in K_SIBLING_ENUMS
        )
        local_enums.update(match.group(2) for match in K_ENUM_DECLARATION.finditer(code))
    for code in tree.code.values():
        for path, qualifier, name in K_QUALIFIED_UPPER.findall(code):
            if qualifier in local_enums or K_CONSTANT_NAME.fullmatch(name):
                continue
            if qualifier in sibling_enums:
                reason = "D11: SushiBLAS v0.1.0 is asked to rename it; it follows the sibling"
                new = f"{qualifier}::{pascal_case(name)}"
                yield Declaration(
                    "sibling_enumerator",
                    f"{qualifier}::{name}",
                    new,
                    K_SIBLING_FILE,
                    K_STATUS_SIBLING,
                    reason,
                )
            elif (path + qualifier).split("::")[0] in K_SIBLING_NAMESPACES:
                reason = "declared by a sibling and outside D11's request"
                yield Declaration(
                    "sibling_name", name, "", K_SIBLING_FILE, K_STATUS_SIBLING, reason
                )
        for name in sorted(set(K_SIBLING_MACRO.findall(code))):
            yield Declaration(
                "sibling_name", name, "", K_SIBLING_FILE, K_STATUS_SIBLING, "a sibling's macro"
            )


def find_shared_type_names(tree: Tree) -> Iterator[Declaration]:
    """Yields each class or struct that has the name of an enumerator the map renames.

    The row keeps the type's tokens out of the list of unresolved enumerator names.
    """
    owners = {
        found.old_name.rpartition("::")[2]: found.old_name
        for found in find_enumerators(tree)
        if found.kind == "enumerator"
    }
    for file, code in tree.code.items():
        for match in K_TYPE_DECLARATION.finditer(code):
            name = match.group(1)
            if name in owners:
                reason = f"a type; {owners[name]} changes in its qualified form only"
                yield Declaration("type_name", name, name, file, K_STATUS_EXCLUDED, reason)


K_SCANNERS: tuple[Callable[[Tree], Iterator[Declaration]], ...] = (
    find_shared_type_names,
    find_enumerators,
    find_constants,
    find_interfaces,
    find_macros,
    find_sibling_names,
)


def _merged(declarations: Iterator[Declaration]) -> list[Declaration]:
    """Returns one declaration per kind and old name, its files joined in first-seen order.

    A kind of K_PER_FILE_KINDS keeps one declaration per file, since its reason names that
    file's content.
    """
    files: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    first: dict[tuple[str, str, str], Declaration] = {}
    for declaration in declarations:
        apart = declaration.file if declaration.kind in K_PER_FILE_KINDS else ""
        key = (declaration.kind, declaration.old_name, apart)
        first.setdefault(key, declaration)
        if declaration.file not in files[key]:
            files[key].append(declaration.file)
    return [replace(first[key], file="; ".join(files[key])) for key in sorted(first)]


def _string_spellings(tree: Tree, declarations: list[Declaration]) -> Iterator[RenameEntry]:
    """Yields one excluded row per string that spells a renamed name, in any letter case."""
    seen: set[str] = set()
    for declaration in declarations:
        key = declaration.old_name.rpartition("::")[2].lower()
        if declaration.status == K_STATUS_RENAME and key in tree.spellings and key not in seen:
            seen.add(key)
            file, count = tree.spellings[key]
            spelled = declaration.old_name
            reason = f"a string spells {spelled}; literals, JSON and flags are never rewritten"
            yield RenameEntry(
                "string_spelling", key, key, file, count, 0, K_STATUS_EXCLUDED, reason
            )


def count_occurrences(tree: Tree, declarations: list[Declaration]) -> dict[Declaration, int]:
    """Returns the code tokens each declaration stands for.

    A kind the tool rewrites counts what a pass that applies every such row would rename. A
    sibling's enumerator counts its `Qualifier::name` pairs. Any other kind counts its tokens.
    """
    trial = [
        RenameEntry(
            declaration.kind,
            declaration.old_name,
            declaration.new_name or "-",
            declaration.file,
            0,
            0,
            K_STATUS_RENAME if declaration.kind in K_REWRITE_COUNTED_KINDS else declaration.status,
            "",
        )
        for declaration in declarations
    ]
    names = RenameNames.from_entries(trial)
    renamed: Counter[str] = Counter()
    for text in tree.sources.values():
        renamed.update(rewrite(text, names).counts)
    counts: dict[Declaration, int] = {}
    for declaration in declarations:
        qualifier, _, name = declaration.old_name.rpartition("::")
        if declaration.kind in K_REWRITE_COUNTED_KINDS:
            counts[declaration] = renamed[declaration.old_name]
        elif declaration.kind in K_PAIR_COUNTED_KINDS:
            counts[declaration] = tree.pairs[(qualifier, name)]
        else:
            counts[declaration] = tree.identifiers[declaration.old_name]
    return counts


def build_rename_map(root: Path) -> list[RenameEntry]:
    """Returns the rows of the rename map for one repository, sorted by kind, old name and file."""
    tree = Tree(root)
    declarations = _merged(found for scanner in K_SCANNERS for found in scanner(tree))
    counts = count_occurrences(tree, declarations)
    words, pairs = tree.documents
    entries: list[RenameEntry] = []
    for declaration in declarations:
        qualifier, _, name = declaration.old_name.rpartition("::")
        in_documents = pairs[(qualifier, name)] if qualifier else words[name]
        entries.append(
            RenameEntry(
                declaration.kind,
                declaration.old_name,
                declaration.new_name,
                declaration.file,
                counts[declaration],
                in_documents,
                declaration.status,
                declaration.reason,
            )
        )
    entries.extend(_string_spellings(tree, declarations))
    return sorted(entries, key=lambda entry: (entry.kind, entry.old_name, entry.declaring_file))


def list_unresolved(root: Path, entries: list[RenameEntry]) -> list[str]:
    """Returns `file:line: name` for each token a run of the whole map would leave unresolved."""
    names = RenameNames.from_entries(entries)
    found: list[str] = []
    for path in source_files(root):
        shown = path.relative_to(root).as_posix()
        found.extend(f"{shown}:{left}" for left in rewrite(read_source(path), names).unresolved)
    return found


def main(argv: list[str] | None = None) -> int:
    """Builds the map from the command line, writes it and prints its size and what it leaves."""
    parser = argparse.ArgumentParser(description="Builds the rename map of the naming wave.")
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, help=f"default: {K_RENAME_MAP.as_posix()}")
    arguments = parser.parse_args(argv)
    entries = build_rename_map(arguments.root)
    write_rename_map(arguments.output or arguments.root / K_RENAME_MAP, entries)
    sizes = Counter((entry.kind, entry.status) for entry in entries)
    for (kind, status), count in sorted(sizes.items()):
        print(f"{kind}\t{status}\t{count}")
    print(f"total\t\t{len(entries)}")
    unresolved = list_unresolved(arguments.root, entries)
    for left in unresolved:
        print(f"unresolved: {left}")
    print(f"unresolved\t\t{len(unresolved)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
