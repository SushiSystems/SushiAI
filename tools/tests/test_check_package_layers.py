# test_check_package_layers.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the decisions of tools/layering/check_package_layers.py on layer tables and parsed source."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.small

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layering.check_package_layers import (  # noqa: E402
    K_LAYERED,
    _component_problems,
    _forbidden_names,
    _import_problems,
    _unlayered_keys,
    _upward_imports,
)

K_PACKAGES = {package.name: package for package in K_LAYERED}
K_PROVISION = K_PACKAGES["sushicore.provision"]
K_DOCS_BUNDLE = K_PACKAGES["sushicore.docs_bundle"]

K_COMPONENT = """from dataclasses import dataclass
from typing import final

from rich.text import Text

from ..theme import Theme


@final
@dataclass(frozen=True, slots=True)
class Title:
    text: str

    def render(self, theme: Theme) -> Text:
        return Text(self.text)
"""


def _problems(stem: str, text: str) -> list[str]:
    """Returns every problem a ui file named *stem* holding *text* has, shape first, then imports."""
    tree = ast.parse(text)
    return [problem for _, problem in [*_component_problems(stem, tree), *_import_problems(tree)]]


def test_reports_a_module_without_a_layer():
    assert list(_unlayered_keys(K_PROVISION, ["home", "brand_new", ""])) == ["brand_new"]


def test_reports_every_form_of_an_upward_import():
    home = ast.parse("from . import steps\nfrom sushicore.provision import commands\n")
    cuda = ast.parse("def later():\n    from .. import steps\n")
    found = [*_upward_imports(K_PROVISION, "home", [], home), *_upward_imports(K_PROVISION, "gpu.cuda", ["gpu"], cuda)]
    assert found == [(1, "steps"), (2, "commands"), (2, "steps")]


@pytest.mark.parametrize(
    ("package", "own", "text", "expected"),
    [
        (K_PROVISION, "config", "from . import home\n", []),
        (K_DOCS_BUNDLE, "page_text", "from .publish_list import read\n", [(1, "publish_list")]),
    ],
    ids=["provision allows it", "docs_bundle does not"],
)
def test_a_sideways_import_passes_only_where_the_package_allows_it(package, own, text, expected):
    assert list(_upward_imports(package, own, [], ast.parse(text))) == expected


@pytest.mark.parametrize(
    ("package", "text", "expected"),
    [
        (K_DOCS_BUNDLE, "import typer\n\ndef run():\n    import click\n", [(1, "typer")]),
        (K_PROVISION, "def run():\n    import sushihub.cli\n", [(2, "sushihub.cli")]),
    ],
    ids=["module level only", "nested too"],
)
def test_reports_a_forbidden_import_at_module_level_only_where_asked(package, text, expected):
    assert list(_forbidden_names(package, ast.parse(text))) == expected


def test_accepts_a_component_of_the_right_shape():
    assert _problems("title", K_COMPONENT) == []


@pytest.mark.parametrize(
    ("old", "new", "problem"),
    [
        ("@final\n", "", "Title is not marked final"),
        ("slots=True", "slots=False", "Title is not a frozen dataclass with slots"),
        ("self, theme: Theme", "self, theme: Theme, width: int", "Title.render does not take exactly self and theme"),
        ("from ..theme import Theme", "from ..theme import Theme\nfrom ..console import Console",
         "a component may not import sushicore.console"),
    ],
    ids=["not final", "not a frozen slots dataclass", "a wrong render signature", "a forbidden import"],
)
def test_reports_each_way_a_component_breaks_its_shape(old, new, problem):
    assert _problems("title", K_COMPONENT.replace(old, new)) == [problem]


def test_reports_a_file_whose_class_is_not_named_after_it():
    assert _problems("header", K_COMPONENT) == ["defines ['Title'], not the one public class Header"]
