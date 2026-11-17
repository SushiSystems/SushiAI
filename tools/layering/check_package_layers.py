# check_package_layers.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Checks the shape of SushiCore's packages: layered imports, forbidden imports and ui components.

Usage: python tools/layering/check_package_layers.py [repository root] [--report] [--rule NAME]
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, Issue, run  # noqa: E402

K_SOURCE = Path("source")


@dataclass(frozen=True, slots=True)
class LayeredPackage:
    """Holds one package's layer table and what its modules may not import.

    `strict` forbids an import within one layer; `nested` reads imports inside functions too;
    each name in `grouped` is a sub-package whose modules share its one layer.
    """

    name: str
    layers: dict[str, int]
    strict: bool
    nested: bool
    grouped: frozenset[str] = frozenset()
    forbidden: frozenset[str] = frozenset()


K_LAYERED = (
    LayeredPackage(
        name="sushicore.backlog",
        layers={
            "errors": 0, "model": 1, "store": 2, "order": 2,
            "check": 3, "importer": 3, "editor": 3, "commands": 4, "__main__": 5,
        },
        strict=True,
        nested=False,
        forbidden=frozenset({"typer", "click"}),
    ),
    LayeredPackage(
        name="sushicore.design",
        layers={
            "errors": 0, "model": 1, "store": 2, "check": 3, "merge": 3, "commands": 4, "__main__": 5,
        },
        strict=True,
        nested=False,
        forbidden=frozenset({"typer", "click"}),
    ),
    LayeredPackage(
        name="sushicore.docs_bundle",
        layers={
            "errors": 0, "markdown_scan": 0,
            "page_text": 1, "publish_list": 1, "api_reference": 1, "bundle_archive": 1,
            "source_revision": 1,
            "page_order": 2, "page_set": 3, "bundle_manifest": 4, "producer": 5,
            "commands": 6, "__main__": 6,
        },
        strict=True,
        nested=False,
        forbidden=frozenset({"typer", "click"}),
    ),
    LayeredPackage(
        name="sushicore.testing",
        layers={"labels": 0, "options": 0, "result_files": 0, "shards": 0, "gtest_filter": 0, "junit_report": 0, "accelerator_slots": 0, "module_tests": 1, "job_count": 1, "shard_registrations": 1, "slow_rerun": 2, "rerun": 2, "program_runner": 2, "filtered_run": 2, "budget": 3, "session": 4, "commands": 5},
        strict=True,
        nested=False,
        forbidden=frozenset({"typer", "click"}),
    ),
    LayeredPackage(
        name="sushicore.provision",
        layers={
            "_output": 0, "config": 0, "home": 0, "manifests": 0, "integrity": 0, "closure": 1, "closure_memory": 1,
            "download_verifier": 1, "selection": 2,
            "system": 1, "probe": 1, "toolchains.stamp": 1, "gpu.compiler_identity": 1,
            "gpu.windows_installer": 1, "fragments": 1, "registry": 1, "lock": 1, "doctor": 1,
            "pipeline": 1, "sinks": 2, "gpu.backend": 2, "gpu.cuda": 2, "gpu.cuda_hardware": 1,
            "gpu.cuda_redist": 1, "gpu.cuda_selection": 1, "gpu.cuda_toolkit_installer": 1,
            "gpu.toolkit_choice": 0, "gpu.toolkit_choice_file": 0, "gpu.gpu_hardware": 0,
            "gpu.toolkit_vendor": 0, "gpu.chosen_toolkit_locator": 2, "gpu.cuda_vendor": 2,
            "gpu.amd_hardware": 1, "gpu.rocm_package": 0, "gpu.rocm_tarball_index": 1,
            "gpu.rocm_selection": 1, "gpu.rocm_toolkit_installer": 1, "gpu.rocm_vendor": 2,
            "gpu.cuda_release": 0, "gpu.rocm": 2,
            "gpu.level_zero": 2, "gpu.registry": 2, "gpu.provisioning": 2, "gpu.adapter_builder": 2,
            "gpu": 2, "checks": 2, "packages": 3, "toolchains": 4, "toolchains.intel_llvm": 4,
            "toolchains.adaptivecpp": 4, "toolchains.oneapi": 4, "toolchains._process": 1,
            "steps": 5, "outcome": 0, "module_manifest": 2, "machine_seams": 3, "module_provision": 5, "link_command": 6, "module_plan": 6,
            "doctor_command": 7, "setup_command": 8, "commands": 9, "sushi_sources": 1, "git_client": 4,
            "sushi_fetch": 5, "pin_chain": 2, "sushi_setup": 5, "module_checks": 5, "sushi_source_check": 5,
            "links": 0, "user_environment": 0, "user_environment_memory": 0, "migrate": 4, "migration_error": 0, "migration_journal": 0, "migration_disk": 1, "migration_plan": 2, "migration_movers": 3,
            "sushi_packages": 2, "package_install": 1, "package_command": 7,
            "fragments_memory": 1, "pipeline_memory": 1, "sinks_memory": 0, "git_client_memory": 4, "gpu.backend_memory": 2, "gpu.adapter_builder_memory": 2,
            "gpu.toolkit_vendor_memory": 1, "gpu.cuda_selection_memory": 1, "gpu.cuda_vendor_memory": 2,
            "gpu.rocm_vendor_memory": 2,
            "gpu.windows_installer_memory": 1,
        },
        strict=False,
        nested=True,
        grouped=frozenset({"packages", "sushi_packages"}),
        forbidden=frozenset({"sushihub"}),
    ),
)
K_COMPONENTS = "sushicore.ui"
K_COMPONENT_EXEMPT = frozenset({"__init__.py", "component.py"})
K_COMPONENT_IMPORTS = frozenset({"sushicore.theme", "sushicore.brand", "sushicore.markup", "sushicore.ui.component"})
K_COMPONENT_PACKAGES = frozenset({"rich"})


@dataclass(frozen=True, slots=True)
class Repository:
    """Holds the root of a repository whose packages sit under source/."""

    root: Path

    def folder(self, package: str) -> Path:
        """Returns the folder of a dotted package name."""
        return self.root / K_SOURCE / Path(*package.split("."))


def collect(paths: list[Path]) -> Iterable[Repository]:
    """Returns each given folder as a repository."""
    return [Repository(path) for path in paths if path.is_dir()]


def _parse(path: Path) -> ast.Module:
    """Returns the syntax tree of a Python file."""
    return ast.parse(path.read_text(encoding="utf-8"))


def _key(package: LayeredPackage, folder: Path, path: Path) -> str:
    """Returns the layer-table key of a module, "" for the package's own __init__."""
    parts = [part for part in path.relative_to(folder).with_suffix("").parts if part != "__init__"]
    if parts and parts[0] in package.grouped:
        return parts[0]
    return ".".join(parts)


def _imported(tree: ast.Module, nested: bool) -> Iterator[ast.Import | ast.ImportFrom]:
    """Yields the import statements of a file, at module level only unless nested."""
    nodes = ast.walk(tree) if nested else iter(tree.body)
    yield from (node for node in nodes if isinstance(node, (ast.Import, ast.ImportFrom)))


def _targets(package: LayeredPackage, own_package: list[str], node: ast.AST) -> Iterator[tuple[int, str]]:
    """Yields the line and the layer-table key of each module of the package an import names.

    *own_package* holds the folders between the package and the importing file.
    """
    if isinstance(node, ast.Import):
        bases = [alias.name for alias in node.names]
        named: list[list[str]] = [[]]
    else:
        if node.level:
            base = own_package[: len(own_package) - (node.level - 1)] if node.level > 1 else own_package
            bases = [".".join([package.name, *base, *([node.module] if node.module else [])])]
        else:
            bases = [node.module or ""]
        named = [[]] + [[alias.name] for alias in node.names if alias.name != "*"]
    prefix = package.name + "."
    for base in bases:
        for extra in named:
            dotted = ".".join([base, *extra])
            if not dotted.startswith(prefix):
                continue
            parts = dotted[len(prefix):].split(".")
            key = parts[0] if parts[0] in package.grouped else ".".join(parts)
            if key in package.layers:
                yield node.lineno, key


def _modules(repository: Repository, package: LayeredPackage) -> Iterator[tuple[Path, Path]]:
    """Yields the folder of a layered package and each Python file in it."""
    folder = repository.folder(package.name)
    if folder.is_dir():
        yield from ((folder, path) for path in sorted(folder.rglob("*.py")))


def _unlayered_keys(package: LayeredPackage, keys: Iterable[str]) -> Iterator[str]:
    """Yields each layer-table key the package's table does not list, the package's own "" aside."""
    yield from (key for key in keys if key and key not in package.layers)


def rule_unlayered_module(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each module of a layered package that its layer table does not list."""
    for package in K_LAYERED:
        for folder, path in _modules(repository, package):
            for key in _unlayered_keys(package, [_key(package, folder, path)]):
                yield Issue(path, 1, f"{key} has no layer in the {package.name} table")


def _upward_imports(
    package: LayeredPackage, own: str, own_package: list[str], tree: ast.Module,
) -> Iterator[tuple[int, str]]:
    """Yields the line and the target of each import of a higher layer, or the same one when strict.

    *own* is the importing module's layer-table key and *own_package* the folders between the
    package and the importing file.
    """
    for node in _imported(tree, package.nested):
        for line, target in _targets(package, own_package, node):
            upward = package.layers[target] >= package.layers[own] if package.strict \
                else package.layers[target] > package.layers[own]
            if target != own and upward:
                yield line, target


def rule_upward_import(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each import of a module in a higher layer, or the same one when strict."""
    for package in K_LAYERED:
        for folder, path in _modules(repository, package):
            own = _key(package, folder, path)
            if own not in package.layers:
                continue
            own_package = list(path.relative_to(folder).parts[:-1])
            for line, target in _upward_imports(package, own, own_package, _parse(path)):
                yield Issue(path, line, f"{own} imports {target}, which is not in a lower layer")


def _forbidden_names(package: LayeredPackage, tree: ast.Module) -> Iterator[tuple[int, str]]:
    """Yields the line and the name of each import the package's modules may not make."""
    for node in _imported(tree, package.nested):
        names = [alias.name for alias in node.names] if isinstance(node, ast.Import) \
            else [node.module or ""] if not node.level else []
        for name in names:
            if name.split(".")[0] in package.forbidden:
                yield node.lineno, name


def rule_forbidden_import(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each import a layered package's modules may not make."""
    for package in K_LAYERED:
        for _, path in _modules(repository, package):
            for line, name in _forbidden_names(package, _parse(path)):
                yield Issue(path, line, f"{package.name} may not import {name}")


def _pascal(stem: str) -> str:
    """Returns a file stem in Pascal case."""
    return "".join(word.capitalize() for word in stem.split("_"))


def _decorator_names(definition: ast.ClassDef) -> dict[str, ast.expr]:
    """Returns a class's decorators by their last name."""
    named = {}
    for decorator in definition.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        named[ast.unparse(target).split(".")[-1]] = decorator
    return named


def _component_problems(stem: str, tree: ast.Module) -> Iterator[tuple[int, str]]:
    """Yields the line and the problem of each way a ui file named *stem* breaks the component shape."""
    expected = _pascal(stem)
    public = [node for node in tree.body if isinstance(node, ast.ClassDef) and not node.name.startswith("_")]
    if [node.name for node in public] != [expected]:
        yield 1, f"defines {[node.name for node in public]}, not the one public class {expected}"
        return
    definition = public[0]
    decorators = _decorator_names(definition)
    if "final" not in decorators:
        yield definition.lineno, f"{expected} is not marked final"
    dataclass = decorators.get("dataclass")
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in getattr(dataclass, "keywords", [])}
    if keywords.get("frozen") != "True" or keywords.get("slots") != "True":
        yield definition.lineno, f"{expected} is not a frozen dataclass with slots"
    render = [node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == "render"]
    if not render or [argument.arg for argument in render[0].args.args] != ["self", "theme"]:
        yield definition.lineno, f"{expected}.render does not take exactly self and theme"


def _absolute(node: ast.ImportFrom) -> str:
    """Returns the module a ui file's from-import names, a relative one resolved."""
    if not node.level:
        return node.module or ""
    package = K_COMPONENTS.split(".")
    base = package[: len(package) - (node.level - 1)]
    return ".".join([*base, *([node.module] if node.module else [])])


def _import_problems(tree: ast.Module) -> Iterator[tuple[int, str]]:
    """Yields the line and the problem of each import a ui component may not make."""
    for node in ast.walk(tree):
        names = [alias.name for alias in node.names] if isinstance(node, ast.Import) \
            else [_absolute(node)] if isinstance(node, ast.ImportFrom) else []
        for name in names:
            root = name.split(".")[0]
            if root not in sys.stdlib_module_names and root not in K_COMPONENT_PACKAGES \
                    and name not in K_COMPONENT_IMPORTS:
                yield node.lineno, f"a component may not import {name}"


def rule_ui_component(repository: Repository) -> Iterator[Issue]:
    """Yields an issue for each ui file that is not one final frozen component importing what it may."""
    folder = repository.folder(K_COMPONENTS)
    for path in sorted(folder.glob("*.py")) if folder.is_dir() else []:
        if path.name in K_COMPONENT_EXEMPT:
            continue
        tree = _parse(path)
        for line, problem in [*_component_problems(path.stem, tree), *_import_problems(tree)]:
            yield Issue(path, line, problem)


K_CHECKER = Checker(
    description="Checks SushiCore's package layers, forbidden imports and ui components.",
    collect=collect,
    rules={
        "rule_unlayered_module": rule_unlayered_module,
        "rule_upward_import": rule_upward_import,
        "rule_forbidden_import": rule_forbidden_import,
        "rule_ui_component": rule_ui_component,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))
