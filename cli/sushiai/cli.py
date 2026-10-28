# cli.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""SushiAI developer CLI (`sa`).

Thin Typer layer: commands only parse arguments and delegate to the service
layer in ``sushiai.services``.
"""

from __future__ import annotations

from typing import Optional

import typer
from sushicore.diag_commands import register_diagnostic_commands
from sushicore.backlog import register_backlog_commands
from sushicore.design import register_design_commands
from sushicore.docs_bundle import ApiSource, register_docs_commands
from sushicore.entry import run as run_entry
from sushicore.provision.commands import ModuleProvision, register_provision_commands
from sushicore.root_options import register_root_options
from sushicore.typer_help import help_group

from . import console
from .aliases import K_ALIASES
from .config import PROFILE, find_project_root, load_config
from .services import profiler as profiler_svc
from .services import project as project_svc
from .services.diag import DIAGNOSTICS
from .services.project import K_DEFAULT_TYPE, BuildType, Suite
from .services.sanitizer import Sanitizer

K_DISTRIBUTION = "sushiai-cli"
K_PROJECT = "Project"
K_MODELS = "Models"
K_DIAGNOSTICS = "Diagnostics"
K_HELP_FLAG = "--help"
K_TRAIN_TARGET = "sushiai_train"
K_INFERENCE_TARGET = "sushiai_inference"
K_IMPORT_TARGET = "sushiai_import"
K_TOKENIZE_TARGET = "sushiai_tokenize"
K_API_XML = "build/docs/api-site/xml"
K_FORWARDING = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
    "help_option_names": [],
}
K_DEMO_MLP_EXAMPLES = (
    "sa demo mlp  # Train with the defaults\n"
    "sa demo mlp --epochs 4 --optimizer sgd  # Flags go to the demo unchanged"
)

_help_group = help_group(console.current)

app = typer.Typer(
    name="sa",
    cls=_help_group,
    help="SushiAI developer CLI — build, test, and run SushiAI.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

register_root_options(app, distribution=K_DISTRIBUTION)


def _show_page_while_unbuilt(ctx: typer.Context, target: str) -> None:
    """Prints the command's own help page when help was asked for and *target* is not built."""
    if K_HELP_FLAG in ctx.args and not project_svc.is_built(target):
        typer.echo(ctx.get_help())


def _print_error(message: str) -> None:
    """Prints one error line through the console, which is built only now."""
    console.error(message)


@app.command(
    "build",
    rich_help_panel=K_PROJECT,
    epilog="sa build  # Build the debug configuration, in build/default-debug\n"
           "sa build --type release  # Its own tree, build/default-release\n"
           "sa build --sanitize address-undefined  # build/asan-debug\n"
           "sa build --clean  # Wipe the build tree first\n"
           "sa build --report  # Measure time and memory per kind of edge\n"
           "sa build --distributed  # Also build the distributed layer and sushiai_worker\n"
           "sa build --sycl-targets \"spir64;nvidia_gpu_sm_86\"  # AOT device targets\n"
           "sa build -D SA_BUILD_TESTS=OFF  # Pass a CMake cache variable through",
)
def build(
    type: BuildType = typer.Option(
        K_DEFAULT_TYPE, "--type", "-t", case_sensitive=False,
        help="Build type: release | debug | relwithdebinfo. Each type has its own tree, "
             "build/<name>-<type>."),
    clean: bool = typer.Option(
        False, "--clean", help="Delete the build tree before configuring."),
    define: Optional[list[str]] = typer.Option(
        None, "--define", "-D", metavar="VAR=VALUE",
        help="Pass a CMake cache variable through to configure. Repeatable. "
             "Quote values containing ';', e.g. -D SA_SANITIZE=\"address;undefined\"."),
    sanitize: Optional[Sanitizer] = typer.Option(
        None, "--sanitize", case_sensitive=False,
        help="Sanitizers for host code: none | address-undefined | thread. "
             "A set builds in a tree of its own, build/asan-<type> or build/tsan-<type>."),
    sycl_targets: Optional[str] = typer.Option(
        None, "--sycl-targets", metavar="LIST",
        help="Ahead-of-time device targets, e.g. \"spir64;nvidia_gpu_sm_86\". They go to the "
             "SushiBLAS package, so another list is another package. Left out, the list of "
             "the last build stays; an empty string clears it."),
    distributed: Optional[bool] = typer.Option(
        None, "--distributed/--no-distributed",
        help="Build the distributed layer of the stack, modules/core/distribution and "
             "sushiai_worker, or stop building them. Left out, the setting of the last "
             "build stays."),
    report: bool = typer.Option(
        False, "--report",
        help="After the build, print the count and wall time per kind of edge, the peak "
             "memory and the slowest edges, and write build_report.json into the tree."),
    jobs: Optional[int] = typer.Option(
        None, "--jobs", "-j", min=1,
        help="Compile jobs. When omitted the hardware threads run, and "
             "CMAKE_BUILD_PARALLEL_LEVEL is honoured."),
) -> None:
    """Configure and build the project.

    Cache variables set with -D persist in the build tree, so dropping the
    flag on a later run does not unset them; pass VAR= to clear one, or
    --clean to start over.
    """
    raise typer.Exit(project_svc.build(type, clean, define, sanitize, distributed, sycl_targets,
                                       report=report, jobs=jobs))


@app.command(
    "test",
    rich_help_panel=K_PROJECT,
    epilog="sa test  # Run the small and medium tests\n"
           "sa test --label small\n"
           "sa test --type release  # Test build/default-release\n"
           "sa test --filter Small_Version --repeat 5  # Hunt a flaky test",
)
def test(
    suite: Suite = typer.Option(
        Suite.gating, "--label", "-l", case_sensitive=False,
        help="Which size to run: small, medium or large; 'gating' runs small and medium "
             "without the quarantined, 'all' runs every CTest test, and 'package' runs "
             "none of them and builds tests/package/ against a fresh install prefix."),
    filter: Optional[str] = typer.Option(
        None, "--filter", "-f", help="ctest -R regex over 'Suite.Case' test names."),
    repeat: int = typer.Option(
        0, "--repeat", "-r", min=0,
        help="Re-run each test up to N times, stopping on the first failure "
             "(ctest --repeat until-fail). Handy for hunting flaky tests."),
    jobs: Optional[int] = typer.Option(
        None, "--jobs", "-j", min=1,
        help="Tests run at once. When omitted ctest decides, and "
             "CTEST_PARALLEL_LEVEL is honoured."),
    memory_limit: Optional[float] = typer.Option(
        None, "--memory-limit", min=0.0, metavar="GB",
        help="Ceiling on the memory the run may commit, in gigabytes; 0 applies none. "
             "Defaults to SA_MEMORY_LIMIT_GB, or half of the machine's memory."),
    cli: bool = typer.Option(
        False, "--cli", help="Run this CLI's own pytest tests of the label instead."),
    type: BuildType = typer.Option(
        K_DEFAULT_TYPE, "--type", "-t", case_sensitive=False,
        help="The build type whose tree runs the tests: release | debug | relwithdebinfo."),
    sanitize: Optional[Sanitizer] = typer.Option(
        None, "--sanitize", case_sensitive=False,
        help="Run the tests of the sanitizer tree: address-undefined | thread."),
) -> None:
    """Run the test suite via CTest labels.

    For GTest-level knobs (per-case filtering, shuffle, break-on-failure) run
    a tier's program directly: sa run sushiai_core_tests -- --gtest_filter=Small_Version.*

    --label package is the exception: it runs no CTest at all, so --filter,
    --repeat and --jobs are rejected there.
    """
    raise typer.Exit(project_svc.test(suite, filter, repeat, memory_limit, cli, jobs,
                                      build_type=type, sanitizer=sanitize))


@app.command(
    "run",
    rich_help_panel=K_PROJECT,
    epilog="sa run  # Run the configured default target\n"
           "sa run sushiai_example\n"
           "sa run --type release sushiai_example  # Run from build/default-release\n"
           "sa run --sort  # Pick an executable from a list\n"
           "sa run sushiai_core_tests -- --gtest_filter=Small_Version.*  # Pass the rest on",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run(
    ctx: typer.Context,
    target: Optional[str] = typer.Argument(
        None, help="Executable name to run (exact, then substring match)."),
    sort: bool = typer.Option(
        False, "--sort", help="Interactively pick an executable."),
    memory_limit: Optional[float] = typer.Option(
        None, "--memory-limit", min=0.0, metavar="GB",
        help="Ceiling on the memory the run may commit, in gigabytes; 0 applies none. "
             "Defaults to SA_MEMORY_LIMIT_GB, or half of the machine's memory."),
    type: BuildType = typer.Option(
        K_DEFAULT_TYPE, "--type", "-t", case_sensitive=False,
        help="The build type whose tree holds the executable: release | debug | relwithdebinfo."),
    sanitize: Optional[Sanitizer] = typer.Option(
        None, "--sanitize", case_sensitive=False,
        help="Run from the sanitizer tree: address-undefined | thread."),
) -> None:
    """Run a built executable. Args after `--` are forwarded to it."""
    raise typer.Exit(project_svc.run(target=target, sort=sort, app_args=list(ctx.args),
                                     memory_limit_gb=memory_limit, build_type=type,
                                     sanitizer=sanitize))


@app.command(
    "clean",
    rich_help_panel=K_PROJECT,
    epilog="sa clean",
)
def clean() -> None:
    """Remove build/, every build tree and the generated API reference included."""
    raise typer.Exit(project_svc.clean())


def _api_source() -> ApiSource:
    """Returns how `sa` builds the API reference and where Doxygen leaves its XML."""
    return ApiSource(build=project_svc.doxygen, xml_dir=find_project_root() / K_API_XML)


def _docs_alias() -> None:
    """Builds the API reference; the callback behind the old spelling `sa doxygen`."""
    raise typer.Exit(project_svc.doxygen())


register_docs_commands(
    app,
    program=PROFILE.program,
    panel=K_PROJECT,
    project_root=find_project_root,
    api=_api_source,
    report=lambda line: console.success(line),
    group_cls=_help_group,
)

register_backlog_commands(
    app,
    program=PROFILE.program,
    panel=K_PROJECT,
    project_root=find_project_root,
    report=lambda line: console.success(line),
    group_cls=_help_group,
)

register_design_commands(
    app,
    program=PROFILE.program,
    panel=K_PROJECT,
    project_root=find_project_root,
    report=lambda line: console.success(line),
    group_cls=_help_group,
)

K_ALIASES.command(app, "doxygen", _docs_alias, "sa doxygen")


demo_app = typer.Typer(
    cls=_help_group,
    help="Run a SushiAI showcase demo.",
    no_args_is_help=True,
    epilog=K_DEMO_MLP_EXAMPLES,
)
app.add_typer(demo_app, name="demo", rich_help_panel=K_MODELS)


@demo_app.command("mlp", epilog=K_DEMO_MLP_EXAMPLES, context_settings=K_FORWARDING)
def demo_mlp(ctx: typer.Context) -> None:
    """Train an MLP end to end and report the compile count (AI-7 showcase).

    Flags are forwarded verbatim to the demo; once it is built,
    `sa demo mlp --help` prints its full list.
    """
    _show_page_while_unbuilt(ctx, project_svc.demo_target("mlp") or "")
    raise typer.Exit(project_svc.demo("mlp", list(ctx.args)))


@app.command(
    "train",
    rich_help_panel=K_MODELS,
    epilog="sa train applications/examples/heavy_mlp/train.json\n"
           "sa train train.json --profile profile.json",
    context_settings=K_FORWARDING,
)
def train(ctx: typer.Context) -> None:
    """Train a model described by a JSON configuration.

    Usage: sa train <train.json> [--profile <profile.json>]. The configuration
    names the architecture file, the objective, the data, the optimizer, the
    epoch count and where to write the checkpoint; paths inside it resolve
    against its own directory. --profile records the run with SushiProfiler
    as the profile configuration says; `sa profile` reads the record.
    Arguments are forwarded verbatim to the program, --help included once it
    is built.
    """
    _show_page_while_unbuilt(ctx, K_TRAIN_TARGET)
    raise typer.Exit(project_svc.run_app(K_TRAIN_TARGET, "SushiAI Train", list(ctx.args)))


@app.command(
    "inference",
    rich_help_panel=K_MODELS,
    epilog="sa inference applications/examples/heavy_mlp/model.json model.sushi "
           "applications/examples/heavy_mlp/eval.json\n"
           "sa inference model.json model.sushi eval.json --profile profile.json",
    context_settings=K_FORWARDING,
)
def inference(ctx: typer.Context) -> None:
    """Evaluate a trained checkpoint over a dataset.

    Usage: sa inference <model.json> <model.sushi> <eval.json> [--profile
    <profile.json>]. The data is a positional argument because the
    architecture file carries no data block. --profile records the run with
    SushiProfiler. Arguments are forwarded verbatim to the program, --help
    included once it is built.
    """
    _show_page_while_unbuilt(ctx, K_INFERENCE_TARGET)
    raise typer.Exit(
        project_svc.run_app(K_INFERENCE_TARGET, "SushiAI Inference", list(ctx.args)))


@app.command(
    "import",
    rich_help_panel=K_MODELS,
    epilog="sa import darknet yolov7-tiny.cfg yolov7-tiny.weights --output models/yolov7_tiny",
    context_settings=K_FORWARDING,
)
def import_model(ctx: typer.Context) -> None:
    """Write a model of another framework as a model, a weight and a training file.

    Usage: sa import <format> <files...> --output <directory> [--force]. The
    format names an importer; darknet takes a .cfg and optionally a .weights.
    Arguments are forwarded verbatim to the program, --help included once it
    is built.
    """
    _show_page_while_unbuilt(ctx, K_IMPORT_TARGET)
    raise typer.Exit(project_svc.run_app(K_IMPORT_TARGET, "SushiAI Import", list(ctx.args)))


@app.command(
    "tokenize",
    rich_help_panel=K_MODELS,
    epilog="sa tokenize train corpus.txt --vocabulary-size 4096 --output vocab.savb\n"
           "sa tokenize encode corpus.txt --vocabulary vocab.savb --output corpus.satk",
    context_settings=K_FORWARDING,
)
def tokenize(ctx: typer.Context) -> None:
    """Learn a byte-pair vocabulary from a corpus, or encode a corpus with one.

    Usage: sa tokenize train <corpus> --vocabulary-size <N> --output <vocab.savb>,
    or sa tokenize encode <corpus> --vocabulary <vocab.savb> --output <corpus.satk>.
    A file that is there already stops the run unless --force is given.
    Arguments are forwarded verbatim to the program, --help included once it
    is built.
    """
    _show_page_while_unbuilt(ctx, K_TOKENIZE_TARGET)
    raise typer.Exit(
        project_svc.run_app(K_TOKENIZE_TARGET, "SushiAI Tokenize", list(ctx.args)))


@app.command(
    "profile",
    rich_help_panel=K_MODELS,
    epilog="sa profile record show build/profiles/run.sprec.json\n"
           "sa profile record compare before.sprec.json after.sprec.json",
    context_settings=K_FORWARDING,
)
def profile(ctx: typer.Context) -> None:
    """Read a profiling record with SushiProfiler's `sp`.

    Usage: sa profile <args>, which runs `sp <args>` from the project root.
    `sa train --profile` and `sa inference --profile` write the records. When
    `sp` is not on the path the command says where to install it. Arguments
    are forwarded verbatim, --help included.
    """
    if K_HELP_FLAG in ctx.args and not profiler_svc.is_installed():
        typer.echo(ctx.get_help())
    raise typer.Exit(profiler_svc.run_profiler(list(ctx.args)))


register_diagnostic_commands(app, DIAGNOSTICS, program=PROFILE.program, panel=K_DIAGNOSTICS)

register_provision_commands(app, ModuleProvision(
    profile=PROFILE,
    project_root=find_project_root,
    load_config=load_config,
    console=lambda: console,
    locate=lambda name: load_config().locate_sibling(find_project_root(), name),
    panel=K_DIAGNOSTICS,
    uses_base=True,
))


def main() -> None:
    """Runs `sa`; a failure the user can act on leaves one line and exit code 1."""
    run_entry(app, _print_error)


if __name__ == "__main__":
    main()
