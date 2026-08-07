"""SushiAI developer CLI (`sa`).

Thin Typer layer: commands only parse arguments and delegate to the service
layer in ``sushiai.services``.
"""

from __future__ import annotations

from typing import Optional

import typer

from .services import diag as diag_svc
from .services import project as project_svc
from .services.project import BuildType, Suite

app = typer.Typer(
    name="sa",
    help="SushiAI developer CLI — build, test, and run SushiAI.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)


# --------------------------------------------------------------------------- #
# build / test / run / clean / doxygen
# --------------------------------------------------------------------------- #
@app.command("build")
def build(
    type: BuildType = typer.Option(
        BuildType.release, "--type", "-t", case_sensitive=False,
        help="Build type: release | debug | relwithdebinfo."),
    clean: bool = typer.Option(
        False, "--clean", help="Delete the build tree before configuring."),
    define: Optional[list[str]] = typer.Option(
        None, "--define", "-D", metavar="VAR=VALUE",
        help="Pass a CMake cache variable through to configure. Repeatable. "
             "Quote values containing ';', e.g. "
             '-D SB_SYCL_TARGETS="spir64;nvidia_gpu_sm_86".'),
):
    """Configure and build the project.

    Cache variables set with -D persist in the build tree, so dropping the
    flag on a later run does not unset them; pass VAR= to clear one, or
    --clean to start over.
    """
    raise typer.Exit(project_svc.build(type, clean, define))


@app.command("test")
def test(
    suite: Suite = typer.Option(
        Suite.functional, "--suite", "-s", case_sensitive=False,
        help="Which CTest label group to run ('all' runs every test)."),
    filter: Optional[str] = typer.Option(
        None, "--filter", "-f", help="ctest -R regex over 'Suite.Case' test names."),
    repeat: int = typer.Option(
        0, "--repeat", "-r", min=0,
        help="Re-run each test up to N times, stopping on the first failure "
             "(ctest --repeat until-fail). Handy for hunting flaky tests."),
):
    """Run the test suite via CTest labels.

    For GTest-level knobs (per-case filtering, shuffle, break-on-failure) run
    the binary directly: sa run sushiai_tests -- --gtest_filter=Version.*
    """
    raise typer.Exit(project_svc.test(suite, filter, repeat))


@app.command(
    "run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run(
    ctx: typer.Context,
    target: Optional[str] = typer.Argument(
        None, help="Executable name to run (exact, then substring match)."),
    sort: bool = typer.Option(
        False, "--sort", help="Interactively pick an executable."),
):
    """Run a built executable. Args after `--` are forwarded to it.

    Example: sa run sushiai_example
    """
    raise typer.Exit(project_svc.run(target=target, sort=sort, app_args=list(ctx.args)))


@app.command("clean")
def clean():
    """Remove the build/ tree."""
    raise typer.Exit(project_svc.clean())


@app.command("doxygen")
def doxygen():
    """Generate Doxygen documentation."""
    raise typer.Exit(project_svc.doxygen())


# --------------------------------------------------------------------------- #
# demo (AI-7 showcase, per ST-2)
# --------------------------------------------------------------------------- #
demo_app = typer.Typer(help="Run a SushiAI showcase demo.", no_args_is_help=True)
app.add_typer(demo_app, name="demo")


@demo_app.command(
    "mlp",
    # Unknown options are collected instead of rejected, and Typer's own --help
    # is disabled, so every flag - including --help - reaches the demo. The
    # demo owns its flag surface; restating it here would give two places to
    # change and one to forget.
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)
def demo_mlp(ctx: typer.Context):
    """Train an MLP end to end and report the compile count (AI-7 showcase).

    Flags are forwarded verbatim to the demo; `sa demo mlp --help` prints its
    full list. Example: sa demo mlp --epochs 4 --optimizer sgd
    """
    raise typer.Exit(project_svc.demo("mlp", list(ctx.args)))


# --------------------------------------------------------------------------- #
# train / inference (config-driven, per the config-driven training design)
# --------------------------------------------------------------------------- #
# Arguments are forwarded verbatim, exactly as `demo mlp` forwards its flags:
# the argument surface lives in the C++ program, and restating it here would
# give two places to change and one to forget. That matters more here than for
# the demo, because the whole point of these two programs is that they are
# written once and never edited again.
@app.command(
    "train",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)
def train(ctx: typer.Context):
    """Train a model described by a JSON configuration.

    Usage: sushiai train <train.json>. The configuration names the
    architecture file, the objective, the data, the optimizer, the epoch count
    and where to write the checkpoint; paths inside it resolve against its own
    directory. Example: sushiai train examples/heavy_mlp/train.json
    """
    raise typer.Exit(
        project_svc.run_app("sushiai_train", "SushiAI Train", list(ctx.args)))


@app.command(
    "inference",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)
def inference(ctx: typer.Context):
    """Evaluate a trained checkpoint over a dataset.

    Usage: sushiai inference <model.json> <model.ckpt> <eval.json>. The data
    is a positional argument because the architecture file carries no data
    block, so an optional flag would always error when omitted.
    """
    raise typer.Exit(
        project_svc.run_app("sushiai_inference", "SushiAI Inference", list(ctx.args)))


# --------------------------------------------------------------------------- #
# config / env (diagnostics)
# --------------------------------------------------------------------------- #
@app.command("config")
def config():
    """Print the resolved config and where each value came from."""
    raise typer.Exit(diag_svc.config_show())


@app.command("env")
def env(
    all: bool = typer.Option(
        False, "--all", help="Show every variable, not just build-relevant ones."),
):
    """Print the environment cmake/ctest/run subprocesses execute under."""
    raise typer.Exit(diag_svc.env_dump(show_all=all))


if __name__ == "__main__":
    app()
