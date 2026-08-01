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
):
    """Configure and build the project."""
    raise typer.Exit(project_svc.build(type, clean))


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


@demo_app.command("mlp")
def demo_mlp():
    """Run the MLP training showcase (AI-7).

    Not implemented yet: AI-7 (the MLP training demo) hasn't landed - only the
    AI-0 scope-reset skeleton has. See docs/slop/make_sushiai_great_again.md.
    """
    from . import console
    console.error("`sa demo mlp` is not implemented yet — see AI-7 in the master plan.")
    raise typer.Exit(1)


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
