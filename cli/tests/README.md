# cli/tests

The pytest suite of the `sa` CLI. Run it from `cli/` with `python -m pytest -q`. The sections
below hold the reasoning that the test files cite.

## conftest.py

`sushiai` imports `sushicore`, the shared core underneath every Sushi CLI, at module load,
so the suite cannot be collected without it.

These tests once ran against a hand-written stub of that package. The stub was a liability
in two ways. It had to grow every time sushicore did, and a test could end up pinning the
imitation and not the contract; one of them asserted on an attribute only the fake had.
sushicore ships inside the SushiStack repository, so the real package is reachable, either
installed in the environment or in a checkout next door, and the tests use it.

## test_console_lazy.py

`console.py` once built the console at import time. That called `config_dir()`, which
called `find_project_root()`, which raised `SystemExit` outside a checkout. `cli.py` imports
the console transitively, so every `sa` invocation died with "Not inside a SushiAI project",
including `sa --help`, which Typer can answer with no project.

Four of the five CLIs fixed this independently and sushiengine's copy never did, so
`se --help` outside a checkout failed for as long as the copies existed. The lifecycle lives
in `sushicore.cli_console.LazyConsole`. These tests pin that this CLI is wired to it, which
a shared implementation cannot check on its own.

In `test_output_helper_falls_back_outside_project`, no project means there are no config
files to layer, and the helper must still not crash.

## test_demo.py

This file is not pure logic. It shells out to the installed `sa` and to a real build tree,
because what needs guarding is the whole path from the CLI through the demo binary to the
printed contract. A demo nobody runs stops being true, and this one's output is quoted in
`docs/README.md`.

The tests skip, and do not fail, when `sa` is not installed or the tree has not been built,
since neither is true in a clean checkout. The C++ suite pins the same claim where it always
runs: the `Integration_Demo.MlpRunsAndReportsOneCompile` entry in `tests/CMakeLists.txt`.

`REPO_ROOT` is two parents above the file: `cli/tests/test_demo.py`, `cli/tests`, `cli`,
then the repository root.

The demo prints the compile count twice, once in the run report and once in the banner.
`_COMPILE_ONCE` accepts either match; what matters is that the number is 1.

`_DEMO_ARGS` is deliberately tiny: two steps on the synthetic fallback. `--mnist-dir` points
at a directory that cannot exist, so the fallback is the path under test whether or not the
developer has MNIST downloaded.

`sa demo mlp` forwards its arguments verbatim, so a flag the demo grew is reachable only if
nothing in the CLI swallows it. `_PRECISION_CASES` is the cheapest way to see that: fp32
must report no conversions and mixed must report some, from the same command line, through
the same forwarding. The compile-once contract holds in both modes, which is why mixed
precision lives in the graph and is not a per-step dtype switch.

`test_sa_demo_mlp_loss_scale_flag_reaches_the_demo_and_is_exact` is the same forwarding
check for `--loss-scale`. It asserts the one thing about the flag worth asserting through
the CLI: a dyadic scale is exact, so a scaled fp32 run must report the same loss as an
unscaled one to the last printed digit. A rounding anywhere in the scale and unscale round
trip would move that digit.

`--activation` picks between two model types, so the demo instantiates a different template
for each. That makes "the flag reached the demo" and "the right model was built" the same
question, and the fusion report answers it: ReLU folds as `gemm+bias+relu` and GELU as
`gemm+bias+gelu`. The second form exists only because its epilogue spills the pre-activation
its gradient reads.

## test_suites.py

`sa test` maps every suite to a `ctest -L` regex through `_SUITE_LABEL_REGEX`, with two
deliberate exceptions: `all` selects no label and `package` runs no ctest. Both are absent
from the mapping, so both depend on `test()` branching before it indexes the mapping. A
missing branch is a `KeyError` at the moment the user runs the command, and these tests
exist to prevent it.

The file mirrors `sushiblas/cli/tests/test_suites.py`, for the same reason
`sa test --suite package` mirrors `sb test --suite package`.

In `test_toolchain_args_exclude_the_prefix_path_and_the_source_siblings` the configuration
is a stand-in for a fully populated Windows config. Every optional host tool is set, so the
assertion sees the largest flag list the helper can emit.

## test_train.py

Like `test_demo.py`, these tests shell out to the installed CLI and a real build tree. What
needs guarding is the whole path from the CLI through the app binary to the printed
contract, and no in-process test can see it.

The argument surface is deliberately not checked. Both commands forward verbatim, so
restating their arguments in Typer would give two places to change and one to forget. The
diagnostics come from the C++ side, and this file asserts that they reach the caller intact.

The tests skip, and do not fail, when the CLI is not installed or the tree has not been
built. CTest pins the same claims where they always run: the two `Integration_App.*`
entries in `tests/CMakeLists.txt`.

`_TRAIN_CONFIG` is generated by `tests/CMakeLists.txt` into the build tree. A `train.json`'s
relative paths resolve against its own directory, and a committed one would write its
checkpoint into the source tree.

`_run` states the encoding and does not leave it to the locale. The CLI's help screen is
rendered by Rich and carries box-drawing characters, which a cp1252 default cannot decode.
The failure is not an exception the caller sees; `stdout` silently comes back as `None`.

`test_sushiai_inference_loads_the_checkpoint_train_wrote` runs the training command itself
before it runs inference. The training test is what writes the checkpoint, and running it
again removes the dependence on pytest's collection order.
