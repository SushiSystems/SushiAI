# cli

The `sa` command line interface: the Python package `sushiai` and its tests.
[`docs/guides/CLI_GUIDE.md`](../docs/guides/CLI_GUIDE.md) is the user's guide to the
commands. This file holds the reasoning that the source files cite; the tests have their own
in [`tests/README.md`](tests/README.md).

## config.py

Configuration is layered, low to high: built-in defaults, `config.toml`,
`config.local.toml`, then `SA_*` environment variables.

SushiAI is a head of the stack, like SushiEngine. It consumes the shared SushiStack SYCL
toolchain and vcpkg tree and provisions neither. Unlike SushiEngine it depends on two
siblings directly, SushiRuntime and SushiBLAS, so each gets its own resolved directory.

Locating the project and layering the files is the job of
`sushicore.module_config.ModuleConfig`. Resolving the compiler and vcpkg out of the shared
dependency tree is the job of `sushicore.stack_config.StackConfig`. What is left in
`config.py` is SushiAI's own: its identity and the siblings it builds against.

`PROFILE` states once how SushiAI describes itself to the shared machinery. Config loading,
the console, executable discovery and diagnostics all read it, so they cannot disagree about
what this module is.

`SUSHIRUNTIME_DIR` and `SUSHIBLAS_DIR` carry no `SA_` prefix. They name the same checkouts
for every CLI that builds against them, so `sa` must read the variables those CLIs read.

`_ENV_OVERRIDES` maps each field to its environment variable and is derived from the
profile's prefix. It is a module attribute because `sa config` reports which values came
from the environment and needs the reverse map.

Each Sushi library the fragment pins, SushiLog among them, reaches CMake as `<NAME>_DIR`
through sushicore's `sushi_definitions`; `services/project.py` adds them to every configure.

An empty `sushiruntime_dir` means `../sushiruntime` relative to the project root, and an
empty `sushiblas_dir` means `../sushiblas`. Both match the defaults of `SUSHIRUNTIME_DIR`
and `SUSHIBLAS_DIR` in the root `CMakeLists.txt`.

## console.py

The module is a thin wrapper around `sushicore`. The theme, icon and renderer logic and the
`[cli]` config schema live there and are shared with every other Sushi CLI; sushicore's
README says how to change colours.

`sushicore.cli_console.LazyConsole` builds the console on first use. Constructing it needs
this repository's `cli/` directory, and locating that directory fails outside a checkout.
Building at import time would make every `sa` invocation abort with "Not inside a SushiAI
project", including `sa --help`, which Typer can serve with no project at all.

`__getattr__` is the PEP 562 module hook. Attributes resolve on first access, so importing
the module never touches the filesystem or the project layout.

## env.py

A parent process cannot `call vcvars64.bat` and inherit the result, so the shell runs as a
child and its environment is dumped. `sushicore.build_env` holds the mechanism and the
cache.

SushiAI selects no SYCL toolchain of its own. It builds with the intel/llvm clang++ in the
shared SushiStack dependency tree, so its environment work is what
`sushicore.build_env.StackBuildEnv` does: load MSVC on Windows and put the shared
toolchain's `bin` and `lib` on `PATH`. The sibling checkouts in the cache key come from
`PROFILE` and are not listed a second time.

## memory_limit.py

`sa test` and `sa run` put the CLI process into a Windows job object whose committed memory
is capped, before they start anything. A child inherits the job, so the ceiling covers a
test binary and whatever it starts. A program that reaches it has its next allocation
refused: SushiBLAS reports the size, the step returns `out_of_memory`, and the test fails
by name. Without the ceiling the same program fills the page file and the machine stops
answering, which is what a training step of the full YOLOv3 did on 2026-10-11 on a machine
of 15 GB.

The ceiling is `--memory-limit GB`, else `SA_MEMORY_LIMIT_GB`, else half of the physical
memory; 0 applies none. `resolve_limit_bytes` decides it from values it is handed and
`apply_limit` enforces it, so the first has tests that need no operating system.

On a platform other than Windows `apply_limit` applies nothing and returns False. An
address-space limit is the mechanism at hand there, and it counts what a device runtime
reserves and never commits, so it would refuse programs that fit. `sushiai train` and the
other program commands run without a ceiling: their memory is the user's to size.

## services/diag.py

`sa config` and `sa env` are read-only troubleshooting. When a build picks the wrong
compiler or a path looks off, `sa config` answers what the CLI resolved and where each value
came from, and `sa env` answers which environment cmake and ctest run under.

Both commands are `sushicore.diag.Diagnostics`. `diag.py` supplies the three things that are
SushiAI's: its profile, its config module and how it finds its build environment.

## services/discovery.py

The walk over the build tree is shared; see `sushicore.discovery.ExecutableIndex`. The
module contributes only the directories to prune, and it does not spell them out either.
They are the sibling checkouts `PROFILE` already names, because a module that builds a
dependency in-tree must not offer that dependency's executables as its own.

## services/profiler.py

`sa profile <args>` runs `sp <args>` from the project root and returns its exit code. `sp` is
SushiProfiler's own command line, which reads the records `sa train --profile` and
`sa inference --profile` write, so SushiAI repeats none of its commands. The arguments are
forwarded verbatim, `--help` included. When `sp` is not on the path the command prints one error
line that names SushiProfiler and where to get it, and returns 1.

## services/project.py

### Build policy

The module holds SushiAI's own build policy: which cache variables it passes and which demo
targets and test suites it has. It drives cmake and ctest through
`sushicore.cmake_driver.CMakeDriver` with the snapshotted build environment.

SushiAI configures against the SushiRuntime sibling's bundled clang++ and against vcpkg,
through SushiBLAS's own transitive resolution. The configure arguments resolve both from the
sibling checkouts or from the config overrides and provision nothing. This mirrors
SushiEngine's `se` CLI and not SushiBLAS's `sb`, which still models an older oneAPI build
from before SB-0 and SB-2.

### Sizes and CTest labels

`cmake/SushiAITestProgram.cmake` registers tests through `gtest_discover_tests` and gives each
case the CTest label of its size, `small`, `medium` or `large`, from the GTest suite-name
prefix, with the size's timeout. A case named in `SUSHIAI_QUARANTINED_TESTS` also carries
`quarantined`. The `Suite` enum lists those labels, and `sa test --label` takes one.

`_SUITE_LABEL_REGEX` maps each label to the regex `ctest -L` receives. `gating`, the default, is
an alternation over `small` and `medium`, and its run also passes `-LE quarantined`. `all`
selects everything and passes no `-L`. `package` is absent on purpose, because it runs no CTest
at all.

`package` is neither a CTest label nor part of `all`. It is an out-of-tree consumer of the
installed package, which has to be installed and configured on its own before it can run.

`test()` passes `--jobs` to ctest as the number of tests run at once, and leaves the count to
ctest when it is omitted. It refuses `--filter`, `--repeat` and `--jobs` for the package
suite. All three are ctest knobs and that suite runs no ctest, and refusing is better than accepting a flag and silently doing
nothing with it.

`all` means every CTest test, which is not every test the repository has. `test()` prints a
line that says so and names `sa test --label package`.

### Demo targets

`_DEMO_TARGETS` maps `sa demo <name>` to the executable that is that demo. A demo is a
separate binary from an example on purpose: it is a supported entry point with a stable flag
and output contract, while `applications/examples/source/example.cpp` is a teaching file people
are meant to edit. [`applications/examples/README.md`](../applications/examples/README.md) has
the full argument.

### The package suite starts from nothing

`test_package` removes the install prefix and the consumer build directory before it does
anything else. A stale prefix would let a header deleted since the last run keep satisfying
an include. A stale consumer directory would let a DLL staged by an earlier run, or by hand,
keep satisfying the loader, and that is how a deployment regression hides. This was found
the hard way: a hand-copied `hwloc-15.dll` from a manual run months earlier was still in the
directory, and the run step proved nothing.

### The consumer's configure

The consumer configure passes `--no-warn-unused-cli`. The consumer is CXX-only and never
invokes vcpkg's own installer, so a couple of the shared toolchain flags go unread there and
that is expected. Silencing that one warning keeps a clean run clean, and anything else
cmake has to say still comes through.
