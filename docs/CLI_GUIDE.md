# The SushiAI CLI

SushiAI ships with a small command-line tool that drives everything you do
day to day: building the C++ library, running its tests, launching the
example, and generating documentation. It is a thin wrapper around CMake and
CTest that reads your machine-specific toolchain paths from a config file so
you don't have to retype long compiler flags. It's modeled directly on
[SushiEngine](https://github.com/SushiSystems/SushiEngine)'s `se` CLI, not
SushiBLAS's `sb` (which still models an older, pre-toolchain-refresh build).

## Installing the CLI

The CLI is a Python package that lives in the `cli/` folder. Install it once
and it puts two commands on your `PATH`:

- **`sa`** — short name.
- **`sushiai`** — long name.

They are identical; every example below uses `sa`.

```bash
ss install-cli sushiai      # add --no-editable for a non-dev install
```

On Windows this installs through **pipx**; on Linux and macOS with **pip**
into your current environment.

```bash
pipx uninstall sushiai-cli   # to remove it later
```

### How the CLI finds your compiler and siblings

SushiAI builds against the shared SushiStack SYCL toolchain — it selects no
toolchain of its own — and against **two sibling checkouts**, SushiRuntime
and SushiBLAS. The CLI reads all of this from:

- **`cli/config.toml`** — committed, shared defaults.
- **`cli/config.local.toml`** — your machine's absolute paths (gitignored).

In the common case — sibling `../sushiruntime` and `../sushiblas` checkouts
with the bundled toolchain already provisioned by `ss install`, or a
SushiStack workspace — no local config is needed at all. Run `sa config` to
see exactly what the CLI resolved.

## Command overview

| Command | What it's for |
|---------|----------------|
| `sa build` / `test` / `run` / `clean` / `doxygen` | Build, test, run, and document the C++ project |
| `sa demo` | Run a SushiAI showcase demo |
| `sa config` | Show the resolved configuration |
| `sa env` | Show the environment your builds run under |

Run any command with `--help` to see its options.

## `sa build`

```bash
sa build                  # Release build (the default)
sa build --type debug     # Debug build
sa build --type relwithdebinfo
sa build --clean          # delete the build tree first, then build from scratch
```

The `--type` (`-t`) option accepts `release`, `debug`, or `relwithdebinfo`.

If either sibling can't be found, `sa build` fails fast with a clear message
naming which one and where it looked, rather than letting CMake's own
`add_subdirectory` error surface. `sa config` shows what the CLI resolved for
both `SushiRuntime` and `SushiBLAS`; set `sushiruntime_dir` / `sushiblas_dir`
in `cli/config.local.toml` (or `SUSHIRUNTIME_DIR` / `SUSHIBLAS_DIR` in the
environment) to point either elsewhere.

## `sa test`

```bash
sa test                          # functional (unit+integration) — the default
sa test --suite unit
sa test --suite integration
sa test --suite all              # every test
```

```bash
sa test --filter 'Version.*'  # ctest -R regex over 'Suite.Case' test names
sa test --repeat 5            # re-run each test up to 5x, stop on first failure
```

There is no `regression` suite yet — nothing has shipped that a regression
could guard against (see [CONTRIBUTING.md §3](CONTRIBUTING.md#3-building-and-testing-your-change)).

For GoogleTest-level options CTest doesn't expose, run the binary directly
with `sa run` and pass flags after `--`.

## `sa run`

Runs a built executable. With no target it runs the configured default
(`sushiai_example`).

```bash
sa run                                # run the default target
sa run sushiai_example
sa run --sort                         # interactively pick from the list of executables

sa run sushiai_tests -- --gtest_filter='Unit_Version.*'
```

## `sa demo` — the AI-7 showcase (not implemented yet)

```bash
sa demo mlp
```

This command exists in the CLI's surface today so its final shape doesn't
need inventing later, but it does not run anything yet: it prints "`sa demo
mlp` is not implemented yet — see AI-7 in the master plan" and exits with a
non-zero status. See
[`docs/slop/make_sushiai_great_again.md`](slop/make_sushiai_great_again.md)
for where AI-7 sits in the roadmap.

## `sa clean` and `sa doxygen`

```bash
sa clean       # remove the build/ tree
sa doxygen     # generate the Doxygen API reference
```

`sa doxygen` runs Doxygen against the repo-root `Doxyfile`. Doxygen is an
external tool, not vendored — install it or point `doxygen_exe` in
`cli/config.local.toml` at an existing binary; the command prints the right
install line if it cannot find one.

## `sa config` and `sa env` — diagnostics

```bash
sa config         # print the resolved config and where each value came from
sa env            # print the environment cmake/ctest/run subprocesses run under
sa env --all      # show every variable, not just the build-relevant ones
```

`sa config` prints both resolved sibling directories, the compiler, and the
vcpkg root, plus which source (default / `config.toml` / `config.local.toml`
/ an `SA_*` environment variable) supplied each field.

## Building without the CLI

The CLI is a convenience, not a requirement. See **Building with CMake
directly** in the top-level [README.md](../README.md#building-with-cmake-directly).
