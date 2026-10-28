# Install SushiAI

SushiAI installs in two ways. Both give the same checkout and the same `sa` command, and neither
needs the other: SushiHub is not required to build SushiAI, and a checkout made without it joins
a SushiHub workspace later without any change.

## Standalone

You need:

- Git, Python 3.11 or later and CMake 3.20 or later (3.22 for the test suite).
- A SYCL 2020 compiler. `sa setup` installs the intel/llvm bundle SushiAI builds with.
- Network access, for the toolchain and the C++ dependencies.

SushiAI builds on SushiRuntime and SushiBLAS. Clone them beside this repository; `sa build` builds
a package of each, and of SushiLog and SushiProfiler, at the tags `cli/sushistack.deps.toml` pins.

```bash
git clone https://github.com/SushiSystems/SushiRuntime.git
git clone https://github.com/SushiSystems/SushiBLAS.git
git clone https://github.com/SushiSystems/SushiAI.git
cd SushiAI
pip install -e ./cli
sa setup
sa build
sa test
```

`pip install` also installs `sushicore`, the command-line layer `sa` shares with the other Sushi
modules, from PyPI at the version `cli/pyproject.toml` pins. `sa setup` writes the paths of this
machine to `cli/config.local.toml`, which git ignores; `sa config` shows what it resolved.

## With SushiHub

```bash
hub add sushiai            # clone it into the workspace
hub install-cli sushiai    # put sa on the path
hub link sushiai <path>    # adopt a checkout you already have, instead of hub add
```

`hub link` records where the checkout is and changes nothing in it. `hub install` provisions the
toolchain and the libraries of every module in the workspace once, and `sa` reads the result.
After either route, `sa build` and `sa test` are the commands above.
