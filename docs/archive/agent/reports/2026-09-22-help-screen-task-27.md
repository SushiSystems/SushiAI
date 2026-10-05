# Task 27: the help screen and one set of colours in `sa`

Branch `feature/sushicore_help_screen`, off `main`. Nothing pushed, merged or tagged. The checkout is left on that branch.

## Files changed

- `cli/sushiai/console.py`: added `current()`, which returns `_lazy.get()`. This module never rebinds `_lazy`, so nothing else was needed.
- `cli/sushiai/cli.py`: `_help_group = help_group(console.current)`; `cls=_help_group` on the root `app` and on `demo_app` (the only two Typers); `rich_help_panel` on every top-level command and on `add_typer`; `epilog` on the seven commands whose `--help` draws a page and on `demo_app`. The `Example:` line in `run`'s docstring is gone, because the epilog now shows it.
- `cli/sushiai/services/project.py`: three `[dim]{exe}[/dim]` became `[muted]{exe}[/muted]` (lines 474, 510, 543).
- `cli/pyproject.toml`: `sushicore>=0.1.0` to `sushicore>=0.4.0`.
- `README.md`: one paragraph after the "Other CLI commands" block. `sa` has no `cli/README.md`; its command list is in the root README and in `docs/guides/CLI_GUIDE.md` (its overview table stays true and was not touched).
- `docs/reference/CHANGELOG.md`: two lines under Unreleased, in the file's `date — area: Verb ... (`file`).` format.
- `cli/tests/test_help_screen.py` (new, 26 test cases) and one test added to `cli/tests/test_console_lazy.py` for `current()`.

## Panel names

- Project: `build`, `test`, `run`, `clean`, `doxygen`. CLI_GUIDE's overview opens with "Build, test, run, and clean the C++ project"; `doxygen` sits next to it there and has no group of its own worth a heading.
- Models: `train`, `inference`, `demo`. The overview lists `demo mlp`, `train` and `inference` together, and all three produce or evaluate a model.
- Diagnostics: `config`, `env`. `cli.py` already labels that section "diagnostics", as does the guide's `sa config` and `sa env` section.

## Examples dropped, and why

- `train`, `inference`: no epilog. Both set `help_option_names: []`, so `--help` is forwarded to the C++ program and Typer never draws their page; an epilog there could not be read. A test pins that the two commands keep an empty help-option list.
- `demo mlp`: same reason, so its examples went on the `demo` sub-app's epilog instead. Its options `--epochs` and `--optimizer sgd` are checked against `examples/demo_mlp.cpp` (lines 357 and 399-401), because `cli.py` does not declare them.
- Every example in the other seven epilogs was checked against `cli.py`: `--type`, `--clean`, `-D`, `--suite unit`, `--filter`, `--repeat`, `--sort`, `sa run <target> -- <args>`, `--all`. None was dropped for naming an option that does not exist.

## Commands and output

Baseline, before any change (from `D:/Projects/sushiai/cli`):

```
$ PYTHONPATH=D:/Projects/sushicore python -m pytest tests -q -p no:cacheprovider
FAILED tests/test_demo.py::test_sa_demo_mlp_runs_and_reports_one_compile - As...
FAILED tests/test_demo.py::test_sa_demo_mlp_precision_flag_reaches_the_demo[fp32]
FAILED tests/test_demo.py::test_sa_demo_mlp_precision_flag_reaches_the_demo[mixed]
FAILED tests/test_demo.py::test_sa_demo_mlp_loss_scale_flag_reaches_the_demo_and_is_exact
FAILED tests/test_demo.py::test_sa_demo_mlp_rejects_a_non_dyadic_loss_scale
FAILED tests/test_demo.py::test_sa_demo_mlp_activation_flag_reaches_the_demo[relu]
FAILED tests/test_demo.py::test_sa_demo_mlp_activation_flag_reaches_the_demo[gelu]
FAILED tests/test_train.py::test_sushiai_train_runs_and_reports_one_compile
FAILED tests/test_train.py::test_sushiai_inference_loads_the_checkpoint_train_wrote
9 failed, 21 passed in 10.92s
```

The nine failures are in tests that shell out to the built demo and train binaries; their message is `No kernel named _ZTS... was found` (a SYCL kernel lookup in the built tree), before and after.

The new tests, first run before the implementation (16 failed, 17 passed): the failures were the missing `current`, Typer's own screen instead of the grouped page, and the `[dim]` markup in `project.py`. That last failure came from an f-string, so the source check does look inside f-strings (it walks `ast.Constant` nodes).

After:

```
$ PYTHONPATH=D:/Projects/sushicore python -m pytest tests -q -p no:cacheprovider
9 failed, 48 passed in 89.06s (0:01:29)
```

The same nine tests fail as in the baseline; 27 passed tests were added (21 to 48).

```
$ python -m py_compile sushiai/cli.py sushiai/console.py sushiai/services/project.py tests/test_help_screen.py tests/test_console_lazy.py; echo "exit $?"
exit 0
```

Rendered pages through `CliRunner` in-process on a fake 90-column truecolour console (script `render_help.py` in the session scratchpad; colour codes stripped):

```
=== sa --help (exit 0 , ansi bytes: 6918 )

     ▄▄▀▀▀▀▀▀▀▀▄      ▄█▀▀▀▀ ██  ██ ▄█▀▀▀▀ ██  ██ ██                    
   ▄▀▀▀▀▀▀▀▀▀██▀▀▄    ▀█▄▄▄  ██  ██ ▀█▄▄▄  ██▄▄██ ██                    
  ▄▀▀██▀▀▀▀▀██▀████       ██ ██  ██     ██ ██  ██ ██                    
  ███████████▀██████  ▀▀▀▀▀   ▀▀▀▀  ▀▀▀▀▀  ▀▀  ▀▀ ▀▀                    
  ██████████████████   ▄▄▄▄▄ ▄▄  ▄▄  ▄▄▄▄▄ ▄▄▄▄▄▄ ▄▄▄▄▄▄ ▄▄    ▄▄  ▄▄▄▄▄
  █████▀▀██▀▀████▀▀▀  ██     ▀█▄▄█▀ ██       ██   ██     ███▄▄███ ██    
   ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀     ▀▀▀█▄   ██    ▀▀▀█▄   ██   ██▀▀   ██ ██ ██  ▀▀▀█▄
     ▀▀▀▀▀▀▀▀▀▀▀      ▄▄▄▄█▀   ██   ▄▄▄▄█▀   ██   ██▄▄▄▄ ██    ██ ▄▄▄▄█▀


sa
SushiAI developer CLI — build, test, and run SushiAI.

Usage: sa [OPTIONS] COMMAND [ARGS]...

Project
  build    Configure and build the project.    
  test     Run the test suite via CTest labels.
  run      Run a built executable.             
  clean    Remove the build/ tree.             
  doxygen  Generate Doxygen documentation.     

Models
  train      Train a model described by a JSON configuration.
  inference  Evaluate a trained checkpoint over a dataset.   
  demo       Run a SushiAI showcase demo.                    

Diagnostics
  config  Print the resolved config and where each value came from.        
  env     Print the environment cmake/ctest/run subprocesses execute under.

Options
  --help  Show this message and exit.

=== sa build --help (exit 0 , ansi bytes: 158 )
sa build
Configure and build the project.

Cache variables set with -D persist in the build tree, so dropping the flag on a later run
does not unset them; pass VAR= to clear one, or --clean to start over.

Usage: sa build [OPTIONS]

Options
  -t, --type [release|debug|relwithdebinfo]  Build type: release | debug | relwithdebinfo.
                                             [default: release]                           
  --clean                                    Delete the build tree before configuring.    
  -D, --define VAR=VALUE                     Pass a CMake cache variable through to       
                                             configure. Repeatable. Quote values          
                                             containing ';', e.g. -D                      
                                             SB_SYCL_TARGETS="spir64;nvidia_gpu_sm_86".   
  --help                                     Show this message and exit.                  

Examples
  sa build                        Build the release configuration    
  sa build --type debug                                              
  sa build --clean                Wipe the build tree first          
  sa build -D SA_BUILD_TESTS=OFF  Pass a CMake cache variable through

=== sa demo --help (exit 0 , ansi bytes: 145 )
sa demo
Run a SushiAI showcase demo.

Usage: sa demo [OPTIONS] COMMAND [ARGS]...

Commands
  mlp  Train an MLP end to end and report the compile count (AI-7 showcase).

Options
  --help  Show this message and exit.

Examples
  sa demo mlp                             Train with the defaults       
  sa demo mlp --epochs 4 --optimizer sgd  Flags go to the demo unchanged
```

(`sa run --help` and `sa env --help` render the same way; both are in the script's output and show their Examples.)

Other checks: `tools/documentation/check_changelog.py` printed nothing on the changelog. `check_source_comments.py` reports only pre-existing C++ findings under `tests/`.

## Findings and what was not done

- The demo sub-app is a group, not a collapsed single command: `sa demo --help` draws a page. No app in `sa` is affected by the one-command-no-callback case.
- Bare `sa` exits with code 2 and draws the help page, because `no_args_is_help=True` raises Click 8.2's `NoArgsIsHelpError` (click 8.2.1 is installed). This is how the root behaved before this change; I left it. The test compares the bare page with `--help` after stripping whitespace and does not assert the exit code.
- No raw Rich colour name is left under `cli/sushiai/`, and the mapping table had no gap: the only markup found was the three `[dim]`. A test greps every file in the package for the six names.
- The plan's "sushicore 0.4.0 checkout on `PYTHONPATH`" matters here: the installed `sushicore` in this Python is 0.3.0, and the tests only ran against 0.4.0 through `PYTHONPATH`.
- Not done: `sa demo mlp --help`, `sa train --help` and `sa inference --help` still forward to the C++ programs and were not rendered through Typer, as designed. No `sa build`/`sa test` was run. The nine environment failures were not investigated.
