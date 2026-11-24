# tools

This repository's side of the shared checkers, which run from the sushicore package with
`python -m sushicore.tools check .`:

| Path | Holds |
| --- | --- |
| `settings.py` | The settings the checkers read for this repository |
| `known_findings.toml` | The findings the checkers accept for now, each with its backlog ID |
| `hooks/pre-commit`, `hooks/commit-msg` | The git hooks; `python -m sushicore.tools install-hooks .` points git at them |

The repository's own tools:

| Path | Does |
| --- | --- |
| `layering/check_device_sources.py` | `DEVICE_SOURCES` of each `sushiai_add_module` call against the units that reach a kernel-launching header, directly or through SushiAI headers: a reaching unit not listed, a listed unit that reaches none, a test or application source that reaches one | Repository root |
| `cli_source/check_colour_names.py` | The string literals of `cli/sushiai/` against the raw Rich colour tags (`[green]`, `[/red]`, `[dim]`): a message must take its colour from the console theme | Repository root |
| `contract/check_contract.py` | `throw` sites of `modules/*/*/include`, `modules/*/*/source` and `applications/` against `contract/site_classes.tsv`: table shape, every site a row, every row a site; any use of `SA_THROW_IF`, `SushiAI::Error` or another removed name under those folders and `tests/` is a finding; prints the open questions of the table as notes | Repository root |
| `naming/build_rename_map.py` | Nothing; lists the constants, enumerators, interfaces, enums and macros the naming ruling changes, writes `naming/rename_map.tsv` and prints the tokens the rename could not resolve | Repository root, `--output` |
| `naming/apply_rename.py` | Nothing; applies `naming/rename_map.tsv` to the C++ tree as identifier-token replacements, one kind at a time with `--kind`, or prints the per-file counts with `--dry-run`; refuses a tree with uncommitted work under the code roots | Repository root, `--map`, `--kind`, `--dry-run` |
