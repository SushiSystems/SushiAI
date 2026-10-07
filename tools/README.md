# tools

The checkers every Sushi Systems repository carries, copied whole from `tools/` in SushiSkills.
A rule is fixed there first and copied here; `K_` settings are the only local changes.

| Path | Checks | Takes |
| --- | --- | --- |
| `common/checker.py` | Nothing itself; runs a checker's rule table and prints findings | |
| `documentation/check_source_comments.py` | C++, GLSL, TypeScript and Python: license block, file header, block ceiling, `//` and `#` runs, separators, history words, `@date` | Files or folders |
| `documentation/check_docs_layout.py` | `docs/` entries, required documents, document names, work folders, status lines, design ceiling, links under `docs/`, reachability from `docs/README.md`, READMEs under `modules/<tier>/`, archive candidates | Repository root |
| `documentation/check_changelog.py` | Headings and entry shape, releases kept live, length, one sentence, nesting, cited places | Repository root |
| `layering/check_layering.py` | Declared tiers, upward includes, reaches into another module's `source/` | Repository root |
| `licensing/write_license_block.py` | Nothing; writes the license block of `source-comments` into every tracked source file, or lists the files that lack it with `--report` | Repository root, `--project` |
| `contract/check_contract.py` | `SA_THROW_IF` and `throw` sites of `include/`, `src/`, `apps/` and `examples/` against `contract/site_classes.tsv`: table shape, no file with more sites than rows, every site a row, every row a site; prints the open questions of the table and the `Detail::bridge(` count per module as notes | Repository root |
| `naming/build_rename_map.py` | Nothing; lists the constants, enumerators, interfaces, enums and macros the naming ruling changes, writes `naming/rename_map.tsv` and prints the tokens the rename could not resolve | Repository root, `--output` |
| `naming/apply_rename.py` | Nothing; applies `naming/rename_map.tsv` to the C++ tree as identifier-token replacements, one kind at a time with `--kind`, or prints the per-file counts with `--dry-run`; refuses a tree with uncommitted work under the code roots | Repository root, `--map`, `--kind`, `--dry-run` |
| `tests/` | Unit tests for the checker rules; run `python -m unittest discover -s tools/tests` | |

`check_docs_layout.py` reads inline Markdown links, with or without a title, wrapped over a
line or in angle brackets. It does not read reference-style links or a link around an image.

Three tools here write files. `naming/build_rename_map.py` writes the rename map and
`naming/apply_rename.py` rewrites the C++ tree from it; [`naming/README.md`](naming/README.md)
says what each reads and when it runs, and [`contract/README.md`](contract/README.md) does the
same for the contract checker and its table. `write_license_block.py` takes its licence lines
from `check_source_comments.py`, covers `.in` templates and `.inc` fragments by the file they become, `--closed` for a closed repository, `--skip GLOB` for generated
files and `--upstream PATH=LINE;LINE` for a ported file; a later run without it keeps those lines. A `--report` run that prints nothing
proves a repository's headers are in place.

Every checker takes `--report` and `--rule NAME`, exits 1 on findings, 0 when clean or
reporting, 2 on a bad path. Python 3.11, standard library only.

## Per-repository settings

`K_` constants at the top of each checker. `check_layering.py` does nothing until
`K_TIER_ORDER` lists the repository's tiers, lowest first. `K_COPYRIGHT_LINE` and
`K_LICENSE_LINES` in `check_source_comments.py` hold the repository's licence lines; a closed
repository sets `K_LICENSE_LINES` to `("All rights reserved. No licence is granted.",)`.
