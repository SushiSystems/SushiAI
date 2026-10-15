# tools

The checkers every Sushi Systems repository carries, copied whole from `tools/` in SushiSkills.
A rule is fixed there first and copied here with `checking/copy_shared_tools.py`, which keeps
`settings.py`, the license block of each file and the tools below that SushiSkills does not hold.

| Path | Checks | Takes |
| --- | --- | --- |
| `check_all.py` | Nothing itself; runs the six checkers and the tests, prints only what failed | Repository root |
| `checking/checker.py` | Nothing itself; runs a checker's rule table and prints findings | |
| `checking/copy_shared_tools.py` | Nothing; copies the shared tools of SushiSkills into this repository, or lists the files that differ with `--report` | Repository root, `--from` |
| `checking/ignored_paths.py` | Nothing itself; names the files and folders git ignores, which no checker reads | |
| `common/checker.py`, `common/ignored_paths.py` | The same as the two files above, for `layering/check_layering.py` alone | |
| `settings.py` | Nothing; holds the settings of this repository that the shared checkers read | |
| `documentation/check_source_comments.py` | C++, GLSL, TypeScript and Python: license block, file header, block ceiling, `//` and `#` runs, separators, history words, `@date` | Files or folders |
| `documentation/check_docs_layout.py` | `docs/` entries, required documents, document names, work folders, status lines, design ceiling, links under `docs/`, reachability from `docs/README.md`, READMEs under `modules/<tier>/<module>/`, archive candidates | Repository root |
| `documentation/check_docs_truth.py` | The manual against the source: named symbols, cited paths, example sources, history words, page length, duplicated passages, em dashes, the allow-list | Repository root, `--source` |
| `documentation/check_changelog.py` | Headings and entry shape, releases kept live, length, one sentence, nesting, cited places | Repository root |
| `layering/check_layering.py` | Declared tiers, upward includes, reaches into another module's `source/` | Repository root |
| `testing/check_test_mirror.py` | `tests/` mirrors the source: one `test_<name>` file per header or source file, no test without a unit, no folder without a counterpart, no test that includes a header under `source/` | Repository root |
| `licensing/write_license_block.py` | Nothing; writes the license block of `source-comments` into every tracked source file, or lists the files that lack it with `--report` | Repository root, `--project` |
| `contract/check_contract.py` | `SA_THROW_IF` and `throw` sites of `modules/*/*/include`, `modules/*/*/source` and `applications/` against `contract/site_classes.tsv`: table shape, no file with more sites than rows, every site a row, every row a site; prints the open questions of the table and the `Detail::bridge(` count per module as notes | Repository root |
| `routing/summarize_dispatches.py` | Nothing; sums the dispatch lines of every `REPORT.md` per model and effort, for `model-routing` | Repository root |
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

`settings.py` holds them: `K_TIER_ORDER`, the tiers lowest first; `K_CLOSED`, whether the
repository is closed source; `K_MACRO_PREFIXES`, the prefixes `check_docs_truth.py` reads as
macro names.

`layering/check_layering.py` and `tests/test_check_layering.py` are shared files that differ
from SushiSkills: the copy there does not yet resolve the tier include form, the umbrellas at
the include root or one module name in two tiers. They keep their own settings at the top of the
checker and import `common/`, until SushiSkills takes the same shape and the two are copied again.
