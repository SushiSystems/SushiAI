# tools

The checkers every Sushi Systems repository carries, copied whole from `tools/` in SushiSkills.
A rule is fixed there first and copied here with `checking/copy_shared_tools.py`, which keeps
`settings.py`, the license block of each file and the tools below that SushiSkills does not hold.

| Path | Checks | Takes |
| --- | --- | --- |
| `check_all.py` | Nothing itself; runs the eleven checkers, then the repository's own `<area>/check_<what>.py` files and the tests, and prints only what failed | Repository root |
| `checking/checker.py` | Nothing itself; runs a checker's rule table and prints findings | |
| `checking/copy_shared_tools.py` | Nothing; copies these files, `.github/CODE_OF_CONDUCT.md` and `.github/SECURITY.md` into a repository, keeping its license blocks, `settings.py`, this README and its own tools, or lists the files that differ with `--report` | Repository root, `--from` |
| `checking/ignored_paths.py` | Nothing itself; names the files and folders git ignores, which no checker reads | |
| `settings.py` | Nothing; holds the settings that differ between repositories | |
| `documentation/check_source_comments.py` | C++, GLSL, TypeScript and Python: license block, file header, block ceiling, `//` and `#` runs, separators, history words, `@date` | Files or folders |
| `documentation/check_docs_layout.py` | `docs/` entries, required documents, document names, work folders, status lines in the four forms, design ceiling, links under `docs/`, reachability from `docs/README.md`, READMEs under `modules/<tier>/`, archive candidates by the date in the status line, roadmap, journal, handover, milestone, priority and to-do files | Repository root |
| `documentation/check_docs_truth.py` | The manual against the source: named symbols, cited paths, example sources, history words, page length, duplicated passages, em dashes, the allow-list | Repository root, `--source` |
| `documentation/check_changelog.py` | Headings and entry shape, releases kept live, length, one sentence, nesting, cited places | Repository root |
| `documentation/check_backlog.py` | `REMAINING_WORK.md` and `KNOWN_ISSUES.md`: shape, IDs and the counter, paths, references here and in siblings checked out beside the repository, entry size, a backlog entry for every open design document; `--report` lists the siblings read and the entries naming one | Repository root |
| `documentation/check_design_format.py` | Design documents: the five sections in order, a status line of at most 160 characters, the phase table and its states, no commit hash, no line that dates built or merged work | Repository root |
| `documentation/design_status.py` | Nothing itself; finds the design documents and work folders and reads their status lines | |
| `layering/check_layering.py` | Declared tiers and domains, upward includes from a module or its tests, cross-domain and unlisted same-tier includes, reaches into another module's `source/`, dependencies the build file lists, internal headers in installed ones | Repository root |
| `testing/check_test_mirror.py` | `tests/` mirrors the source: one `test_<name>` file per header or source file (`<name>.test.ts` in TypeScript), no test without a unit, no folder without a counterpart, no test that includes a header under `source/`; in a Unity repository, `Assets/<Product>/Tests/` with `<Name>Tests.cs` | Repository root |
| `testing/check_test_sizes.py` | Every test names its size in its source: a `Small_`, `Medium_` or `Large_` C++ suite, a size mark on a pytest file, class or function, a `small:` TypeScript `describe` | Repository root |
| `testing/check_probes.py` | Folders named as probes or benchmarks outside `applications/probes/` and `applications/benchmarks/`, such a program using a test framework or added to the test runner, one named after another of its kind, one its README does not list | Repository root |
| `logs/check_logging.py` | Console writes outside the paths `settings.py` allows, empty `catch` and `except: pass`, C++ log calls without their module's category, a logging module without `<module>_log_category.hpp`, Python root-logger, named-logger and f-string records | Repository root |
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

`settings.py` holds them. `K_TIER_ORDER` lists the tiers lowest first. `K_DOMAIN_TIERS` names the
tiers that share a rank and never include each other, here `vision` and `language`.
`K_MODULE_FUNCTION` names `sushiai_add_module`, so `check_layering.py` compares every include
with the `PUBLIC_DEPENDS` and `PRIVATE_DEPENDS` of the module's `CMakeLists.txt`, and reads the
same-tier edges from there; `K_SAME_TIER_EDGES` is empty.
`K_CLOSED` makes `check_source_comments.py` expect the closed licence lines, and
`K_MACRO_PREFIXES` names the prefixes `check_docs_truth.py` reads as macro names.
