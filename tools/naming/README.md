# tools/naming

The two tools of the naming wave and the map between them. The design is
[`SPEC.md`](../../docs/agent/2026_10_07_ERROR_CONTRACT_AND_NAMING/SPEC.md), section 2.13.

| File | Is |
| --- | --- |
| `build_rename_map.py` | Reads the tree and writes the map |
| `rename_map.tsv` | The map: the artefact a person reviews before the rename runs |
| `apply_rename.py` | Reads the map and rewrites the C++ tree. It owns the map's columns, its reader and its writer |

## `rename_map.tsv`

Tab separated, UTF-8, one header row, sorted by kind, then by old name, then by declaring file.

| Column | Holds |
| --- | --- |
| `kind` | `constant`, `enumerator`, `interface`, `macro`, `enum_underlying`, `unscoped_enum`, `type_name`, `sibling_enumerator`, `sibling_name` or `string_spelling` |
| `old_name` | The present spelling. An enumerator is written `Enum::NAME` |
| `new_name` | The spelling after the wave; empty where a sibling decides it |
| `declaring_file` | The file or files that declare the name, joined with `; `. `(sibling)` for a name SushiBLAS or SushiRuntime declares |
| `occurrences` | Identifier tokens under `include/`, `src/`, `tests/`, `apps/` and `examples/`. For a `constant`, `enumerator`, `interface` or `macro`: the tokens the tool would rewrite if the row were applied, the declaration included. For a `sibling_enumerator`: every `Qualifier::NAME` pair, under a sibling's namespace or not. For any other kind: every token that spells the old name |
| `document_occurrences` | Whole-word mentions in Markdown under those five roots and `docs/`. The tool does not edit documents |
| `status` | `rename`, `by_hand`, `excluded` or `sibling` |
| `reason` | Why a row is not `rename` |

Only `rename` rows are applied.

| Status | Means |
| --- | --- |
| `rename` | `apply_rename.py` rewrites it |
| `by_hand` | The ruling changes it, and a whole-word replacement would be wrong: the new name is taken, the name is lower-case or shorter than three characters, an unscoped enum declares the same word, or the edit is no rename (an underlying type, an unscoped enum) |
| `excluded` | It does not change: a prefixed macro, a trait variable template (D12), a string that spells a renamed name, or a type that shares an enumerator's name |
| `sibling` | SushiBLAS or SushiRuntime declares it; it changes when the sibling renames it |

A `string_spelling` row lists a C++ string literal, a JSON string or a CLI flag whose text, read
without letter case, equals a renamed name. It is there for the reviewer: the tool never rewrites
a literal, so `"weight"` stays while `ParameterRole::WEIGHT` becomes `ParameterRole::Weight`.

A `type_name` row lists a class or struct whose name is also the last part of a renamed
enumerator, as the class `SGD` is beside `OptimizerKind::SGD`. The row tells the tool that a bare
`SGD` is explained, so it is not reported as an enumerator the rename missed.

An unscoped enum has one row per file that declares it, since two files may declare enums of one
name with different enumerators. Every other kind has one row per name, its files joined.

The file carries no `format_version`: the header row is its schema, and the reader refuses any
other header.

## `build_rename_map.py`

```
python tools/naming/build_rename_map.py [repository root] [--output PATH]
```

It reads the C++ files and the Markdown under the roots above, the JSON under `examples/` and
`tests/`, and the flags in `cli/`. It writes one file, the map, and prints the row count per kind
and status. It then prints one `unresolved: file:line: name` line for each token that would still
spell a renamed enumerator after a run of the whole map, and their count. `apply_rename.py`
refuses to run while that count is above zero. Each scanner is one function in `K_SCANNERS`:

| Scanner | Lists |
| --- | --- |
| `find_shared_type_names` | Classes and structs that have the name of a renamed enumerator |
| `find_enumerators` | Enumerators of an `enum class` not in PascalCase; an `enum class` with no underlying type; an unscoped enum |
| `find_constants` | `constexpr` and `static const` names not spelled `K_UPPER_SNAKE` |
| `find_interfaces` | Classes and structs named `I` and an upper-case letter |
| `find_macros` | Every `#define` |
| `find_sibling_names` | Upper-case names qualified by `SushiBLAS`, `SushiRuntime`, `sb`, `sr`, or by an alias of `DataType`, `Layout` or `ExecutionMode` |

`K_ENUMERATOR_EXCEPTIONS` holds the spellings fixed by hand (`NUL` to `Null`). A constant takes
`K_` and its present spelling; a camel-case name is split after each lower-case letter that an
upper-case one follows, so `IM2COL` gives `K_IM2COL` and `maxRank` gives `K_MAX_RANK`.

`find_constants` reads `constexpr` and `static const`. A namespace-scope constant declared with
plain `const` is not listed and is renamed by hand.

Run it again whenever the tree gained or lost a constant or an enumerator, and once more on the
frozen tree just before the rename. Commit the map it writes.

## `apply_rename.py`

```
python tools/naming/apply_rename.py [repository root] [--map PATH] [--kind KIND] [--dry-run]
```

It rewrites identifier tokens and nothing else:

- An enumerator changes in its qualified form, `Enum::NAME`, and anywhere inside the body of its
  own enum, as a declaration or in an initialiser.
- A constant, an interface or a macro changes as a whole word, unless the qualifier before it is
  an enum the map knows, the chain starts at a sibling's namespace (`SushiBLAS::` or
  `::SushiBLAS::`), or the word stands where an enum body declares an enumerator.
- A string literal, a character literal and an `#include` line never change.
- A comment changes only in the name that follows `@ref`, `@c` or `@see`. Other mentions are
  counted and printed as "comment mentions left".

It stops before writing anything when:

| Condition | Exit code |
| --- | --- |
| `git status --untracked-files=all` lists a path under `include/`, `src/`, `tests/`, `apps/` or `examples/` (not checked under `--dry-run`) | 2 |
| The map is absent or malformed | 2 |
| A new name is already a token in the tree | 1 |
| The string literals of a file would differ after the rewrite | 1 |
| A token still spells an applied enumerator and nothing explains it; each is printed as `file:line: name` | 1 |
| A second pass over a rewritten file would rename a token again | 1 |

A surviving enumerator is explained when an enum of the map qualifies it, when it stands in
another enum's body, or when another row lists the same word as an unqualified old name. What is
left is the list SPEC 2.13 gives to a person: `K::NAME` through an alias, `decltype(x)::NAME`, a
variable of the same spelling. Rename it by hand and commit, or give the word a row, then run
again.

`--dry-run` prints `path: count` per file and the total, and writes nothing.

`--kind` applies the `rename` rows of one kind: `enumerator`, `constant`, `interface` or `macro`.
The whole map is still read, so the rows left out keep protecting their enums and names.

Run it at the freeze of SPEC section 5, three times in the order SPEC 2.13 fixes, each run on a
clean tree and each as the only change of its commit:

```
python tools/naming/apply_rename.py --kind enumerator
python tools/naming/apply_rename.py --kind constant
python tools/naming/apply_rename.py --kind interface
```

The three runs read the same map. Do not rebuild it between them. It does not move
`nn/layer.hpp`, add an underlying type or rewrite an unscoped enum; those are the `by_hand` rows.
It does not compare the per-file token counts of safeguard 3, and it does not apply the inverse
map to its own output, which SPEC 2.13 lists as safeguard 4.

Their tests are `tools/tests/test_naming.py`.
