# tools/contract

The class table of the error contract and the checker that holds the tree to it. The design is
[`SPEC.md`](../../docs/agent/2026_10_07_ERROR_CONTRACT_AND_NAMING/SPEC.md), sections 2.1, 5 and 6.

| File | Is |
| --- | --- |
| `site_classes.tsv` | One row per failure site of the modules' `include/` and `source/` folders and of `applications/`: where it is, why it fails and what it becomes |
| `check_contract.py` | The checker that compares the tree with the table |

## `site_classes.tsv`

Tab separated, UTF-8, one header row, sorted by file and then by line. A cell holds no tab and no
line break.

| Column | Holds |
| --- | --- |
| `file` | The path from the repository root, with forward slashes |
| `line` | The line of the site when the row was written. A hint: the checker does not match on it |
| `function` | The enclosing function as the classifier named it. For the reader and for the conversion scripts; the checker does not parse C++ scopes |
| `kind` | `macro` for an `SA_THROW_IF(` call, `throw` for a `throw` statement |
| `message_head` | The first 60 characters of the format string, adjacent literals joined, `\"` read as `"` |
| `cause` | `user-content`, `outside`, `precondition`, `invariant`, or `unclassified` |
| `mechanism` | `status`, `fatal`, or `unclassified` |
| `open_question` | Empty, or the keys of the questions the row's class waits on, joined by `;` |
| `reason` | One sentence: why the row has that class |

An `open_question` key is a `snake_case` word that names one question of the work folder's
`PLAN.md`, where the question is written out. A row that carries a key is not converted by a
script until the question is answered and the cell is emptied by hand. A key that starts with
`label_` leaves the mechanism settled: it questions the cause, the status code, the `function`
cell or how the row was read.

A row whose cause and mechanism are `unclassified` holds a site no classifier has read. The rows
of `applications/` are in that state until their wave.

A site and a row are the same site when `file`, `kind` and `message_head` agree. Two sites of one
file with the same head pair with their rows in line order.

The file carries no `format_version`: the header row is its schema, and the checker refuses any
other header.

## `check_contract.py`

```
python tools/contract/check_contract.py [repository root] [--report] [--rule NAME]
```

It reads every `.hpp`, `.cpp`, `.h` and `.inl` file under `modules/<tier>/<module>/include/`,
`modules/<tier>/<module>/source/` and `applications/`, and the table. It writes nothing. A
comment, a string literal and the `#define` of `SA_THROW_IF` itself hold no site. Any other
`#define` does: a macro that wraps `SA_THROW_IF` is a site of the file that defines it.

Root `tests/` and a module's `tests/` folder are not read. The test helpers that call
`SA_THROW_IF` are outside the ratchet.
SPEC section 6 names `convert_tests.py` to convert them; that script is not in this folder yet.

| Rule | Fails when |
| --- | --- |
| `rule_table_shape` | The table is missing, its header differs, a row is malformed, names an unknown class, holds an `open_question` cell that is no key list, or is out of order |
| `rule_ratchet` | A file holds more `SA_THROW_IF(` sites than rows of kind `macro` |
| `rule_unlisted_site` | A site has no row |
| `rule_stale_row` | A row has no site |

`K_FROZEN` names the files a later wave edits; it is empty since the SushiBLAS v0.1.0 migration. A surplus or unlisted site in one of them is
printed as a note and does not fail the run. A stale row fails wherever its file is.

Notes start with `note:` and never change the exit code:

| Note | Says |
| --- | --- |
| `note_frozen_growth` | A frozen file gained a site or lost a row's site |
| `note_unclassified` | A row has `unclassified` as its cause or mechanism |
| `note_open_question` | One line per `open_question` key, at the first row that carries it, with the number of rows |
| `note_bridge_count` | The `Detail::bridge(` calls per module, named `<tier>/<module>`, and in total; the last wave requires zero |

## When to run

- Before reporting any task that touched a module's `include/` or `source/` folder, or
  `applications/`.
- After a conversion wave: delete the rows of the sites the wave converted, in the same commit,
  and run the checker. A converted site whose row stays is a `rule_stale_row` finding.
- A new failure site uses `SA_FAIL_IF`, `SA_REFUSE_IF` or `SA_FATAL_IF`. A new `SA_THROW_IF` is a
  `rule_ratchet` finding; it is not answered by adding a row.

Its tests are `tools/tests/test_check_contract.py`.
