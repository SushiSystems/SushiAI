# datasets

## Owns

The token file, which holds the ids of a tokenised corpus; the layout of fixed-length windows
over it; the collator interface of the language tasks with the collator of the next-token task;
and `TokenDataset`, the `Data::Dataset` that reads one batch of windows each time it is asked.
The module owns the token file format: its schema, its reader and its writer. Everything is in
namespace `SushiAI::Language::Datasets`.

A corpus is tokenised once, with `Text::encode_text`, and written with `write_token_file`.
Training reads ids and never text. The module calls no tokenizer; it takes `TokenID` and
`K_MAX_VOCABULARY_SIZE` from `language/text` and nothing else.

`TokenDataset` runs no thread and keeps no state between reads but the epoch it was told, which
only a shuffled dataset reads.

The module also holds the factories a data block reaches a dataset through,
`make_tokens_dataset`, `make_prompt_dataset` and `make_decisions_dataset`, and `PromptDataset`,
the one batch a generation starts from. The decision set has its own file and dataset: the
decision file, which holds the questions of the decision model with their answers, and
`DecisionDataset`, which batches them. `modules/language/registration` adds the factories to a
`Data::DatasetRegistry`.

## Depends on

- `failure`, `dtype`, `shape` and `data` (`PUBLIC_DEPENDS`), for `Status`, `Shape`, `Data::Dataset`,
  `Data::Batch`
  and the role names.
- `language/text` (`PUBLIC_DEPENDS`), for `Text::TokenID` in the public signatures, and for
  the vocabulary file and the encoder `make_prompt_dataset` reads a text with.
- `config` (`PUBLIC_DEPENDS`), for `Config::Value` and for `Config::SectionReader`, which
  `source/section.hpp` reads the keys of a data block with.

## Public entry points

| Header | Declares |
| --- | --- |
| `<SushiAI/language/datasets/token_file.hpp>` | `TokenFileHeader`, `write_token_file`, `read_token_file_header`, `read_token_range` over a stream and over a path, the magic, the version and the header size |
| `<SushiAI/language/datasets/window_layout.hpp>` | `WindowLayout`, `count_windows`, `first_token_of`, `K_MAX_WINDOW_TOKENS` |
| `<SushiAI/language/datasets/collator.hpp>` | `Collator` |
| `<SushiAI/language/datasets/next_token_collator.hpp>` | `NextTokenCollator`, `K_TASK_NEXT_TOKEN` |
| `<SushiAI/language/datasets/token_dataset.hpp>` | `TokenDatasetOptions`, `TokenDataset` |
| `<SushiAI/language/datasets/tokens_section.hpp>` | `TokensSection`, `read_tokens_section`, `make_tokens_dataset`, `K_TOKENS_DATA_TYPE` |
| `<SushiAI/language/datasets/prompt_dataset.hpp>` | `PromptRow`, `lay_out_prompt`, `PromptSection`, `read_prompt_section`, `PromptDataset`, `make_prompt_dataset`, `K_PROMPT_DATA_TYPE`, `K_ROLE_PROMPT_LENGTH` |
| `<SushiAI/language/datasets/decision_file.hpp>` | `DecisionType`, `DecisionRecord`, `DecisionFile`, `write_decision_file`, `read_decision_file` over a stream and over a path, the magic, the version, the header size and the list limits |
| `<SushiAI/language/datasets/decision_dataset.hpp>` | `DecisionShape`, `DecisionRows`, `check_decision_fits`, `lay_out_decisions`, `DecisionDatasetOptions`, `DecisionDataset`, `DecisionsSection`, `read_decisions_section`, `make_decisions_dataset`, `K_DECISIONS_DATA_TYPE`, `K_ROLE_LENGTHS`, `K_ROLE_OPTIONS`, `K_ROLE_OPTION_LENGTHS`, `K_ROLE_TYPE_MASK`, `K_DECISION_TYPES` |

## The data blocks

`make_tokens_dataset` builds a `TokenDataset` with a `NextTokenCollator`:

```json
{ "type": "tokens", "file": "corpus.satk", "window": 256, "stride": 256, "batch_size": 32,
  "shuffle": true, "seed": 1 }
```

| Key | Required | Holds |
| --- | --- | --- |
| `file` | Yes | The token file; a relative path is relative to the working directory |
| `window` | Yes | `T`, the positions of one input row; a window reads `T + 1` ids |
| `stride` | No | The ids between the starts of two windows; what `TokenDatasetOptions` gives when absent |
| `batch_size` | No | The windows of one batch; 16 when absent |
| `shuffle` | No | `true` redraws the window starts in every epoch; `false` when absent |
| `seed` | No | The seed of those draws, any 64-bit pattern; 0 when absent |

`read_tokens_section` reads and checks the keys alone, and returns them as a `TokensSection`.
The factory then reads the header of the file, because the collator states the vocabulary
size and only the header knows it.

`make_prompt_dataset` builds a `PromptDataset`:

```json
{ "type": "prompt", "text": "Once upon a time", "vocabulary": "vocab.savb", "window": 256 }
```

All three keys are required. `read_prompt_section` reads and checks them alone. The factory
reads the vocabulary file, encodes `text` with `Text::encode_text` and keeps the last `window`
ids; `lay_out_prompt` states that row, the kept ids and then zeros up to the window, with no
engine. A key neither factory lists is refused.

`make_decisions_dataset` builds a `DecisionDataset`:

```json
{ "type": "decisions", "file": "train.sadc", "window": 128, "options": 8, "batch_size": 32 }
```

| Key | Required | Holds |
| --- | --- | --- |
| `file` | Yes | The decision file of one split; a relative path is relative to the working directory |
| `window` | Yes | `T`, the ids every text and every option is padded to |
| `options` | Yes | `K`, the options every choice is padded to |
| `batch_size` | No | The records of one batch; 16 when absent |

`read_decisions_section` reads and checks the keys alone, and returns them as a `DecisionsSection`.

## PromptDataset

One batch of one row. The inputs are `[1, window]` int32: the kept ids at the front and zeros
behind them. The role `prompt_length`, an int32 of `[1, 1]`, holds how many ids are the
prompt, since a zero is a token id too. `make` refuses a prompt of no id, a negative id and a
window outside 1 to `K_MAX_WINDOW_TOKENS`. `sushiai inference` reads this batch for a model
whose task is `next_token_generation`.

## The token file

A binary file, by convention `*.satk`. Every integer is little-endian, written and read byte by
byte whatever the host's byte order. An id is a signed 32-bit integer in two's complement.

| Offset | Bytes | Field | Rule |
| --- | --- | --- | --- |
| 0 | 4 | `magic` | The bytes `S`, `A`, `T`, `K` |
| 4 | 4 | `format_version` | Unsigned; `1` |
| 8 | 4 | `vocabulary_size` | Unsigned, in `[1, 1 048 576]`; every id is in `[0, vocabulary_size)` |
| 12 | 4 | `id_bytes` | Unsigned; `4` |
| 16 | 8 | `token_count` | Unsigned; `N` |
| 24 | `4 * N` | Ids | Id `i` is at byte `24 + 4 * i` |

The file ends at byte `24 + 4 * N`. The file of the ids 1, 258 and 0 over 300 tokens is 36
bytes: `53 41 54 4B`, `01 00 00 00`, `2C 01 00 00`, `04 00 00 00`, `03 00 00 00 00 00 00 00`,
then `01 00 00 00`, `02 01 00 00`, `00 00 00 00`.

`write_token_file` and `read_token_file_header` also take a stream, so the format is read and
written in memory with no file. `write_token_file` checks every id against the vocabulary size before it opens the file, so a
refused corpus truncates nothing. `read_token_file_header` refuses, with `InvalidFile` and the
byte offset in the message: a file that ends before the end its header declares, bytes after
that end, another magic, a version it does not know, and a field out of its bounds.
`read_token_range` seeks to the first id asked for and reads that range alone, so the file is
never loaded whole. It checks every id it reads against the header's vocabulary size, and an id
outside it is `InvalidFile` naming the id and its byte offset.

| Version | Release | Change |
| --- | --- | --- |
| 1 | 0.1.0, not yet released | The first version |

## Windows

A window is a fixed number of consecutive ids. `WindowLayout{tokens, stride}` states that
window `w` reads `tokens` ids from id `w * stride`, and `count_windows` returns
`(N - tokens) / stride + 1` for a file of `N` ids, or 0 when `N` is below `tokens`. Only whole
windows are counted.

Without `shuffle` the starts are strided and the same in every epoch. `TokenDataset` defaults the
stride to the window length, which reads every id once per epoch; a smaller stride gives
overlapping windows.

`TokenDataset` implements `Data::EpochAware`: `begin_epoch(epoch)` stores the epoch, and a read
takes it once at its start. With `shuffle`, the epoch gives the starts two draws, both from
`Data::EpochPermutation`, so they depend on the seed and the epoch alone and a thread may read
while no thread calls `begin_epoch`:

- Position `p` of the epoch reads window `EpochPermutation(seed, epoch, windows).at(p)`, so
  every window is read once in an order of its own. The windows past the last whole batch differ
  from epoch to epoch.
- One offset in `[0, slack]` moves every start of the epoch. `slack` is
  `N - tokens - (windows - 1) * stride`, the ids the last window leaves over; it is below the
  stride, and the last window still ends inside the file. The offset is
  `EpochPermutation(seed, epoch, slack + 1).at(0)`.

Two epochs of one seed give different starts, and one seed and one epoch give the same starts in
every dataset (RW-90). The epoch is 0 until `begin_epoch` is called.

## Collators

A `Collator` knows one task. It states how many ids of the file one row needs
(`tokens_per_window()`), the positions of one input row (`window_length()`, `T`), the
vocabulary size its ids must be below, and the roles it writes. `collate` takes the ids of `N`
windows one after another and writes `Batch::inputs`, int32 of `[N, T]`, and its roles.

`NextTokenCollator` reads `T + 1` ids `w` per window. Row `n` of the input holds `w[0..T)` and
row `n` of the role `labels` (`Data::K_ROLE_LABELS`), int32 of `[N, T]`, holds `w[1..T]`:
`labels[n][t]` is the id that follows `inputs[n][t]`. Its `TargetDescription` has the
per-sample shape `[T]`, the element type `Int32` and `classes` equal to the vocabulary size.

## TokenDataset

`TokenDataset::make(engine, file, collator, options)` reads the header of the token file and
refuses a file whose vocabulary size is above the collator's, a batch size or a stride below 1,
and a file that holds fewer windows than one batch. Batch `i` holds windows
`[i * batch_size, (i + 1) * batch_size)`; the windows past the last whole batch are not read.

`read(index, batch)` opens the file, reads each window with `read_token_range`, prepares the
batch and calls the collator. Every id is checked against the vocabulary size of the file
before the collator writes it, and one outside it fails the read with the file, the id and its
byte offset. A batch passed again is written in place when nothing else references its tensors.

`sample_shape()` is `[T]` and `sample_dtype()` is `Int32`. `read` passes
`Data::describe_input(*this)` to `Data::prepare_batch`, which allocates and reuses the int32
input and the tensors of the roles. A step whose input leaf is not int32 of `[N, T]` refuses
the dataset in `Train::match_input`.

## The decision file

A binary file, by convention `*.sadc`, one per split. Every integer is little-endian, written and
read byte by byte. An id is a signed 32-bit integer. The file ends at the end of its last record.

| Offset | Bytes | Field | Rule |
| --- | --- | --- | --- |
| 0 | 4 | `magic` | The bytes `S`, `A`, `D`, `C` |
| 4 | 4 | `format_version` | Unsigned; `1` |
| 8 | 4 | `vocabulary_size` | Unsigned, in `[1, 1 048 576]`; every id is in `[0, vocabulary_size)` |
| 12 | 8 | `record_count` | Unsigned; `R` |
| 20 | | Records | `R` records, one after the other |

A record is four words, then its text, then its options:

| Word | Holds |
| --- | --- |
| `type` | `0` choice, `1` yes or no, `2` number |
| `answer` | A choice: the index of the right option. A yes or no: `0` or `1`. A number: the bits of an IEEE float |
| `text_count` | `n`, the ids of the text, in `[1, 1 048 576]` |
| `option_count` | `k`, the options, in `[1, 65 536]` for a choice and `0` for the other types |

The `n` ids of the text follow, then `k` options, each a word `m` in `[1, 1 048 576]` and `m` ids.
The file of one yes-or-no record with the text `[5]` over 10 tokens is 40 bytes: `53 41 44 43`,
`01 00 00 00`, `0A 00 00 00`, `01 00 00 00 00 00 00 00`, then `01 00 00 00`, `01 00 00 00`,
`01 00 00 00`, `00 00 00 00`, `05 00 00 00`.

A record holds the answer of its own type alone: `DecisionRecord` states `choice`, `yes` and
`number`, and the two a type does not use are `0`, `false` and `0`, so a record read back equals the
record written. `write_decision_file` checks every record before it opens the file, so a refused
set truncates nothing. `read_decision_file` reads the whole file and refuses, with `InvalidFile` and
the byte offset: a file that ends inside a field, bytes after the last record, another magic or
version, a count a file of that size cannot hold, a type or a yes-or-no answer out of range, an id
outside the vocabulary, an option index past the options and a number that is not finite.

| Version | Release | Change |
| --- | --- | --- |
| 1 | 0.1.0, not yet released | The first version |

## DecisionDataset

`DecisionDataset::make(engine, file, options)` reads the whole decision file, since a record has no
fixed size, and refuses a batch size or an extent out of its bounds, a record whose text or option
is longer than `T` or whose options are more than `K`, and a file of fewer records than one batch.
Batch `i` holds records `[i * batch_size, (i + 1) * batch_size)`; the records past the last whole
batch are not read. `lay_out_decisions` is the layout without an engine: it pads the rows of
records to `T` and `K` and returns them as host arrays, and `read` copies them into the tensors.

The batch holds the texts as its input, int32 `[N, T]`, and eight roles, in this order:

| Role | Type and shape | Holds |
| --- | --- | --- |
| `lengths` | float32 `[N, 1]` | The ids of each text; the rest of the row is `0` |
| `options` | int32 `[N, K, T]` | The ids of each option, then `0`; an option the record lacks is all `0` |
| `option_lengths` | float32 `[N, K]` | The ids of each option; `0` for an option the record lacks |
| `option_valid` | float32 `[N, K]` | `1` for an option of at least one id, `0` for any other |
| `choice` | int32 `[N]` | A choice: the index of the right option, over `K` classes. Another type: `0` |
| `yes_no` | float32 `[N, 1]` | A yes: `1`. A no: `0`. Another type: `0` |
| `number` | float32 `[N, 1]` | A number: its value. Another type: `0` |
| `type_mask` | float32 `[N, 3]` | One `1` in the column of the type: choice, yes or no, number |

The lengths and the flags are float32 because a decision model reads them as real values, through the
extra inputs of its model file (`docs/guides/MODEL_FILE.md`, section "`inputs`"). The three answer
roles carry the placeholder `0` on every row of another type, and an objective keeps its own rows
with `type_mask`, so a role holds no `-1` and no row needs a second pass. An objective reads a role
by its `target` key (`docs/guides/RUNNING_MODELS.md`, section "One objective over several outputs").

The input is not a role: the texts are `Batch::inputs`, as the token ids are in `TokenDataset`.
`read` keeps no state, so several threads may read at once, each with its own `Batch`.

## Tests

`tests/language/datasets/` holds one test file per header; its README lists them and each file has
a sidecar that describes its cases. The window count, the collators' description and refusals,
`lay_out_prompt`, `lay_out_decisions`, the keys of the three data blocks, and the token file and
the decision file over in-memory streams run in `Small_` suites. The suites that allocate tensors on an engine, read a file from the disk
or stop the process are `Medium_`. The files they write are under the system's temporary
directory and are removed when the test ends.
