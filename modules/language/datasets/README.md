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

`TokenDataset` runs no thread, shuffles nothing and keeps no state between reads.

The module also holds the two factories a data block reaches a dataset through,
`make_tokens_dataset` and `make_prompt_dataset`, and `PromptDataset`, the one batch a
generation starts from. `modules/language/registration` adds both factories to a
`Data::DatasetRegistry`.

## Depends on

- `core` and `data` (`PUBLIC_DEPENDS`), for `Status`, `Shape`, `Data::Dataset`, `Data::Batch`
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
| `<SushiAI/language/datasets/tokens_section.hpp>` | `make_tokens_dataset`, `K_TOKENS_DATA_TYPE` |
| `<SushiAI/language/datasets/prompt_dataset.hpp>` | `PromptDataset`, `make_prompt_dataset`, `K_PROMPT_DATA_TYPE`, `K_ROLE_PROMPT_LENGTH` |

## The data blocks

`make_tokens_dataset` builds a `TokenDataset` with a `NextTokenCollator`:

```json
{ "type": "tokens", "file": "corpus.satk", "window": 256, "stride": 256, "batch_size": 32 }
```

| Key | Required | Holds |
| --- | --- | --- |
| `file` | Yes | The token file; a relative path is relative to the working directory |
| `window` | Yes | `T`, the positions of one input row; a window reads `T + 1` ids |
| `stride` | No | The ids between the starts of two windows; what `TokenDatasetOptions` gives when absent |
| `batch_size` | No | The windows of one batch; 16 when absent |

The factory reads the header of the file first, because the collator states the vocabulary
size and only the header knows it.

`make_prompt_dataset` builds a `PromptDataset`:

```json
{ "type": "prompt", "text": "Once upon a time", "vocabulary": "vocab.savb", "window": 256 }
```

All three keys are required. The factory reads the vocabulary file, encodes `text` with
`Text::encode_text` and keeps the last `window` ids. A key neither factory lists is refused.

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

`write_token_file` checks every id against the vocabulary size before it opens the file, so a
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

The starts are strided and the same in every epoch. `Data::Dataset` offers
`batch_count()` and `read(index, batch) const`: it is not told which epoch it is in, and
`read` may be called from several threads, so a dataset has nowhere to keep a generator that
advances per epoch. A start drawn per epoch needs the epoch in the call. `TokenDataset`
defaults the stride to the window length, which reads every id once per epoch; a smaller
stride gives overlapping windows.

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

## Tests

`tests/test_datasets_token_file.cpp`, `test_datasets_window_layout.cpp`,
`test_datasets_next_token_collator.cpp` and `test_datasets_token_dataset.cpp`, in the suites
`Unit_LanguageDatasetsTokenFile`, `Unit_LanguageDatasetsWindowLayout`,
`Unit_LanguageDatasetsNextTokenCollator` and `Unit_LanguageDatasetsTokenDataset`. The last two
allocate tensors on an engine. The files they write are under the system's temporary directory
and are removed when the test ends.
