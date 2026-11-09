# tests/language/datasets

The tests of the module `language/datasets`, one file per header of `modules/language/datasets/include/`. The suite prefix of a case gives its ctest label. Each test file has a sidecar of the same name with the `.md` suffix, which lists its suites and describes every case.

The logic of the module runs `Small_`: the window count, the collator's description and refusals, the row layout of a prompt, the keys of a data block, and the token file format over `std::ostringstream` and `std::istringstream`. `Medium_` stays where a case reaches a real boundary: an engine's tensors, a file on the disk, or a process stop.

## test_collator.cpp

Suites: `Small_LanguageDatasetsCollator`. Cases: see `test_collator.md`.

Tests that a caller sizes a batch from the `Collator` interface in `collator.hpp`.

## test_next_token_collator.cpp

Suites: `Small_LanguageDatasetsNextTokenCollator`, `Small_LanguageDatasetsNextTokenCollatorMake`, `Small_LanguageDatasetsNextTokenCollatorRefusal`, `Medium_LanguageDatasetsNextTokenCollator`. Cases: see `test_next_token_collator.md`.

Tests `NextTokenCollator` in `next_token_collator.hpp`.

## test_prompt_dataset.cpp

Suites: `Small_LanguageDatasetsPromptDataset`, `Small_LanguageDatasetsPromptDatasetLayout`, `Small_LanguageDatasetsPromptDatasetBlock`, `Medium_LanguageDatasetsPromptDataset`, `Medium_LanguageDatasetsPromptDatasetRange`. Cases: see `test_prompt_dataset.md`.

Tests `PromptDataset`, `lay_out_prompt`, `read_prompt_section` and `make_prompt_dataset` in `prompt_dataset.hpp`.

## test_token_dataset.cpp

Suites: `Medium_LanguageDatasetsTokenDataset`, `Medium_LanguageDatasetsTokenDatasetStride`, `Medium_LanguageDatasetsTokenDatasetRefusal`, `Medium_LanguageDatasetsTokenDatasetRange`. Cases: see `test_token_dataset.md`.

Tests `TokenDataset` in `token_dataset.hpp` over a token file in a scratch directory.

## test_token_file.cpp

Suites: `Small_LanguageDatasetsTokenFile`, `Small_LanguageDatasetsTokenFileWriterRefusal`, `Small_LanguageDatasetsTokenFileTruncation`, `Small_LanguageDatasetsTokenFileHeaderRefusal`, `Small_LanguageDatasetsTokenFileIdRefusal`, `Medium_LanguageDatasetsTokenFile`. Cases: see `test_token_file.md`.

Tests `write_token_file`, `read_token_file_header` and `read_token_range` in `token_file.hpp`.

## test_tokens_section.cpp

Suites: `Small_LanguageDatasetsTokensSection`, `Small_LanguageDatasetsTokensSectionRefusal`, `Medium_LanguageDatasetsTokensSection`. Cases: see `test_tokens_section.md`.

Tests `read_tokens_section` and `make_tokens_dataset` in `tokens_section.hpp`.

## test_window_layout.cpp

Suites: `Small_LanguageDatasetsWindowLayout`, `Small_LanguageDatasetsWindowLayoutRefusal`. Cases: see `test_window_layout.md`.

Tests `count_windows` and `first_token_of` in `window_layout.hpp`.
