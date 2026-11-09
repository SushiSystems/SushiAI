# tests/language/training

The tests of the module `language/training`, one file per header of
`modules/language/training/include/`. The suite prefix of a case gives its ctest label.

## Test files

Each test file has a page beside it that says what its cases pin.

| Test file | Header | Suites | Page |
| --- | --- | --- | --- |
| `test_expert_balance_objective.cpp` | `expert_balance_objective.hpp` | `Small_LanguageTrainingExpertBalance*`, `Medium_LanguageTrainingExpertBalance*` | [`test_expert_balance_objective.md`](test_expert_balance_objective.md) |
| `test_next_token_objective.cpp` | `next_token_objective.hpp` | `Small_LanguageTrainingNextTokenObjective*`, `Medium_NextTokenObjective`, `Medium_NextTokenTrainer`, `Medium_DecoderOverfit` | [`test_next_token_objective.md`](test_next_token_objective.md) |
| `test_next_token_targets.cpp` | `next_token_targets.hpp` | `Small_LanguageTrainingNextTokenTargets*` | [`test_next_token_targets.md`](test_next_token_targets.md) |
| `test_perplexity.cpp` | `perplexity.hpp` | `Small_LanguageTrainingPerplexity*` | [`test_perplexity.md`](test_perplexity.md) |

## What the sizes use

The small suites use values only: spans, shape-only tensors, graphs built and read on the host,
and the host graph evaluator for the loss of the expert balance. The medium suites allocate
tensors on the engine (the usage tensors of `measure`), compare the loss and gradient of a step
with a host computation, fit a bigram table through `Train::Trainer`, and overfit a learned-position
decoder and a rotary decoder through the typed case `Medium_DecoderOverfit`.

Fixtures reach the test namespace as `SaTest`, which stays reachable inside a `TEST_F` body.
Tolerances come from `tools/testing/support/tolerance.hpp`.
