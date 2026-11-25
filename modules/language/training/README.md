# training

## Owns

The next-token objective, the fill of its target leaf and its metrics, and the objective
that adds the balance terms of the blocks with experts and reports how the experts are used. It
also holds the calibration of a classifier: the temperature fitted on a held-out set and the
expected calibration error.
Everything is in namespace `SushiAI::Language::Training`.

The module holds no training loop. `NextTokenObjective::feed` writes its target leaf from the
`labels` role of a batch, so `Train::Trainer` drives it as it drives any objective of the
layers tier; `tests/language/training/test_next_token_objective.cpp` does. A caller may still run a
`Train::TrainingStep` and call `fill_next_token_targets` itself, as
`tests/language/training/test_next_token_objective.cpp` does.

## Depends on

- `failure`, `dtype`, `shape`, `seam`, `graph`, `config`, `data`, `nn` and `train`
  (`PUBLIC_DEPENDS`): `Status`, the
  graph builder, the attributes a factory receives, `Data::Batch` and the role names,
  `NN::ModelOutputs`, and the `Objective` interface with its registry. Of the core modules, `dtype`,
  `shape` and `seam` are private (`PRIVATE_DEPENDS`).
- `allocation` (`PRIVATE_DEPENDS`), for the host copies of `host_copy.hpp`.

## Public entry points

| Header | Declares |
| --- | --- |
| `<SushiAI/language/training/next_token_objective.hpp>` | `NextTokenObjective`, `make_next_token_objective`, `K_NEXT_TOKEN_OBJECTIVE_TYPE`, `K_NEXT_TOKEN_OUTPUT_KEY`, `K_NEXT_TOKEN_TARGETS_LEAF` |
| `<SushiAI/language/training/expert_balance_objective.hpp>` | `ExpertBalanceObjective`, `ExpertUsageOutput`, `make_expert_balance_objective`, `K_EXPERT_BALANCE_OBJECTIVE_TYPE`, `K_BALANCE_OUTPUT_PREFIX`, `K_USAGE_OUTPUT_PREFIX`, `K_EXPERT_USAGE_MIN_METRIC`, `K_EXPERT_USAGE_MAX_METRIC` |
| `<SushiAI/language/training/next_token_targets.hpp>` | `fill_next_token_targets` |
| `<SushiAI/language/training/next_token_inputs.hpp>` | `check_next_token_inputs` |
| `<SushiAI/language/training/perplexity.hpp>` | `perplexity`, `next_token_metrics` |
| `<SushiAI/language/training/temperature_scaling.hpp>` | `fit_temperature`, `mean_negative_log_likelihood`, `K_MIN_TEMPERATURE`, `K_MAX_TEMPERATURE`, `K_TEMPERATURE_LOG_TOLERANCE` |
| `<SushiAI/language/training/calibration_error.hpp>` | `expected_calibration_error`, `K_CALIBRATION_BINS` |

## The loss

For `[N, T, V]` logits and `[N, T]` labels the loss is the mean, over the `R = N * T`
positions, of minus the log-probability of each position's label. The objective emits:

| Step | Operation | Result |
| --- | --- | --- |
| 1 | `RESHAPE` of the logits | `[R, V]`, one row per position |
| 2 | `LOG_SOFTMAX` | `[R, V]` log-probabilities |
| 3 | `TAKE_ALONG` by the `targets` leaf | `[R]`, the log-probability of each label |
| 4 | `RESHAPE`, `SUM_ROWS`, then `SCALE` by `-1 / R` | `[1]`, the loss |

The `targets` leaf is int32 of `[R]`. `fill_next_token_targets` writes element `r` as
`labels[r]`, on the host, once per batch: `R` integers. It checks every label against
`V` first, so a refused batch leaves the leaf as it was. The fill has two overloads. One takes
the labels and the leaf as spans of int32 and holds every rule of the fill: the counts, a
vocabulary of at least 1 and the range of each label. The other takes tensors, refuses a tensor
that is not int32 and passes their elements to the first. No product `R * V` enters an index, so
no size of the batch or the vocabulary is refused.

`feed` also checks every input id of the batch with `check_next_token_inputs`, before it asks the pool for a tensor: an id below 0 or at or above `V` is `Incompatible`, naming its position, the id and the row count, so it never reaches the gather kernel. The bound is `ObjectiveContext::input_classes`, the `Embedding::num_embeddings` of the first embedding of the model, when it is not 0; `training_setup.cpp` fills it. A context that states none (a model with no embedding, or a caller that builds the objective by hand) bounds the ids by `V`, the last extent of the logits, which is the embedding's row count when the head is tied to it. `NextTokenObjective::make` takes the same value as `input_classes`.

### Why not the one-hot loss

A reshape to `[R, V]` in front of `LOG_SOFTMAX` and `NLL_LOSS` would be enough to keep the
fused gradient: `nll_loss_backward` tests that the producer of its log-probabilities is a
`LOG_SOFTMAX` and then emits `(exp(L) - Y) / R` straight to the logits
(`modules/core/autograd/source/rules/loss.cpp`). That path needs `Y`, a float leaf of
`[R, V]`, which `fill_one_hot` zeroes and rewrites from the host for every batch. For `N` =
32, `T` = 256 and `V` = 8192 that is 67 108 864 floats, 256 MiB written per step, against
8192 integers here.

What the take costs: the gradient reaches the logits through `PUT_ALONG` into an `[R, V]`
value and `LOG_SOFTMAX_BACKWARD`, and not through the fused `EXP_SUB_SCALE` kernel. Both
paths compute `(softmax - one_hot) / R`. The index is one column per row, so no product
`R * V` is indexed and `NextTokenObjective::make` refuses no size; a batch of 64 rows of 1024
positions over 50 257 tokens builds.

## The factory

`make_next_token_objective` takes one optional key, `output`: the name of the published
output that holds the logits. A model that publishes one output needs no key. It reads `N`,
`T` and `V` from that output, and refuses a model whose input is not `[T]` ids per sample, a
model that publishes several outputs when no key names one, a name no output has, logits that
are not `[N, T, V]`, and data that states more classes than `V`. The objective requires the
role `labels` as int32 of `[T]` with no class count, so a token file of a smaller vocabulary
serves the same step.

`NextTokenObjective::loss` stops the process when the model's output is not the `[N, T, V]`
the objective was made for. A factory call has already compared the two. An emitter that
refuses while the loss is traced returns its failure from `loss`, as it does in
`ExpertBalanceObjective::loss`.

## Metrics

`NextTokenObjective::measure` appends `perplexity`, the exponential of the loss in the
`Train::StepView` it is given, so a loop that prints the loss and the metrics of a step prints
both. `next_token_metrics(mean_loss)` returns `loss` and `perplexity` for a loop that holds
the loss alone.

As a term of a `sum` the objective is given the loss of the sum, so `perplexity` is then the
exponential of the fit plus every other weighted term. A `Train::StepView` holds one loss.

## The balance of the experts

A block with experts publishes two values beside its result: `balance`, one element, and
`usage`, one share per expert (`modules/language/blocks/README.md`, under
`moe_feed_forward.hpp`). `ExpertBalanceObjective` reads them as model outputs. It declares no
leaf, requires no role and its `feed` writes nothing.

| Reads | From | Gives |
| --- | --- | --- |
| Every balance output | `NN::ModelOutputs`, in `loss` | Their mean, as one element: 1 node per output for the view, and an add and a scale between several |
| Every usage output | `Train::StepView`, in `measure` | `expert_usage_min` and `expert_usage_max`, the extremes over every expert of every block |

`make_expert_balance_objective` takes no key. It takes every output whose name begins with
`balance` as a balance and every output whose name begins with `usage` as a usage, and
refuses a model that publishes none of either and a balance of more than one element. The
row is a term: a training file states it inside a `sum` beside `next_token`, with the weight
of the balance as the weight of the term.

A share is the mean probability the router gives an expert over the positions of the batch,
so the shares of one block sum to 1 and an equal split of `E` experts is `1 / E` each. An
`expert_usage_min` that falls towards 0 while `expert_usage_max` rises is a router that
stopped using an expert. `measure` reads float32 and float64 outputs; a half-precision usage
is refused by `Train::require_host_reals`.

## Calibration

Both functions are host-side and take a held-out set as `[rows, classes]` float logits, row-major,
and one int32 label per row. They return a `Result`: `InvalidDescription` for an empty set, fewer
than two classes, a logit count that is not `rows * classes`, a logit that is not finite, a label
outside `[0, classes)` naming its row, and a temperature that is not positive and finite. A head of
one logit, such as a yes or no, passes the two logits `[0, z]`.

`mean_negative_log_likelihood(logits, labels, classes, T)` is the mean of minus the log-probability
of each label under `softmax(logits / T)`, in nats. Each row is evaluated as `ln(1 + e) + (best
logit - label logit) / T` with `e` the sum of `exp((z - best) / T)` over the other classes, so a
tail too small to move 1 still counts and nothing overflows.

`fit_temperature(logits, labels, classes)` returns the `T` that minimises that loss. It searches
`ln T` between `ln K_MIN_TEMPERATURE` (0.01) and `ln K_MAX_TEMPERATURE` (100) by golden section and
stops when the bracket is `K_TEMPERATURE_LOG_TOLERANCE` (1e-8) wide, returning the middle of it. The
loss is convex in `1 / T`, so it crosses its slope once along `ln T` and the search finds the
minimum. The search holds no random source, so a set gives one `T`. A set the logits predict right
on every row has no finite minimum and returns `K_MIN_TEMPERATURE` within the bracket; a set wrong on
every row returns `K_MAX_TEMPERATURE`. The loss is flat to the precision of a double within about
`sqrt(u)` of the minimum, so `T` is known to about 1e-7, not to the width of the bracket.

`expected_calibration_error(logits, labels, classes, T = 1)` bins the rows by confidence, the largest
probability of `softmax(logits / T)`, into `K_CALIBRATION_BINS` (15) equal-width bins over `[0, 1]`.
A confidence `c` is in bin `min(floor(15 c), 14)`, so a confidence of exactly 1 is in the last bin
and one on a boundary is in the upper bin. A row is right when the first class of its largest logit
is its label. The error is the sum over the bins of the share of rows in the bin times the absolute
difference between its accuracy and its mean confidence; an empty bin adds nothing. A caller measures
the effect of scaling by passing the fitted `T` here.

## Registration

This module knows no registry. `modules/language/registration` adds the rows `next_token` and
`expert_balance` beside the node rows and the dataset rows of the domain, and the programs
call that module.

## Tests

`tests/language/training/` holds one file per header.

| File | Suites | What they use |
| --- | --- | --- |
| `test_next_token_objective.cpp` | `Small_LanguageTrainingNextTokenObjective*`, `Medium_NextTokenObjective`, `Medium_NextTokenTrainer`, `Medium_DecoderOverfit` | The small suites build graphs and refuse descriptions on the host; the medium ones compare the loss and its gradient with a host computation, fit a bigram table through `Train::Trainer`, and overfit a learned-position decoder and a rotary decoder |
| `test_next_token_targets.cpp` | `Small_LanguageTrainingNextTokenTargets*` | Spans of values and shape-only tensors |
| `test_perplexity.cpp` | `Small_LanguageTrainingPerplexity*` | Doubles |
| `test_temperature_scaling.cpp` | `Small_LanguageTrainingTemperatureScaling*` | Spans of float logits and int32 labels; no engine |
| `test_calibration_error.cpp` | `Small_LanguageTrainingCalibrationError*` | The same spans, hand-computed bins |
| `test_expert_balance_objective.cpp` | `Small_LanguageTrainingExpertBalance*`, `Medium_LanguageTrainingExpertBalance*` | The host graph evaluator for the loss; an engine for the usage tensors of `measure` |

`tests/language/registration/test_registration.cpp` trains a decoder of experts with and
without the balance.

Source: `docs/design/LANGUAGE/README.md`, phases 4, 6 and 7.
