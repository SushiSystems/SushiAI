# training

## Owns

The next-token objective, the fill of its target leaf and its metrics, and the objective
that adds the balance terms of the blocks with experts and reports how the experts are used.
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

## Public entry points

| Header | Declares |
| --- | --- |
| `<SushiAI/language/training/next_token_objective.hpp>` | `NextTokenObjective`, `make_next_token_objective`, `K_NEXT_TOKEN_OBJECTIVE_TYPE`, `K_NEXT_TOKEN_OUTPUT_KEY`, `K_NEXT_TOKEN_TARGETS_LEAF` |
| `<SushiAI/language/training/expert_balance_objective.hpp>` | `ExpertBalanceObjective`, `ExpertUsageOutput`, `make_expert_balance_objective`, `K_EXPERT_BALANCE_OBJECTIVE_TYPE`, `K_BALANCE_OUTPUT_PREFIX`, `K_USAGE_OUTPUT_PREFIX`, `K_EXPERT_USAGE_MIN_METRIC`, `K_EXPERT_USAGE_MAX_METRIC` |
| `<SushiAI/language/training/next_token_targets.hpp>` | `fill_next_token_targets` |
| `<SushiAI/language/training/perplexity.hpp>` | `perplexity`, `next_token_metrics` |

## The loss

For `[N, T, V]` logits and `[N, T]` labels the loss is the mean, over the `R = N * T`
positions, of minus the log-probability of each position's label. The objective emits:

| Step | Operation | Result |
| --- | --- | --- |
| 1 | `RESHAPE` of the logits | `[R, V]`, one row per position |
| 2 | `LOG_SOFTMAX` | `[R, V]` log-probabilities |
| 3 | `RESHAPE` | `[R * V, 1]`, a table of single values |
| 4 | `GATHER_ROWS` by the `targets` leaf | `[R, 1]`, the log-probability of each label |
| 5 | `SUM_ROWS`, then `SCALE` by `-1 / R` | `[1]`, the loss |

The `targets` leaf is int32 of `[R]`. `fill_next_token_targets` writes element `r` as
`r * V + labels[r]`, on the host, once per batch: `R` integers. It checks every label against
`V` first, so a refused batch leaves the leaf as it was.

### Why not the one-hot loss

A reshape to `[R, V]` in front of `LOG_SOFTMAX` and `NLL_LOSS` would be enough to keep the
fused gradient: `nll_loss_backward` tests that the producer of its log-probabilities is a
`LOG_SOFTMAX` and then emits `(exp(L) - Y) / R` straight to the logits
(`modules/core/autograd/source/rules/loss.cpp`). That path needs `Y`, a float leaf of
`[R, V]`, which `fill_one_hot` zeroes and rewrites from the host for every batch. For `N` =
32, `T` = 256 and `V` = 8192 that is 67 108 864 floats, 256 MiB written per step, against
8192 integers here.

What the gather costs: the gradient reaches the logits through `SCATTER_ADD_ROWS` into a
`[R * V, 1]` value and `LOG_SOFTMAX_BACKWARD`, and not through the fused `EXP_SUB_SCALE`
kernel. Both paths compute `(softmax - one_hot) / R`. The index is int32, so
`NextTokenObjective::make` refuses `N * T * V` above 2 147 483 647; a batch of 64 rows of 1024
positions over 50 257 tokens does not fit.

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

## Registration

This module knows no registry. `modules/language/registration` adds the rows `next_token` and
`expert_balance` beside the node rows and the dataset rows of the domain, and the programs
call that module.

## Tests

`tests/language/training/test_next_token_objective.cpp` and `test_training_next_token_targets.cpp`, in
the suites `Medium_LanguageTrainingNextTokenObjective`, `Medium_LanguageTrainingNextTokenTargets`
and `Small_LanguageTrainingPerplexity`. The second suite allocates tensors on an engine.
`tests/language/training/test_expert_balance_objective.cpp` holds
`Small_LanguageTrainingExpertBalanceObjective` and, on an engine,
`Medium_LanguageTrainingExpertBalanceMetrics`.
`tests/language/training/test_next_token_objective.cpp` compares the loss and its gradient with a
host computation, `tests/language/training/test_next_token_objective.cpp` trains with it,
`tests/language/training/test_next_token_objective.cpp` fits it through `Train::Trainer`, and
`tests/language/registration/test_registration.cpp` trains a decoder of experts with and without the balance.

Source: `docs/design/LANGUAGE/README.md`, phases 4 and 7.
