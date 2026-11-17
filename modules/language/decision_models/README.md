# language/decision_models

An internal module: only the test programs link it, its headers are never installed, and it is built only when `SA_BUILD_TESTS` is on.

## Owns

The decision model the language tests train end to end: the synthetic decision set and its rules,
the session that builds the model from `applications/examples/decision_model/model.json` and trains
it under `train.json` through the calls `sushiai train` makes, the fit of the two temperatures and
their place in the model file. The module holds no model and no objective of its own: the files do.

## Depends on

`config`, `data`, `execution`, `failure`, `graph`, `nn`, `optim`, `shape` and `train` of the lower
tiers, and `language/blocks` (the names of the three output ports), `language/datasets`,
`language/text` and `language/training` of its own. `dtype` and `seam` are private. The rows a model
file names come from the registry the caller hands the session; the test builds it with
`language/registration`.

## Public entry points

Each header is named by its include path under `include/SushiAI/language/decision_models/`.

| Header | Declares |
| --- | --- |
| `decision_model.hpp` | `DecisionExtents`, `K_TEST_EXTENTS`, `K_ECE_BOUND` |
| `decision_set.hpp` | `DecisionSetOptions`, `generate_decision_set`, `write_decision_set`, `right_option_of`, `yes_no_answer_of`, `number_answer_of` and the id ranges of the set |
| `decision_session.hpp` | `DecisionSessionOptions`, `DecisionSession` |
| `decision_predictions.hpp` | `DecisionPredictions`, `LogitRows`, `choice_rows`, `yes_no_rows` |
| `decision_calibration.hpp` | `DecisionCalibration`, `fit_decision_calibration`, `store_calibration`, `read_calibration`, the keys of the calibration |
| `decision_scores.hpp` | `DecisionScores`, `score_decisions` |
| `calibrate_decision_model.hpp` | `calibrate_decision_model` over a spec and over a model file |

`source/decision_run.hpp` is private: one compiled step over the shared state, which the session
owns twice.

## The synthetic set

`generate_decision_set` draws records from a seed, with the three types taking turns and the order
shuffled by the same seed. The ids are 0 for padding, 1 to 8 for the keys of a choice, 9 to 16 for its
options, 17 to 24 for a yes-or-no question and 25 to 31 for a number question, 32 in all. Each type
has one rule a small encoder can learn.

| Type | The question | The answer |
| --- | --- | --- |
| Choice | A key from 1 to 8 and a noise id from 25 to 31 | The option with the id `key + 8`, among 2 to `K` distinct options of one id each, at a random place |
| Yes or no | Three ids from 17 to 24 | Yes when at least two of the ids are 17 to 20 |
| Number | Three ids from 25 to 31 | The mean of `(id - 28) / 3`, a value in `[-1, 1]` |

`write_decision_set` writes the records as a decision file of `modules/language/datasets`.

## The two files

`K_TEST_EXTENTS` holds the window, the options and the batch the tests read, which `train.json` states
for its data. Nothing else of the model is in C++.

`model.json` declares the question as the model input, an int32 `[8]`, and four extra inputs, which
the `decisions` dataset publishes as roles of the same names: `lengths` (float32 `[1]`), `options`
(int32 `[4, 8]`), `option_lengths` and `option_valid` (float32 `[4]` each). The one `decision_heads` row
reads the five by `from`, and publishes the ports `choice`, `yes_no` and `number`.

`train.json` states a `sum` of three terms, each naming its output, the role it reads and the column of
`type_mask` that keeps its rows:

| Term | Output | Target | Column |
| --- | --- | --- | --- |
| `cross_entropy` | `choice` | `choice`, int32 over `K` classes | 0 |
| `binary_cross_entropy` | `yes_no` | `yes_no`, float32 `[1]` | 1 |
| `mean_squared_error` | `number` | `number`, float32 `[1]` | 2 |

A row of another type holds 0 in each role, and its column of the mask drops it from the loss, so the
three terms share one batch with no wrapper. The cross entropy takes its `K` from the width of the
`choice` output, since the dataset states no `labels` role. The data block, the optimizer and the
epochs are read from the same file.

## The session

`DecisionSession::make(backend, options, training)` takes the parsed model file, the parsed training
file, the registries of the rows and of the objective types, and the training dataset. It makes the
calls `TrainingSetup` makes for `sushiai train`: `NN::build_model`, `Train::build_probe_graph` for the
shapes the objective factory reads, `Train::make_objective`, `Train::build_training_graph`,
`Train::match_roles`, `NN::ModelState::create`, `Train::bind_parameters`,
`Train::bind_buffer_updates`, `Optim::make_optimizer` and `Train::Trainer::make`. It compiles two steps
over one state: a training step with the optimizer of the file, and a probe step with an SGD of rate 0.
A file that states a `schedule`, a `cluster`, `weights` or `resume` is refused with `Unsupported`,
since the session runs the epochs of the file with its optimizer as it states it. `fit` trains for the
epochs of the file. `predict` runs the probe step over every whole batch of a dataset and returns the
choice logits, the yes-or-no logits and the numbers of every record with its answers, so a prediction
reads the weights `fit` left and changes none. Both steps compile once, at `make`.

The rows `decision_heads` and `decisions` are rows of `language/registration`; a registry without them
refuses the model file naming the row.

## Calibration

`fit_decision_calibration` fits one temperature for each classifying head with `fit_temperature`
of `language/training`, on the rows of its own type: the choice head on its `[rows, K]` logits, and
the yes-or-no head on the two-class rows `{0, z}` for the logit `z` of a yes, whose softmax is
`sigmoid(z / T)`. The number head has none.

A model file carries scalar settings in its `task` object, whose `type` the domain owns, so the
temperatures need no new key of a node row. A decision file states

```json
"task": { "type": "decision", "calibration": { "choice": 1.31, "yes_no": 0.87 } }
```

| Key | Holds |
| --- | --- |
| `task.type` | `decision` |
| `task.calibration.choice` | The temperature that divides the choice logits before their softmax, in `[0.01, 100]` |
| `task.calibration.yes_no` | The temperature that divides the logit of a yes before its sigmoid, in `[0.01, 100]` |

`store_calibration` writes the object and keeps the other keys of the task, `read_calibration` reads
it and returns 1 and 1 for a `decision` task with none, and both refuse a task of another type.
`calibrate_decision_model` reads the answers of a session to the calibration split, fits, stores,
and for a model file on disk writes the file back. `score_decisions` gives each accuracy, the
mean absolute error of the numbers and the expected calibration error over `K_CALIBRATION_BINS` bins
of each classifying head, at the temperatures it is given. `K_ECE_BOUND`, 0.05, is the bound of the
ruling LANGUAGE 6 for the synthetic set.

## Tests

`tests/language/decision_models/` holds the test file and its page. The `Medium_` cases read the two
example files and check them against the dataset's roles; the `Large_` suite trains the model of the
files through `DecisionSession`. No case has been run.
