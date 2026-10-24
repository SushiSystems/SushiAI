# train

## Owns

The train module turns a model into the graphs a training loop runs. `build_training_graph`
traces a model in training mode, appends the loss of an `Objective` and differentiates it;
`build_probe_graph` traces a forward-only graph with the buffer updates dropped,
`build_evaluation_graph` traces one in evaluation mode, and `build_classifier` wraps
`build_training_graph` for a cross-entropy objective over [batch, features]. `Objective` is the
interface of a loss, of the fill of its target leaves and of its metrics. This tier implements
it four times: `CrossEntropyObjective`, `BinaryCrossEntropyObjective`, `MultiLabelObjective`
and `MeanSquaredErrorObjective`. `SumObjective` is the composite: it owns other objectives and
adds their weighted losses. An `ObjectiveRegistry` holds the types a training file may
name. An objective declares its graph leaves and the
batch roles it reads in `ObjectiveLeaves`; `match_roles` checks those roles against the ones a
dataset publishes, and `match_input` checks the per-sample shape and the element type of the
input leaf against the input the dataset states. Each objective writes its own leaves in `feed`, on the host, once per batch:
cross-entropy reads `labels` and calls `fill_one_hot`. `TrainingStep` and `EvaluationStep`
compile a graph into a replayable step, and `BufferUpdateRecorder` records the
running-statistic updates behind a training step. `Trainer` copies the inputs of a batch, has the
objective feed its leaves, runs the step, reads the loss and asks the objective for its
`Metrics`; its `fit` runs `options.epochs` passes over a dataset
and returns a `RunReport`. `fit` reads every batch into one `Data::Batch` through
`Data::Dataset::read` and returns the failure of a read as its own. `fit` begins at epoch
index 0 and batch 0 on every call; a caller that resumes a run drives `run_step` itself.

`Trainer::restore_run_count(count)` and `EvaluationStep::restore_run_count(count)` forward to
`Graph::CompiledStep::restore_run_count`, which a resumed run calls once before its first step
with the count the saved run reached. `Trainer::step().run_count()` returns that count plus
the steps since. `Trainer::step_count()` counts the steps of this trainer alone and starts at
0 in a resumed run. `TrainingStep::restore_run_count(count)` restores every part of its step.

## A step that stops for the host

`TrainingStepOptions::host_stage` names the values the host reads and the leaves it writes
inside one step. `TrainingStep::make` then compiles two parts over the one graph and the one
pool: the nodes up to the producer of the last value read, and every other node with the
optimizer and the buffer updates behind them. It refuses, as `InvalidDescription`, a value
read that the graph lacks or that no node produces, and a node of the first part that reads a
leaf the stage writes.

`run(host_work)` replays the first part, calls `host_work()`, and replays the rest; a failure
of the work ends the run before the rest. `run()` replays both parts with nothing between.
`has_host_stage()` tells the two kinds of step apart, `lead()` returns the first part or null,
and `step()` the part that holds the trailing work. A step with no host stage refuses
`run(host_work)` and does not call the work. The values read must be pinned in the memory
plan, as the outputs of a training graph are, or a later node of the first part may reuse
their bytes.

## Depends on

From `CMakeLists.txt`: `PUBLIC_DEPENDS config core data graph nn optim` and
`PRIVATE_DEPENDS autograd`. The module's headers use these:

- `failure`, `dtype`, `shape`, `seam`, `execution`: `Status`, `Result`, `Shape` and `DType`.
- `graph`: the IR, the builder, `TensorPool`, `CompiledStep`, `StepRecorder` and the cast and
  fusion reports.
- `nn`: `NN::Trace`, `NN::ModelState`, `NN::ModelOutputs` and `trace_layer`.
- `optim`: `Optim::Optimizer` and `Optim::ParameterSlot`.
- `data`: `Data::Dataset`, `Data::Batch`, `Data::TargetDescription` and the role names.
- `config`: `Config::Value`, the attributes an objective factory receives.

`autograd` is linked privately. Only `source/steps.cpp` includes it, through
`autograd/backward.hpp`; no header under `include/` does, so a consumer does not link it. The
headers also include `SushiBLAS/engine/engine.hpp`, `SushiBLAS/tensor/tensor.hpp`,
`SushiRuntime/core/span.hpp` and `SushiRuntime/sushiruntime.hpp`, which reach the module through
`failure`, `dtype`, `shape`, `seam` as external public
dependencies.

## Public entry points

The headers sit in `modules/layers/train/include/SushiAI/train/`.

| Header | Declares |
| --- | --- |
| `objective.hpp` | `ObjectiveLeaves`, `find_leaf_tensor`, `Objective`, `CrossEntropyObjective`, `make_cross_entropy` |
| `metrics.hpp` | `Metric`, `Metrics`, `K_METRIC_CAPACITY`, `describe_metric_value` |
| `step_view.hpp` | `StepOutput`, `StepView`, `only_output` |
| `batch_role.hpp` | `require_batch_role` |
| `host_values.hpp` | `require_host_reals`, `host_real` |
| `binary_cross_entropy_objective.hpp` | `BinaryCrossEntropyObjective`, `make_binary_cross_entropy`, `K_BINARY_CROSS_ENTROPY_TYPE` |
| `multi_label_objective.hpp` | `MultiLabelObjective`, `make_multi_label`, `K_MULTI_LABEL_TYPE` |
| `mean_squared_error_objective.hpp` | `MeanSquaredErrorObjective`, `make_mean_squared_error`, `K_MEAN_SQUARED_ERROR_TYPE` |
| `dense_losses.hpp` | `emit_mean_of_elements`, `emit_logistic_loss`, `emit_squared_error` |
| `dense_targets.hpp` | `fill_dense_targets`, `fill_binary_targets` |
| `binary_confusion.hpp` | `BinaryConfusion`, `ColumnOfLogits`, `count_confusion`, `accuracy_of`, `precision_of`, `recall_of`, `f1_of` |
| `decision_threshold.hpp` | `K_THRESHOLD_KEY`, `K_DEFAULT_THRESHOLD`, `validate_threshold`, `read_threshold`, `logit_of` |
| `objective_context.hpp` | `OutputShape`, `ObjectiveContext`, `list_output_shapes`, `find_output_shape` |
| `objective_registry.hpp` | `ObjectiveFactory`, `ObjectiveEntry`, `ObjectiveRegistry`, `register_objectives`, `make_objective` |
| `sum_objective.hpp` | `K_SUM_TYPE`, `K_SUM_TERMS_KEY`, `K_SUM_TERM_TYPE_KEY`, `K_SUM_TERM_WEIGHT_KEY`, `SumTerm`, `SumObjective`, `make_sum_objective` |
| `training_graph.hpp` | `StepSpec`, `ParameterBinding`, `TrainingGraph` |
| `steps.hpp` | `K_MODEL_INPUT_NAME`, `GraphOptions`, `TracedStep`, `batch_shape`, `retain`, `fill_host_tables`, `build_training_graph`, `build_probe_graph`, `build_evaluation_graph`, `build_carrying_graph`, `build_incremental_graph`, `build_classifier` |
| `buffer_update_recorder.hpp` | `BufferUpdateSlot`, `bind_buffer_updates`, `BufferUpdateRecorder` |
| `training_step.hpp` | `OptimizerPlacement`, `HostStage`, `TrainingStepOptions`, `TrainingStep` |
| `evaluation_step.hpp` | `NamedFeed`, `FeedSpan`, `EvaluationStep` |
| `feed.hpp` | `check_feed` |
| `input_match.hpp` | `describe_input_leaf`, `match_input` |
| `one_hot.hpp` | `fill_one_hot` |
| `parameter_groups.hpp` | `takes_weight_decay` |
| `role_match.hpp` | `match_roles` |
| `trainer.hpp` | `bind_parameters`, `bind_step_results`, `TrainerOptions`, `StepMetrics`, `RunReport`, `Trainer` |

## Where a dataset meets a step

`GraphOptions::dtype` names the element type of the input leaf of a step.
`GraphOptions::seed` seeds the random sites of a Training trace, which today are the dropout
layers; `build_training_graph` and `build_probe_graph` pass it to `NN::Trace::training`, and
it is 0 when the caller states none.
`describe_input_leaf(graph.value(input))` returns the leaf's extents past the batch axis
and that type as a `Data::InputDescription`, and `match_input(required, dataset)` compares it
with `Data::describe_input(dataset)`. A difference is `Incompatible` and names the dataset
and both sides, for example "publishes its input as int32 [16] per sample, but the step was
compiled for float32 [16] per sample". `Trainer::make` describes its input leaf and
`Trainer::fit` calls `match_input` before `match_roles`, so a dataset of another type is
refused before a batch is read. A loop that drives a `TrainingStep` or an `EvaluationStep`
itself calls the two functions the same way. `check_feed` still compares every batch tensor
with its leaf when a step runs.

## A step that carries state between runs

`build_carrying_graph(model, batch_shape)` traces a model in Evaluation mode through
`NN::Trace::carrying`, and `build_incremental_graph(model, slice_shape)` traces
`model.trace_incremental` the same way. Both return a `TracedStep` whose `fed` lists the
leaves a layer declared beside the model input and whose `carried` lists the values the step
keeps. `build_probe_graph` and `build_evaluation_graph` leave both lists empty.

`EvaluationStep::make` reads the two lists. It names the model input `K_MODEL_INPUT_NAME`,
"inputs", and the fed leaves after it in the order they were declared. For each carried value
it pins the next value, so the planner keeps it past its last reader, and hands a
`BufferUpdateSlot` of the leaf, the next value, momentum 1 and scale 1 to the
`BufferUpdateRecorder` it compiles behind the graph. That is a scale by zero and an axpy over
the leaf on every run, the work a running statistic costs in a training step.

| Call | Does |
| --- | --- |
| `run(inputs)` | Feeds the model input; `Incompatible` for a step that is fed further leaves |
| `run_in_order(feeds)` | Feeds one tensor per entry of `input_names()`, in that order |
| `run_by_name(feeds)` | Feeds each tensor into the leaf its `NamedFeed` names; a name that is no leaf, a name given twice and a leaf left out are `InvalidDescription` |
| `reset()` | Fills every carried leaf with its initial value; `make` calls it once |
| `carries_state()`, `input_names()`, `input(name)` | Say what the step is fed and keeps |

A step whose lists are empty compiles with no trailing recorder and feeds one leaf, which is
what it did before the lists existed. `make` refuses a fed leaf named "inputs", a carried
value whose next value shares the storage of its leaf, and a pool whose plan packs a next
value.

`tests/layers/train/test_steps.cpp` runs a model that sums `x * gain` across runs.

## Host tables

Every graph builder of `steps.hpp` names the `tables` list of its result as the sink of
`NN::Trace::host_table`, so `TrainingGraph::tables` and `TracedStep::tables` hold the leaves
a layer filled on the host with their values. `fill_host_tables(tables, pool)` copies each
table into its leaf, converting to float32 where the leaf is float32. `bind_parameters`
and `EvaluationStep::make` call it once, after `materialise_leaves` and before the step is
compiled, so a run never writes a table. It returns `Incompatible` for a leaf that is not
float32 or float64 or that holds another count of elements.

## Parameter groups

`takes_weight_decay(role)` is the one statement of which parameters an optimizer decays: a
`Weight` does; a `Bias`, a `Scale` and an `Embedding` do not. `build_training_graph` copies
each parameter's role into its `ParameterBinding`, and `bind_parameters` sets
`Optim::ParameterSlot::decays` from it. A binding built by hand has the role `Weight`. The
optimizer option `decay_all` decays every slot whatever the flag says.

`bind_step_results(built, state, pool)` binds what a step produces and plans no memory: the
state's storage, a zeroed gradient per parameter, the loss and the sources of the buffer
updates, each as its own tensor. It returns the slots as `bind_parameters` does, and stops the
process for a pool that holds a plan. A caller that reads those results and never runs the step,
such as the master of a cluster run, holds no leaf and no intermediate value.
`tests/layers/train/test_steps.cpp` covers the rule and the roles of a traced model.

## The objective owns its feed and its metrics

`Trainer` names no role and no task. `Trainer::make(engine, built, pool, objective, optimizer)`
keeps the leaves and the outputs of the `TrainingGraph` and a reference to the objective.
`run_step` then does five things: `check_feed` on the inputs, `objective.feed(batch, leaves,
pool)`, a copy of the inputs, the step, and `objective.measure(view, metrics)` unless
`TrainerOptions::track_metrics` is off. `fit` matches the dataset against
`leaves.required()`, the roles the objective declared.

| Virtual | Default | Overridden by |
| --- | --- | --- |
| `feed(batch, leaves, pool)` | `Unsupported`: the objective is fed by its own step | Every objective of this tier; `pixel_cross_entropy` and `dice` of `vision/training`; `next_token` of `language/training` |
| `is_fed_by_batch()` | `false` | Every objective that overrides `feed`, with `true` |
| `measure(step, metrics)` | Success, no metric | The same objectives |

`is_fed_by_batch` states ahead of a step what the default `feed` states by refusing: whether a
batch alone fills the leaves. `SumObjective::make` reads it, so an objective that overrides
`feed` and not this is refused as a term.

`feed` looks its leaf up with `find_leaf_tensor` and its role with `require_batch_role`, so a
missing leaf or role is a `Status` that names it. `measure` reads a `StepView`: the batch, the
loss the step computed and the tensor of every output the model published. `Metrics` holds up
to eight named numbers in the order they were added; `RunReport::last_metrics` holds those of
the last step and `RunReport::to_string` prints one line each.

The anchor objective of `vision/training` keeps both defaults. Its targets are deltas a host
pass computes between a forward-only run and the training run, so no function of the batch
alone fills them; `AnchorTrainingStep` and `run_anchor_training` drive it.

A caller that traced with `build_classifier` makes the trainer with a
`CrossEntropyObjective(batch, classes, dtype)` of the same three values, which is the objective
`build_classifier` traced. `RunReport::last_metrics.value_or("accuracy", 0.0)` reads the
accuracy of the last step.

## The composite objective

`SumObjective` owns a list of `SumTerm`, an objective and a weight each, and is itself an
`Objective`, so `Trainer` and `build_training_graph` take it as any other.

| Call | What the composite does |
| --- | --- |
| `declare` | Has term `i` declare into its own `ObjectiveLeaves`, then records each leaf as `term<i>.<role>` and each required role as it is |
| `loss` | Gives each term the leaves under its prefix, by the term's own role names; views each loss as `[1]`, emits `SCALE` by the weight unless it is exactly 1, and `ADD`s them in order |
| `feed` | Calls `feed` of each term, in order, with the leaves under its prefix; the first refusal ends the call |
| `measure` | Calls `measure` of each term, in order; of two metrics of one name the first term's is kept and the other dropped |
| `is_fed_by_batch` | `true` |

`SumObjective::make` refuses an empty list, a weight that is not finite and above zero, and a
term whose `is_fed_by_batch()` is false, with the term's position and name: the anchor objective
is fed by `AnchorTrainingStep` between two runs, and no `feed` of a composite can do that.

A training file states it as

```json
{ "type": "sum", "terms": [ { "type": "pixel_cross_entropy", "weight": 1.0 },
                            { "type": "dice", "weight": 0.5, "ignore_index": 255 } ] }
```

`make_sum_objective` reads `terms` alone. A term states `type`, may state `weight`, 1 when
absent, and hands every other key to the factory of its type, built by `make_objective` over
`ObjectiveContext::registry` with the same context. A term is named `<context> 'terms' entry
<i>` in a refusal. A term may be a `sum`; its leaves then carry both prefixes.

Costs. `loss` and `feed` build one `ObjectiveLeaves` per term per call, a short vector of
names. `measure` runs every term's `measure` before it drops a repeated name, so two terms
that compute one metric compute it twice: `pixel_cross_entropy` and `dice` both read every
score for `mean_iou`. The merged set holds at most `K_METRIC_CAPACITY` metrics.
`build_training_graph` sets `StepSpec::targets` from a leaf named `targets`; a composite has
none under that name, so the field stays unset, as it does for `dice`.

`tests/layers/train/test_sum_objective.cpp` covers the terms, the prefixes, the emitted nodes, the merge and
the factory. `tests/vision/training/test_pixel_cross_entropy_objective.cpp` runs a step over
`pixel_cross_entropy` plus half of `dice` and compares the loss and its gradient with the host.

## The objectives over one [N, columns] output

| Type | Output | Role | Loss | Metrics |
| --- | --- | --- | --- | --- |
| `binary_cross_entropy` | `[N, 1]` logits | `labels`, float32 `[N]`, 0 or 1 | `emit_logistic_loss` | `accuracy`, `precision`, `recall` |
| `multi_label` | `[N, C]` logits | `labels`, float32 `[N, C]`, 0 or 1 | `emit_logistic_loss` | `f1`, the mean of each class's F1 |
| `mean_squared_error` | `[N, K]` values | `values`, float32 `[N, K]` | `emit_squared_error` | `mean_absolute_error` |

`emit_logistic_loss` emits `max(x, 0) - x * y + log(1 + exp(-|x|))` per element as `RELU`,
`MUL`, `ABS`, `SCALE` by -1, `SOFTPLUS`, `SUB` and `ADD`, then the mean. The exponential reads
a value at or below zero, so a logit of any magnitude gives a finite loss. Every operation it
uses had a backward rule and a lowering already. `emit_mean_of_elements` reshapes to
`[elements, 1]`, sums the column with `SUM_ROWS` and scales by `1 / elements`.

The two logit objectives read one optional key, `threshold`, a probability above 0 and below
1 that defaults to 0.5. A logit reads as 1 where it is at least `logit_of(threshold)`, which
is where its sigmoid is at least the threshold. A ratio with a zero denominator is 0: the
precision of a batch with no predicted 1, the recall of a batch with no true 1, and the F1 of
a class with neither. `fill_binary_targets` refuses a label that is neither 0 nor 1 and leaves
the leaf as it was.

Each factory takes its extents from the one output of `ObjectiveContext::outputs` and
refuses a model that publishes another count of outputs, another rank or another batch. The
`loss` of each stops the process when the model's output is not the shape the objective was
made for; a factory call has already compared the two.

## The objective registry

An `ObjectiveRegistry` holds rows of a type and a factory, as `Data::DatasetRegistry` and
`NN::NodeRegistry` do. `register_objectives` adds `cross_entropy`, `binary_cross_entropy`,
`multi_label`, `mean_squared_error` and `sum`; a domain adds its own rows
through its registration module, so this tier names no objective of a domain.
`make_objective(registry, type, given, attributes, context)` calls the row's factory and returns
its `Result`; an unknown type is `InvalidDescription` with every type the registry holds.

Every factory has one signature and receives an `ObjectiveContext` beside its own section:

| Field | Holds | Read by |
| --- | --- | --- |
| `batch` | Samples per step | Every objective of this tier, `anchor` |
| `classes` | Classes of the data, 0 when it states none | `cross_entropy`; `anchor` compares it with the task |
| `input_sample` | Shape of one sample entering the model | `anchor` |
| `task` | The `task` object of the model file, null for none | `anchor` |
| `outputs` | Name and extents of each published output, batch first | `binary_cross_entropy`, `multi_label`, `mean_squared_error`, `anchor` |
| `registry` | The registry the factory's row was found in; `make_objective` sets it on a copy | `sum` |

`list_output_shapes(graph, outputs)` fills `outputs` from any trace of the model.
`build_probe_graph` needs no engine, so the shapes are known before an objective is made and
before a tensor is allocated.

`tests/layers/train/test_objective_registry.cpp` and `tests/layers/train/test_objective_context.cpp` cover the registry,
the context a factory receives and the shapes of a probe trace. `tests/layers/train/test_metrics.cpp`,
`test_dense_losses.cpp`, `test_dense_targets.cpp`, `test_binary_confusion.cpp` and
`test_dense_objectives.cpp` cover the units above, and
`tests/layers/train/test_trainer/test_dense_objectives.cpp` compares each loss and metric with a host
computation over the outputs of a step.
