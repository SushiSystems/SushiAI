# train

## Owns

The train module turns a model into the graphs a training loop runs. `build_training_graph`
traces a model in training mode, appends the loss of an `Objective` and differentiates it;
`build_probe_graph` traces a forward-only graph with the buffer updates dropped,
`build_evaluation_graph` traces one in evaluation mode, and `build_classifier` wraps
`build_training_graph` for a cross-entropy objective over [batch, features]. `Objective` is the
loss interface, `CrossEntropyObjective` is its one implementation, and `K_OBJECTIVE_REGISTRY`
lists the types a configuration file may name. `TrainingStep` and `EvaluationStep` compile a
graph into a replayable step, and `BufferUpdateRecorder` records the running-statistic updates
behind a training step. `Trainer` feeds batches to a training step and reads the loss and the
accuracy, and its `fit` runs `options.epochs` passes over a dataset and returns a `RunReport`.

## Depends on

From `CMakeLists.txt`: `PUBLIC_DEPENDS core graph config data optim nn` and
`PRIVATE_DEPENDS autograd`. The module's headers use these:

- `core`: `Status`, `Result`, `Shape` and `DType`.
- `graph`: the IR, the builder, `TensorPool`, `CompiledStep`, `StepRecorder` and the cast and
  fusion reports.
- `nn`: `NN::Trace`, `NN::ModelState`, `NN::ModelOutputs` and `trace_layer`.
- `optim`: `Optim::Optimizer` and `Optim::ParameterSlot`.
- `data`: `Data::Dataset` and `Data::Batch`.
- `config`: `Config::Value`, the attributes an objective factory receives.

`autograd` is linked privately. Only `source/steps.cpp` includes it, through
`autograd/backward.hpp`; no header under `include/` does, so a consumer does not link it. The
headers also include `SushiBLAS/engine.hpp`, `SushiBLAS/tensor.hpp` and
`SushiRuntime/core/run_report.hpp`, which reach the module through `core` as external public
dependencies.

## Public entry points

The headers sit in `modules/layers/train/include/SushiAI/train/`.

| Header | Declares |
| --- | --- |
| `objective.hpp` | `Metric`, `ObjectiveLeaves`, `Objective`, `CrossEntropyObjective`, `ObjectiveFactory`, `ObjectiveEntry`, `make_cross_entropy`, `K_OBJECTIVE_REGISTRY`, `make_objective` |
| `training_graph.hpp` | `StepSpec`, `ParameterBinding`, `TrainingGraph` |
| `steps.hpp` | `GraphOptions`, `TracedStep`, `batch_shape`, `retain`, `build_training_graph`, `build_probe_graph`, `build_evaluation_graph`, `build_classifier` |
| `buffer_update_recorder.hpp` | `BufferUpdateSlot`, `bind_buffer_updates`, `BufferUpdateRecorder` |
| `training_step.hpp` | `OptimizerPlacement`, `TrainingStepOptions`, `TrainingStep` |
| `evaluation_step.hpp` | `EvaluationStep` |
| `feed.hpp` | `check_feed` |
| `trainer.hpp` | `bind_parameters`, `TrainerOptions`, `StepMetrics`, `RunReport`, `Trainer` |
