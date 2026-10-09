# training

## Owns

The training of a detection model whose loss the graph cannot state as one expression: the loss
root that hands a host-computed delta to autograd, the objective of a model with anchor heads,
and the two-pass step that computes the deltas between a forward-only run and the training
replay. Everything is in namespace `SushiAI::Vision::Training`. The behaviour is in sections 4.6
to 4.8 of `docs/agent/2026_10_09_DARKNET_YOLO/SPEC.md`.

The delta itself and its reported cost belong to `vision/detect`, part `anchor/`
(`compute_head_delta`, `measure_head_cost`). This module only carries them into a graph.

## Depends on

- `train`, for `Objective`, `EvaluationStep`, `TrainingStep` and the graph builders.
- `vision/detect`, for the anchor layout, the head options, the delta and the cost.
- `graph`, `nn`, `optim` and `core`, as `train` names them.

## Public entry points

| Header | Declares |
| --- | --- |
| `<SushiAI/vision/training/head_delta_loss.hpp>` | `emit_head_delta_loss` |
| `<SushiAI/vision/training/anchor_objective.hpp>` | `AnchorHead`, `AnchorObjective` |
| `<SushiAI/vision/training/anchor_training_step.hpp>` | `AnchorTrainingStep` |

## The delta loss

`emit_head_delta_loss(builder, head, delta, scale)` emits `-scale * sum(delta * head)` as `MUL`,
`RESHAPE`, `SUM_ROWS` and `SCALE`. `delta` is an input leaf, so the gradient at `head` is
`-scale * delta` and the value of the root means nothing.

## The anchor objective

`AnchorObjective::make(network, layout, heads)` takes one `AnchorHead` per head of the layout, in
the layout's order: the name the model publishes the head under, its `HeadShape`, its
`HeadOptions` and its `HeadTrainingOptions`. The shape is part of the description because
`Train::Objective::declare` runs before the model is traced, when no head has a shape yet. Every
head carries the same batch.

- `declare` adds one float32 leaf per head, shaped `[B, H, W, A * (5 + C)]`, under the role
  `delta.<output name>`.
- `loss` adds `emit_head_delta_loss` over the heads with `scale = 1 / batch`. It throws, as the
  objectives of `train` do, when the model does not publish a head or publishes it in another
  shape.
- `measure_cost(deltas)` returns the `HeadCost` of every head summed field by field.
- `score` returns no metric. `Train::Objective::score` receives one output tensor and no delta,
  so the cost of several heads cannot come through it; `measure_cost` reports it instead.

## The training step

`AnchorTrainingStep` owns a `Train::EvaluationStep` over the graph of `build_probe_graph` and a
`Train::TrainingStep` over the graph of `build_training_graph`. Each graph has its own
`TensorPool`; both pools bind the parameters and buffers of one `NN::ModelState`, so the two
passes read the same weights. The caller builds both graphs from one model, binds the training
pool with `Train::bind_parameters`, makes the optimizer, and passes the parts to `make`.

`run(inputs, truth)` does four things in order:

1. It checks the feed and the truth. A truth of another batch or array size is `Incompatible`; a
   valid row whose label names no class of a head is `InvalidDescription`.
2. It runs the forward-only pass. The probe graph is traced in training mode with its buffer
   updates dropped, so the heads come out under batch statistics and no running value moves.
3. It calls `compute_head_delta` on each head and copies the delta into that head's leaf.
4. It copies the inputs into the training graph and replays it: forward, backward, the
   optimizer's update and the running-statistic blend, once.

It returns `measure_cost` of the deltas it fed. `deltas()` returns them, and `evaluation()` gives
the forward-only outputs they were computed on.

The host stage reads the heads and writes the leaves through `Tensor::data_as<float>()`, so both
must be float32 and host-addressable, as the loss and the logits `Train::Trainer` reads are.

## Tests

| File | Covers |
| --- | --- |
| `tests/test_head_delta_loss.cpp` | The four nodes, the root under autograd, the refusals |
| `tests/test_anchor_objective.cpp` | The leaves, the summed root, the summed cost, the refusals |
| `tests/integration/test_head_delta_loss.cpp` (root) | The gradient at the head on an engine |
| `tests/integration/test_anchor_training_step.cpp` (root) | The leaves fed, a bias after one step, a running mean blended once, the refusals |
