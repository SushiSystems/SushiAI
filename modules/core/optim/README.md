# optim

## Owns

The optimizers and the state they keep across a training step. `Optimizer` is the abstract base.
It holds the `ParameterSlot`s of a model and is a `Graph::StepRecorder`, so it records its
update into a step. `SGD`, with momentum, and `AdamW`, with decoupled weight decay, are the two
implementations. `StepScalars` holds the scalars that the update kernels read from device
memory. `LossScaler` applies dynamic loss scaling for mixed precision. `optimizer_registry` maps
a configured type name to a factory through `K_OPTIMIZER_REGISTRY`.

## Depends on

The module declares `core`, `graph` and `config` as public dependencies. It declares no private
dependency.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/optim/adamw.hpp` | `AdamWOptions`, `AdamW` |
| `SushiAI/optim/loss_scaler.hpp` | `LossScaleOptions`, `LossScaleState`, `LossScaler` |
| `SushiAI/optim/optimizer.hpp` | `UpdateKernel`, `ParameterSlot`, `Optimizer` |
| `SushiAI/optim/optimizer_registry.hpp` | `OptimizerFactory`, `OptimizerEntry`, `K_OPTIMIZER_REGISTRY`, `make_adamw`, `make_sgd`, `make_optimizer` |
| `SushiAI/optim/sgd.hpp` | `SGDOptions`, `SGD` |
| `SushiAI/optim/step_scalars.hpp` | `StepScalars` |
