# ops

## Owns

The compile-time operation identities in `SushiAI::Ops`: one identity per operation the graph can
hold, and the tables that say which of them differentiate, which pass no gradient and which are
activations.

## Depends on

The module declares no module dependency. It links `SushiRuntime::core` as public, for
`OperationID`.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/ops/ops.hpp` | `OpName`, `op`, `differentiable_op`, `gradient_barrier_op`, the identities such as `K_LEAF`, `K_MATMUL` and `K_BATCHNORM2D`, `K_ALL`, `GradientSource`, `Activation`, `K_ACTIVATIONS`, `activation_of` |
