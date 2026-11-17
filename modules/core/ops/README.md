# ops

## Owns

The compile-time operation identities in `SushiAI::Ops`: one identity per operation the graph can
hold, and the tables that say which of them differentiate, which pass no gradient and which are
activations.

## Depends on

The module declares no module dependency. It links `SushiRuntime::core` as public, for
`OperationID`.

## Operations added by the graph capabilities

`K_SQRT`, `K_LOG` and `K_POW` are differentiable elementwise math; the exponent of the power is
attribute slot 0. `K_UPSAMPLE2D_BILINEAR` is differentiable and its gradient,
`K_UPSAMPLE2D_BILINEAR_BACKWARD`, is not, as for the nearest pair. `K_TAKE_ALONG` and
`K_PUT_ALONG` are differentiable and each is the other's gradient. `K_BROADCAST_ADD`, `SUB`,
`MUL` and `DIV` and `K_SUM_ALONG` are differentiable and carry their side in attribute flag 0.
`K_TRANSPOSE` is differentiable, copies a value with its non-batch axes permuted, and holds the
permutation in attribute slots 0 to 4; its gradient is the same op with the inverse permutation.
`K_ARGMAX_ROWS` is a gradient barrier, declared through `gradient_barrier_op`; it has no rule, and
`no_gradient_barrier_has_a_rule` fails the build if one is added. `K_LAYER_NORM_DEVIATION` and
`K_LAYER_NORM_APPLY` name the two sub-tasks of the layer-norm row; no node carries them, so they
are listed in `K_UNLOWERED`. Every identity above is a row of `K_ALL`.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/ops/ops.hpp` | `OpName`, `op`, `differentiable_op`, `gradient_barrier_op`, the identities such as `K_LEAF`, `K_MATMUL`, `K_BATCHNORM2D`, `K_SQRT`, `K_LOG`, `K_POW`, `K_UPSAMPLE2D_BILINEAR`, `K_TAKE_ALONG`, `K_PUT_ALONG`, `K_ARGMAX_ROWS`, the four `K_BROADCAST_*`, `K_SUM_ALONG` and `K_TRANSPOSE`, `K_ALL`, `GradientSource`, `Activation`, `K_ACTIVATIONS`, `activation_of` |
