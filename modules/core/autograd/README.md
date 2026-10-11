# autograd

## Owns

Reverse-mode differentiation as a pass from one IR graph to another. `differentiate` appends the
gradient nodes of a loss to a graph and returns one gradient id per requested value.
`Provenance` records which forward node each appended node was emitted for. A backward rule
reads and writes through a `GradientTape`. Rules are declared per domain as `RuleEntry` tables,
and `rules/all.hpp` joins them into `K_RULES`. `gradcheck` compares analytic gradients with
finite differences.

## Depends on

The module declares `core` and `graph` as public dependencies. It declares no private
dependency.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/autograd/backward.hpp` | `has_backward_rule`, `differentiate` |
| `SushiAI/autograd/gradcheck.hpp` | `GradCheckOptions`, `GradCheckResult`, `ScalarFunction`, `GradientFunction`, `push_off_kinks`, `check_full_jacobian`, `check_directional` |
| `SushiAI/autograd/provenance.hpp` | `Provenance`, with its `origin_of` |
| `SushiAI/autograd/rule.hpp` | `BackwardRule`, `RuleEntry`, `concat_rules` |
| `SushiAI/autograd/tape.hpp` | `GradientTape` |
| `SushiAI/autograd/rules/activation.hpp` | `activation_backward`, `K_ACTIVATION_RULES` |
| `SushiAI/autograd/rules/all.hpp` | `K_RULES`, `has_rule`, `every_differentiable_op_has_a_rule`, `no_gradient_barrier_has_a_rule` |
| `SushiAI/autograd/rules/elementwise.hpp` | `add_backward`, `sub_backward`, `mul_backward`, `scale_backward`, `exp_backward`, `accumulate_backward`, `div_backward`, `minimum_backward`, `maximum_backward`, `abs_backward`, `atan_backward`, `select_backward`, `K_ELEMENTWISE_RULES` |
| `SushiAI/autograd/rules/linalg.hpp` | `matmul_backward`, `bias_add_backward`, `sum_rows_backward`, `gather_rows_backward`, `scatter_add_rows_backward`, `causal_mask_backward`, `K_LINALG_RULES` |
| `SushiAI/autograd/rules/loss.hpp` | `log_softmax_backward`, `nll_loss_backward`, `starts_from_implicit_seed`, `K_LOSS_RULES` |
| `SushiAI/autograd/rules/norm.hpp` | `batchnorm2d_backward`, `channel_affine_backward`, `layer_norm_backward`, `K_NORM_RULES` |
| `SushiAI/autograd/rules/random.hpp` | `dropout_backward`, `K_RANDOM_RULES` |
| `SushiAI/autograd/rules/shape.hpp` | `reshape_backward`, `layout_convert_backward`, `K_SHAPE_RULES` |
| `SushiAI/autograd/rules/spatial.hpp` | `im2col_backward`, `col2im_backward`, `maxpool2d_backward`, `avgpool2d_backward`, `upsample2d_backward`, `spatial_mean_backward`, `concat_channels_backward`, `slice_channels_backward`, `K_SPATIAL_RULES` |

## Log-softmax and the fused cross-entropy

`nll_loss_backward` absorbs a `K_LOG_SOFTMAX` that produces its first operand: it skips that node
and contributes `(exp(log_probs) - targets) / N` to the logits. That closed form is the
derivative only when each target row sums to one. Every other `K_LOG_SOFTMAX` is differentiated
by `log_softmax_backward`, a row of `K_LOSS_RULES`, which emits `K_LOG_SOFTMAX_BACKWARD` over the
incoming gradient and the forward result: `dx = dy - exp(log_probs) * rowsum(dy)`.
