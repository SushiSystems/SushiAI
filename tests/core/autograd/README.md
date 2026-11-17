# tests/core/autograd

The tests of the module `core/autograd`, one file per header of `modules/core/autograd/include/`. The suite prefix of a case gives its ctest label.

## Test files

Each test file has a page beside it that says what its cases pin.

| Test file | Page |
| --- | --- |
| `test_gradcheck.cpp` | [`test_gradcheck.md`](test_gradcheck.md) |
| `test_provenance.cpp` | [`test_provenance.md`](test_provenance.md) |
| `rules/test_linalg/test_rules.cpp` | [`rules/test_linalg/test_rules.md`](rules/test_linalg/test_rules.md) |
| `rules/test_linalg/test_gradients.cpp` | [`rules/test_linalg/test_gradients.md`](rules/test_linalg/test_gradients.md) |
| `rules/test_random.cpp` | [`rules/test_random.md`](rules/test_random.md) |
| `rules/test_norm/test_rules.cpp` | [`rules/test_norm/test_rules.md`](rules/test_norm/test_rules.md) |
| `rules/test_norm/test_gradients.cpp` | [`rules/test_norm/test_gradients.md`](rules/test_norm/test_gradients.md) |
| `rules/test_all.cpp` | [`rules/test_all.md`](rules/test_all.md) |
| `rules/test_activation/test_rules.cpp` | [`rules/test_activation/test_rules.md`](rules/test_activation/test_rules.md) |
| `rules/test_activation/test_gradients.cpp` | [`rules/test_activation/test_gradients.md`](rules/test_activation/test_gradients.md) |
| `rules/test_elementwise/test_rules.cpp` | [`rules/test_elementwise/test_rules.md`](rules/test_elementwise/test_rules.md) |
| `rules/test_elementwise/test_gradients.cpp` | [`rules/test_elementwise/test_gradients.md`](rules/test_elementwise/test_gradients.md) |
| `rules/test_loss/test_rules.cpp` | [`rules/test_loss/test_rules.md`](rules/test_loss/test_rules.md) |
| `rules/test_loss/test_gradients.cpp` | [`rules/test_loss/test_gradients.md`](rules/test_loss/test_gradients.md) |
| `rules/test_shape/test_rules.cpp` | [`rules/test_shape/test_rules.md`](rules/test_shape/test_rules.md) |
| `rules/test_shape/test_gradients.cpp` | [`rules/test_shape/test_gradients.md`](rules/test_shape/test_gradients.md) |
| `rules/test_broadcast.cpp` | [`rules/test_broadcast.md`](rules/test_broadcast.md) |
| `rules/test_spatial/test_rules.cpp` | [`rules/test_spatial/test_rules.md`](rules/test_spatial/test_rules.md) |
| `rules/test_spatial/test_gradients.cpp` | [`rules/test_spatial/test_gradients.md`](rules/test_spatial/test_gradients.md) |
| `test_backward.cpp` | [`test_backward.md`](test_backward.md) |

## The gradient check

The whole graph (forward, loss and backward) runs on the device in double precision. The analytic
gradient it produces is compared against a central difference of the loss the same graph
computes. No other check separates a correct backward rule from a plausible one. It has to be
fp64: an fp32 difference quotient bottoms out near 2.4e-5 relative accuracy, the same order as the
error a wrong rule produces. See `gradcheck.hpp`.

`GradientProbe` in `tests/support/gradient_probe.hpp` builds the graph. The variable under test is
declared as the only gradient-requiring leaf. Everything else the build callback declares is a
constant the test fills. The loss is the mean negative log-likelihood of what the callback
returns against a target matrix, which is a valid scalar functional of anything shaped [N, C].
When the model ends in a log-softmax it exercises the fused cross-entropy rule as well.

Fusion is off by default, because almost every case checks a backward rule and fusion must not
change one. `GradientProbeOptions::fused()` switches it on because one case checks the opposite
direction: a fused form that rewrites what the backward pass reads must leave the same derivative
behind. Value ids survive the fusion pass, so nothing the probe recorded before it needs
remapping.
