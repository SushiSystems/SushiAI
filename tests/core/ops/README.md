# tests/core/ops

The tests of the module `core/ops`, one file per header of `modules/core/ops/include/`. The suite
prefix of a case gives its ctest label.

## test_ops.cpp

Suites: `Small_Ops`, `Small_OpsEntry`, `Small_OpsDeclaredName`, `Small_OpsDeclaredIdentity`,
`Small_OpsDifferentiableFlag`, `Small_OpsGradientBarrier`, `Small_OpsGradientPasser`,
`Small_OpsInputActivation` and `Small_OpsActivationRow`.

Tests that the operation table is complete and that the reverse lookup resolves names.

This file needs no device. The uniqueness proof already runs as a static_assert inside
`modules/core/ops/include/SushiAI/ops/ops.hpp`. These tests cover what that assert cannot: that the
table is complete, and that the reverse lookup behaves. A test with a suite of its own runs once per
row, and the row is named after the operation it holds.

### EveryIdentityHashesDistinctly

No two entries of `Ops::K_ALL` have the same `OpID` value.

### EveryNameCarriesTheLibraryPrefix

Runs once per row of `Ops::K_ALL`.

The prefix keeps a SushiAI identity from colliding with SushiBLAS's own (blas.lvl3.gemm and
friends) in the profiler's report.

The test calls `rfind(prefix, 0)` and not `starts_with()`, which is C++20. `rfind` searches
backwards from position 0, so it can only match at position 0, which is the same question asked
with the vocabulary C++17 has.

### NameOfRoundTripsEveryIdentity

Runs once per row of `Ops::K_ALL`. `Ops::name_of` returns the row's name for its identity.

### IsGradientBarrierEqualsTheRowFlag

Runs once per row of `Ops::K_ALL`. `Ops::is_gradient_barrier` equals the row's `blocks_gradient`
flag.

### NoGradientBarrierIsFlaggedDifferentiable

Runs once per row of `Ops::K_ALL`. The row does not have both `differentiable` and
`blocks_gradient` set.

### NameOfReportsUnknownForAForeignIdentity

`Ops::name_of(OpID(0xDEADBEEF))` is "sushiai.unknown".

### TheTableHoldsAsManyRowsAsDeclaredNames

`Ops::K_ALL` has as many rows as the test's list of declared names. A new operation left out of the
list fails the count.

### TheTableContainsEveryDeclaredIdentity

Runs once per declared name. A new operation declared but left out of K_ALL would escape the
uniqueness static_assert entirely, so the test checks membership by name.

### EveryDeclaredIdentityIsListedInAll

Runs once per identity of the spatial, batch-norm, leaky ReLU, elementwise, selection and SiLU or
Softplus groups. K_ALL is what `all_ids_unique()` proves over and what `name_of()` searches. An
identity declared but left out of K_ALL is not covered by the collision proof and prints as
"sushiai.unknown" in every profile. Both failures are silent.

### ThereIsNoConv2dIdentity

CV-1 says convolution is K_IM2COL + K_MATMUL in the IR. A CONV2D identity would be a second, opaque
spelling of the same thing, and the first person to add one will do it by analogy and not by
reading the spec.

### LeakyReLUDifferentiatesAtItsOutput

For alpha > 0, leaky_relu(x) > 0 exactly when x > 0. Reading the output in place of the input is
therefore a bit-identical substitution, not an approximation, and it is the same argument the
table already records for K_RELU. It is what makes the fold free: an ordinary use count licenses
it with no activation-specific knowledge in the fusion pass.

### AnIdentityCarriesItsDifferentiableFlag

Runs once per identity. `differentiable` is true for K_DIV, K_MINIMUM, K_MAXIMUM, K_ABS, K_ATAN,
K_SOFTPLUS, K_SELECT and K_SILU, and false for K_LESS_EQUAL, K_SILU_BACKWARD and
K_SOFTPLUS_BACKWARD.

### ABarrierIdentityIsAGradientBarrier

Runs once for each of K_ZERO, K_CONSTANT, K_LESS_EQUAL and K_STOP_GRADIENT.
`Ops::is_gradient_barrier` is true.

### ExactlyFourIdentitiesBlockTheGradient

Exactly four rows of `Ops::K_ALL` carry the `blocks_gradient` flag. A fifth barrier fails the count
until the test's list of barriers names it.

### AGradientPasserIsNotAGradientBarrier

Runs once for each of K_LEAF, K_CAST, K_DYNAMIC_SCALE, K_LOG_SOFTMAX, K_LOG_SOFTMAX_BACKWARD,
K_RELU_BACKWARD, K_SILU_BACKWARD, K_SOFTPLUS_BACKWARD, K_MAXPOOL2D_BACKWARD,
K_BATCHNORM2D_BACKWARD, K_BATCHNORM_UPDATE_STATS, K_BATCHNORM_FOLD, K_GEMM_BIAS, K_EXP_SUB_SCALE
and K_ADAMW_UPDATE. `is_gradient_barrier` is false. These either pass a gradient or throw when one
reaches them. Marking one as a barrier would turn that throw into a gradient that is dropped
without a message.

### AForeignIdentityIsNotAGradientBarrier

`is_gradient_barrier(OpID(0xDEADBEEF))` is false.

### AnActivationDifferentiatesAtItsInputAndDoesNotSurviveFusion

Runs once for K_SILU and once for K_SOFTPLUS. The `Ops::K_ACTIVATIONS` row names the forward
identity's backward one and `GradientSource::Input` as its source, and
`activation_survives_fusion` is false.

### IsActivationBackwardAcceptsTheGradientAndNotTheForward

Runs once per row of `Ops::K_ACTIVATIONS`. `is_activation_backward` is true for the row's
`backward` identity and false for its `forward` one.

### IsActivationBackwardAcceptsAsManyIdentitiesAsTheTableHasRows

Over `Ops::K_ALL`, `is_activation_backward` accepts as many identities as the table has rows.

### BatchNormFoldIsNeitherDifferentiableNorAGradientBarrier

The row of K_BATCHNORM_FOLD has `differentiable` false and `blocks_gradient` false.
`test_backward.cpp` holds what differentiation does when a gradient reaches the node.
