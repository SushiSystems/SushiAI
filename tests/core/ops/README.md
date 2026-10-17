# tests/core/ops

The tests of the module `core/ops`, one file per header of `modules/core/ops/include/`. The suite
prefix of a case gives its ctest label.

## test_ops.cpp

Suites: `Small_Ops`.

Tests that the operation table is complete and that the reverse lookup resolves names.

This file needs no device. The uniqueness proof already runs as a static_assert inside
`modules/core/ops/include/SushiAI/ops/ops.hpp`. These tests cover what that assert cannot: that the
table is
complete, and that the reverse lookup behaves.

### EveryIdentityHashesDistinctly

No two entries of `Ops::K_ALL` have the same `OpID` value.

### EveryNameCarriesTheLibraryPrefix

The prefix keeps a SushiAI identity from colliding with SushiBLAS's own (blas.lvl3.gemm and
friends) in the profiler's report.

The test calls `rfind(prefix, 0)` and not `starts_with()`, which is C++20. `rfind` searches
backwards from position 0, so it can only match at position 0, which is the same question asked
with the vocabulary C++17 has.

### NameOfRoundTripsEveryIdentity

`Ops::name_of` returns the name of every entry of `Ops::K_ALL` for its identity.

### NameOfReportsUnknownForAForeignIdentity

`Ops::name_of(OpID(0xDEADBEEF))` is "sushiai.unknown".

### TheTableContainsEveryDeclaredIdentity

A new operation declared but left out of K_ALL would escape the uniqueness static_assert
entirely, so the test checks membership by name.

### EverySpatialIdentityIsListedInAll

K_ALL is what `all_ids_unique()` proves over and what `name_of()` searches. An identity declared
but left out of K_ALL is not covered by the collision proof and prints as "sushiai.unknown" in
every profile. Both failures are silent.

### ThereIsNoConv2dIdentity

CV-1 says convolution is K_IM2COL + K_MATMUL in the IR. A CONV2D identity would be a second, opaque
spelling of the same thing, and the first person to add one will do it by analogy and not by
reading the spec.

### LeakyReLUDifferentiatesAtItsOutput

For alpha > 0, leaky_relu(x) > 0 exactly when x > 0. Reading the output in place of the input is
therefore a bit-identical substitution, not an approximation, and it is the same argument the
table already records for K_RELU. It is what makes the fold free: an ordinary use count licenses
it with no activation-specific knowledge in the fusion pass.

### EveryElementwiseAndSelectionIdentityIsListedInAll

`Ops::name_of` resolves K_DIV, K_MINIMUM, K_MAXIMUM, K_ABS, K_ATAN, K_LESS_EQUAL, K_SELECT, K_SILU,
K_SILU_BACKWARD, K_SOFTPLUS and K_SOFTPLUS_BACKWARD to a name other than "sushiai.unknown". The
reason is the one EverySpatialIdentityIsListedInAll gives.

### TheComparisonAndTheActivationGradientsAreNotFlaggedDifferentiable

`differentiable` is true for K_DIV, K_MINIMUM, K_MAXIMUM, K_ABS, K_ATAN, K_SOFTPLUS, K_SELECT and
K_SILU, and
false for K_LESS_EQUAL, K_SILU_BACKWARD and K_SOFTPLUS_BACKWARD.

### TheGradientBarriersAreTheFourIdentitiesThatPassNoGradient

`Ops::is_gradient_barrier` is true for K_ZERO, K_CONSTANT, K_LESS_EQUAL and K_STOP_GRADIENT. Over
every
row of `Ops::K_ALL` it equals the row's `blocks_gradient` flag, and exactly four rows carry the
flag. A fifth barrier therefore fails the count until the list in this test names it.

### NoGradientBarrierIsFlaggedDifferentiable

No row of `Ops::K_ALL` has both `differentiable` and `blocks_gradient` set.

### LeafCastAndTheGradientOperationsAreNotGradientBarriers

`is_gradient_barrier` is false for K_LEAF, K_CAST, K_DYNAMIC_SCALE, K_LOG_SOFTMAX, K_RELU_BACKWARD,
K_SILU_BACKWARD, K_SOFTPLUS_BACKWARD, K_MAXPOOL2D_BACKWARD, K_BATCHNORM2D_BACKWARD,
K_BATCHNORM_UPDATE_STATS, K_BATCHNORM_FOLD, K_GEMM_BIAS, K_EXP_SUB_SCALE and K_ADAMW_UPDATE, and for
the
foreign identity 0xDEADBEEF. These either pass a gradient or throw when one reaches them.
Marking one as a barrier would turn that throw into a gradient that is dropped without a
message.

### SiLUDifferentiatesAtItsInput

The `Ops::K_ACTIVATIONS` row of K_SILU names K_SILU_BACKWARD as its gradient and
`GradientSource::Input` as its source, and `activation_survives_fusion(K_SILU)` is false.

### SoftplusDifferentiatesAtItsInput

The row of K_SOFTPLUS names K_SOFTPLUS_BACKWARD and `GradientSource::Input`, and
`activation_survives_fusion(K_SOFTPLUS)` is false.

### IsActivationBackwardAcceptsOnlyTheTablesGradientColumn

For every row of `Ops::K_ACTIVATIONS`, `is_activation_backward` is true for the row's `backward`
identity and false for its `forward` one. Over `Ops::K_ALL` it accepts as many identities as the
table has rows.

### BatchNormFoldIsListedAndClaimsNoRule

`Ops::name_of(K_BATCHNORM_FOLD)` is "sushiai.batchnorm_fold", and the row has `differentiable`
false and `blocks_gradient` false. `test_backward.cpp` holds what differentiation does when a
gradient reaches the node.
