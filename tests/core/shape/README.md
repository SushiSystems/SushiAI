# tests/core/shape

The tests of the module `core/shape`, one file per header of `modules/core/shape/include/`. The
suite prefix of a case gives its ctest label.

## test_shape.cpp

Suites: `Small_Shape`.

Tests Shape's extents, indexing, validation, equality and hash.

This file needs no device. Shape is host-lane vocabulary and must stay testable without a
runtime, which is what keeps the unit tier fast.

### DefaultIsAScalarWithOneElement

A default `Shape` has rank 0 and one element, and prints as "[]".

### ExtentsAreStoredOutermostFirst

`Shape{64, 784}` has rank 2, `dim(0)` 64, `dim(1)` 784 and 64 * 784 elements, and prints as
"[64, 784]".

### NegativeIndicesCountFromTheInnermost

On `Shape{2, 3, 4}`, `dim(-1)` is 4, `dim(-2)` is 3 and `dim(-3)` is 2.

### OutOfRangeDimensionIsFatal

On `Shape{2, 3}`, `dim(2)` and `dim(-3)` stop the process with a report naming the index.

### NonPositiveExtentIsFatalAtConstruction

`Shape({2, 0})` and `Shape({-1})` stop the process with a report that says an extent must be
positive.

### RankBeyondTheMaximumIsFatal

A list of nine extents, more than `K_MAX_RANK`, stops the process.

### BuildsFromExtentsKnownOnlyAtRunTime

A `Shape` built from a span holding 64 and 784 equals `Shape{64, 784}` and prints as
"[64, 784]".

### BuildsTheScalarFromAnEmptyExtentSpan

A `Shape` built from an empty span has rank 0 and one element.

### RuntimeExtentsGetTheSameValidationAsAListDoes

A span holding 4 and 0 stops the process, and so does a span of `K_MAX_RANK` + 1 ones.

### MakeBuildsTheShapeOfValidRuntimeExtents

`Shape::make` over a span holding 64 and 784 returns `Shape{64, 784}`.

### MakeBuildsTheScalarFromAnEmptySpan

`Shape::make` over an empty span returns the rank-0 shape.

### MakeRefusesANonPositiveExtentWithAStatus

`Shape::make` over 4, 0 and 2 returns `InvalidDescription` naming "Shape extents must be
positive" and "got 0".

### MakeRefusesARankBeyondTheMaximumWithAStatus

`Shape::make` over `K_MAX_RANK` + 1 ones returns `InvalidDescription` naming "Shape rank 7" and
"exceeds the maximum of 6".

### EqualityComparesRankAndEveryExtent

[4, 5] equals [4, 5] and differs from [5, 4], from [4, 5, 1], and [4] differs from the
scalar.

### HashSeparatesShapesThatDifferOnlyByRank

[4, 1] and [4] hold the same element count and the same leading extent. A hash that folded only
the extents would collide them, and the tensor pool keys its constant operands on this hash.

### HashKeepsItsValues

The hashes of the scalar, [4, 3] and [2, 3, 4] equal values computed in Python over the rank and
each extent as eight bytes low first, from before `Shape::hash` called the `hash` module.

### WorksAsAnUnorderedMapKey

An `unordered_map` keyed by `Shape`, written twice under [2, 2] and once under [4], holds two
entries with the values 2 and 7.

## test_shape.cpp, tests/core/graph/test_builder.cpp, tests/core/graph/test_ir.cpp, tests/layers/nn/test_layers.cpp, tests/core/autograd/test_backward.cpp

Suites: `Small_Diagnostics`.

Tests that each host-lane precondition raises a message carrying the values that disagree.

This file needs no device. It holds every host-lane precondition to the message it promises.

Other tests already prove that these preconditions throw. They do not prove that the messages
still say anything. A refactor that funnels five specific diagnostics into one generic "invalid
argument" leaves every EXPECT_THROW green and destroys the only thing that makes a shape
mismatch debuggable. So each case names the values the message must still carry (the two
extents that disagree, the operation that has no rule, the dtype that cannot be served) and
fails when one of them goes missing.

`modules/core/failure_expectations/include/SushiAI/failure_expectations/status_expectations.hpp`
defines `SA_EXPECT_FAILURE`, which compares the code and the message.

The cases fall into five groups, in file order:

- Shape
- GraphBuilder, the shape-inference boundary
- Graph: single assignment and operand existence
- layers
- differentiation

### ShapeRankBeyondTheMaximumNamesTheRankAndTheLimit

K_MAX_RANK is SushiBLAS's, so the number in the message is the one a reader has to reconcile
against the tensor they were trying to build.

### ShapeNonPositiveExtentNamesTheOffendingExtent

A zero-element tensor is refused at the type that describes extents, which is why nothing
further down ever has to handle one.

### ShapeDimensionOutOfRangeNamesTheIndexAndTheRank

On a [4, 3] shape, `dim(2)` stops the process with a report holding "Dimension 2" and "rank-2
shape", and `dim(-3)` one holding "Dimension -3" and "rank-2 shape".

### MatmulContractionMismatchNamesBothInnerExtents

`matmul` of [4, 5] and [7, 3] raises an error holding "MATMUL contracts 5 against 7",
"[4, 5]" and "[7, 3]".

### MatmulRejectsARank1OperandAndPrintsWhatItGot

Rank 1 where rank 2 belongs is the commonest way a bias or a flattened sample reaches a gemm.
The message has to show both shapes or the reader cannot tell which side was wrong.

### MatmulNamesBothDtypesWhenTheyDisagree

`matmul` of a float32 and a float64 operand raises an error holding
"MATMUL needs operands of one element type", "float32" and "float64".

### BiasAddNamesTheAxisThatDoesNotLineUp

`bias_add` raises an error for a bias of 5 entries over [4, 3]
("BIAS_ADD bias has 5 entries", "[4, 3]", "3 columns"), for a [1, 3] bias
("BIAS_ADD needs a rank-1 bias", "[1, 3]") and for a rank-1 left operand
("BIAS_ADD needs a rank-2 left operand", "[3]").

### SumRowsAndLogSoftmaxRejectARank1Operand

Over a [6] operand, `sum_rows` raises an error holding "SUM_ROWS needs a rank-2 operand" and
"[6]", and `log_softmax` one holding "LOG_SOFTMAX needs a rank-2 operand" and "[6]".

### ElementwiseOperationsNameBothShapesAndBothDtypes

Over [4, 3] and [4, 2], `add` raises an error holding "ADD needs equally shaped operands" and
both shapes, and `sub`, `mul` and `accumulate` raise the same sentence under their own
operation names. `mul` of a float32 and a float64 operand raises
"MUL needs operands of one element type" with "float32" and "float64".

### NLLLossNamesTheShapeItCannotServe

`nll_loss` raises an error for [4, 3] against [4, 5]
("NLL_LOSS needs equally shaped operands", "[4, 3]", "[4, 5]") and for rank-1 operands
("NLL_LOSS needs rank-2 operands", "[4]").

### ActivationBackwardRefusesAnOperationThatIsNotOne

`activation_backward` with the identity of RELU raises an error holding "sushiai.relu" and
"is not an activation gradient".

### ASecondProducerNamesBothOperationsAndTheRule

The second write is what would make a differentiated graph silently wrong, so the message has
to say which two operations collided.

### AnUnknownOperandNamesTheOperationThatAskedForIt

`add_node` of a RELU that reads the value 9999 stops the process with a report holding
"sushiai.relu", "reads value 9999" and "does not exist". One that writes the value 9999 stops it
with "writes value 9999" and "does not exist".

### LinearNamesItsOwnDimensionsWhenTheInputDoesNotFit

`Linear{6, 4}` over [8, 5] raises an error holding "Linear(6, 4)", "[8, 5]" and
"inner extent must be 6". `Linear{5, 4}` over a [5] input raises
"Linear needs a rank-2 input" with "[5]", and `Linear{0, 4}` raises
"Linear needs positive feature counts" with "in=0" and "out=4".

### ANonScalarRootNamesTheValueAndItsShape

Differentiating from a [4, 3] ReLU output raises an error holding
"needs a scalar to differentiate" and "[4, 3]".

### AOneElementRootOfADifferentiableOperationIsSeeded

The root is a [1, 1] K_MATMUL of a parameter `x` and an input `w`. Differentiation does not
throw and appends two nodes. The first is a K_CONSTANT in the root's shape, the seed. The second
is a K_MATMUL that reads the seed and `w`, and its result is the gradient of `x`, in the shape of
`x`. `test_loss_roots.cpp` covers the seed in full.

### ARootThatRequiresNoGradientNamesItsProducerAndTheCause

Two roots are refused. A K_MATMUL of two inputs is reached by no parameter, and the message
contains "value 3", "sushiai.matmul" and "requires no gradient". A K_MATMUL that reads the
parameter through `stop_gradient` is refused with a message containing "sushiai.matmul",
"requires no gradient" and "K_STOP_GRADIENT", so the message names the barrier as a cause the
caller can look for.

### ARootWhoseProducerHasNoRuleNamesTheProducer

The root is a K_DYNAMIC_SCALE of a parameter, an operation with no backward rule. The message
contains "sushiai.dynamic_scale", "has no backward rule" and "autograd/rules/", the folder a
rule is added in.

### ALossNoOperationProducesIsRejected

The `orphan` value is declared but never written. There is no operation to blame, so the
message has to say that and cannot name one.

### AnOperationWithNoBackwardRuleNamesItselfAndItsNode

"sushiai.unknown" is what `name_of()` reports for a foreign identity, and the node index is
what turns that into something findable.

### AGradientContributedToTheWrongShapeNamesBothShapes

This error is reachable only from inside a backward rule, so the test provokes it the way a
buggy rule would. It builds a graph by hand whose K_SUB claims a result of a different shape from
its operands, so the rule contributes a [4, 3] gradient to a [2, 6] value.
