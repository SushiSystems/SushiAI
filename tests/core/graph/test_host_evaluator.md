# tests/core/graph/test_host_evaluator.cpp


Suites: `Small_HostEvaluator`, `Small_HostEvaluatorProduct`, `Small_HostEvaluatorBinary`,
`Small_HostEvaluatorUnary`, `Small_HostEvaluatorNll`, `Small_HostEvaluatorCoverage`.

Tests `Graph::HostEvaluator` and `Graph::host_evaluates` in
`modules/core/graph/include/SushiAI/graph/host_evaluator.hpp`. Every case builds a small graph
with a real `Graph::GraphBuilder`, writes its leaves, runs the evaluator and reads a result. No
case needs a device. Results made of small integers are compared exactly; a result that goes
through `exp`, `log`, `tanh` or `sqrt` is compared with its specified value through
`TestSupport::measure_relative_error` over `K_TRANSCENDENTAL_OPERATIONS`, eight rounded
operations.

## AProductMatchesItsHandValues

One row per storage of A = [[1, 2, 3], [4, 5, 6]] and B = [[1, 0], [0, 1], [1, 1]]: plain, A
stored transposed, B stored transposed, and a batch of two whose second B is zero. The product
is [[4, 5], [10, 11]], followed by four zeros for the batch.

## ABinaryRuleCombinesEachPair

One row per ADD, SUB, MUL and LESS_EQUAL over a = {1, 4, -2} and b = {3, 4, 5}: {4, 8, 3},
{-2, 0, -7}, {3, 16, -10} and {1, 1, 1}.

## AUnaryRuleMapsEachElement

One row per SCALE by -0.5, EXP, GELU, SILU, RELU, ABS and SOFTPLUS over {1, -2}. The expected
values are e and 0.1353352832366127 for EXP, 0.8411919906082768 and -0.04540230591222494 for the
tanh form of GELU, and 0.7310585786300049 and -0.2384058440442351 for SILU, v / (1 + exp(-v)).
RELU gives {1, 0} and ABS gives {1, 2}. SOFTPLUS, log(1 + exp(v)), gives 1.3132616875182228 and
0.1269280110429725.

## TheLossIsTheMeanNegativeLogLikelihood

One row per NLL_LOSS case, each compared exactly: the one-hot rows {0, 1} and {1, 0} over the
log-probabilities {-1, -2} and {-3, -4} pick -2 and -3, so the loss is 5 / 2; the soft row
{0.25, 0.75} over {-1, -3} gives 0.25 + 2.25 = 2.5; three one-hot rows over {-1, -2}, {-3, -4}
and {-5, -6} pick -2, -4 and -6, so the loss is 12 / 3 = 4.

## HostEvaluatesNamesTheRuledOperations

`host_evaluates` is true for LEAF, MATMUL, NLL_LOSS, LAYER_NORM, SILU, RELU, ABS, SOFTPLUS, IM2COL,
DEPTHWISE_CONV2D, BATCHNORM2D and BATCHNORM_FOLD, and false for ATAN and CAUSAL_MASK.

## AReshapeKeepsTheElementsInOrder

A [2, 3] value reshaped to [3, 2] reads back as {1, 2, 3, 4, 5, 6}.

## ABiasReachesEveryRow

{10, 20, 30} added to [[1, 2, 3], [4, 5, 6]] gives [[11, 22, 33], [14, 25, 36]].

## TheColumnSumsAddEveryRow

SUM_ROWS of [[1, 2, 3], [4, 5, 6]] is {5, 7, 9}.

## ASelectionFollowsItsCondition

The condition {1, 0, 1} over {1, 2, 3} and {7, 8, 9} gives {1, 8, 3}.

## ACastToFloat32RoundsToFloat

A float64 {0.1, 0.5} cast to float32 reads back as {0.1f, 0.5}.

## AConstantFillsEveryElement

A CONSTANT of -1.5 over [2], which reads no leaf, holds {-1.5, -1.5} after `run()`.

## ALogSoftmaxNormalisesEachRow

The row {0, ln 3} becomes {ln 1/4, ln 3/4}.

## ALayerNormSavesItsStatisticsAndNormalisesTheRow

The row {1, 3} with epsilon 0.25, gain {2, 1} and bias {0, 1} saves the mean 2 and the inverse
deviation 1 / sqrt(1.25), and is written as {-2 / sqrt(1.25), 1 / sqrt(1.25) + 1}.

## AGatherCopiesTheNamedRows

The indices {2, 0} over the table [[1, 2], [3, 4], [5, 6]] give {5, 6, 1, 2}.

## AGatherPastTheTableIsRefused

The index 3 over a table of three rows makes `run()` fail with `InvalidDescription`, naming
"row 3" and "3 rows".

## AChannelSliceKeepsItsRange

Channels 1 and 2 of a [1, 1, 2, 3] value {1, ..., 6} are {2, 3, 5, 6}.

## AConcatJoinsTheLastAxis

{2, 3, 5, 6} of two channels joined with {1, 4} of one channel give {2, 3, 1, 5, 6, 4}.

## AChannelAffineAppliesThePairOfEachChannel

Gain {10, -1} and shift {0, 100} over {2, 3, 5, 6} of two channels give {20, 97, 50, 94}.

## AMaxPoolKeepsTheFirstMaximumOfEachWindow

Windows of two over {3, 3, 2, 7} give the maxima {3, 7} and the flat indices {0, 3}: of two equal
values the first is kept.

## AnUpsampleRepeatsEachPixel

{3, 7} upsampled by 1 by 2 gives {3, 3, 7, 7}.

## AConvolutionSumsEachWindowOverItsFilter

A 2 by 2 `conv2d` over x = 1..9 as a 3 by 3 image, with filters {1, 1, 1, 1} and {1, 0, 0, -1}
and biases {10, 0}, gives {22, -4, 26, -4, 34, -4, 38, -4}: the window sums 12, 16, 24 and 28
plus 10, and -4 for the second filter in every window. A CONV2D has no node of its own; the
builder emits IM2COL, MATMUL, BIAS_ADD and RESHAPE, and the new rule is IM2COL's.

## APaddedConvolutionReadsZeroOutsideTheImage

A 3 by 3 window padded by one over {1, 2, 3, 4} as 2 by 2, with the taps 1..9 and a zero bias,
gives {77, 67, 47, 37}. The first is 1*5 + 2*6 + 3*8 + 4*9.

## ADepthwiseConvolutionFiltersEachChannelByItsOwnTaps

A 2 by 2 depthwise window over two channels, the first 1..9 and the second ten times that,
with taps {1, 1, 1, 1} on the first and {1, 0, 0, -1} on the second, gives
{12, -40, 16, -40, 24, -40, 28, -40}.

## ABatchNormSavesItsStatisticsAndNormalisesEachChannel

Over [1, 2, 2, 2] with channel 0 = {1, 3, 5, 7} and channel 1 = {2, 2, 2, 2}, epsilon 4, gamma
{2, 1} and beta {0, 1}: the means are {4, 2}, the biased variances {5, 0} (exact), the inverse
deviations {1/3, 1/2}, and the result {-2, 1, -2/3, 1, 2/3, 1, 2, 1}.

## AFoldGivesTheScaleAndShiftOfEachChannel

With epsilon 4, gamma {2, 1}, beta {1, 0}, mean {4, 2} and variance {5, 0}, the scale is
{2/3, 1/2} and the shift {-5/3, -1}.

## AnOperationWithoutARuleIsUnsupported

A graph with an ATAN makes `run()` fail with `Unsupported`, naming "no rule" and the operation.

## ALeafLeftUnwrittenIsRefused

`run()` over a leaf "x" that was never written fails with `InvalidDescription`, naming "'x'" and
"write it".

## AReadBeforeRunIsRefused

`read()` of a computed value before `run()` fails with `InvalidDescription` and "no elements".

## AWriteOfAnotherCountIsRefused

Three elements written to a leaf of two fail with `InvalidDescription`, naming "needs 2
elements" and "got 3".

## AWriteOfAComputedValueIsRefused

Writing the result of an EXP fails with `InvalidDescription`, naming the operation and
"computes it".
