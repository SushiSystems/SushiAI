# tests/core/graph/lowering/test_norm.cpp


Suites: `Small_NormLowerings`.

Tests `K_NORM_LOWERINGS` and the scratch rules of batch normalization. The normalized values are
compared with host values in `tests/core/graph/test_lowering/test_hand_values.cpp`, so this file
needs no device. Each case traces BATCHNORM2D over a [2, 3, 3, 4] input and BATCHNORM2D_BACKWARD
over it.

## EachRowPairsItsOperationWithItsLowering

The table has six rows, in the order the header writes them. Each names its operation and the
function the header declares. The two batch normalization rows carry
`request_batchnorm2d_scratch` and `request_batchnorm2d_backward_scratch`, the other four carry
none, and `find_scratch_rule` returns the same for each operation.

## BatchNormAsksForThreeBuffersAndItsGradientForSeven

The forward rule appends three requests and the gradient rule seven. Each request names its own
node, the slots run from 0 in order, the type is float32, and the extents are the four channels
or the whole input.

## AScratchRuleAppendsBehindEarlierRequests

The two rules called on one list leave ten requests: the three of the forward node first, then
the gradient node's, starting again at slot 0.
