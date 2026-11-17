# tests/core/graph/lowering/test_elementwise.cpp


Suites: `Small_ElementwiseLowerings`, `Small_ElementwiseRow`, `Medium_ElementwiseLowerings`,
`Medium_ElementwisePow`.

Tests `K_ELEMENTWISE_LOWERINGS`, DYNAMIC_SCALE, CAST, SQRT, LOG and POW. The other elementwise rows are compared
with host values in `tests/core/graph/test_lowering/test_hand_values.cpp` and
`tests/core/graph/test_lowering/test_host_values.cpp`. The medium cases run on an engine; scaled
values are compared within `K_ABSOLUTE_TOLERANCE`, 32 unit roundoffs of float and converted values exactly, since float32 holds them without
rounding.

## TheTableHoldsExactlyTheRowsTheHeaderNames

The table has as many rows as the header names (twenty).

## EachRowPairsItsOperationWithItsLowering

One row per entry of the table, in the order the header writes them. Each row names its operation,
the function the header declares for it and no scratch rule, and `find_lowering` returns that
function for the operation.

## DynamicScaleReadsItsFactorOnEveryRun

One compiled step scales a [4] input by a one-element factor. The factor is written as 2 and the
step runs, then it is written as -0.5 and the step runs again. Each run writes the input times the
factor written before it, so the row reads the factor when the step runs and not when it is
recorded.

## CastConvertsEachElementToTheResultType

`cast` of a float32 [3] input to float64 gives a float64 tensor holding the same three values.

## SqrtWritesTheRootOfEachElement

`sqrt` of the [4] input 0.25, 1, 2, 4 gives the host `std::sqrt` of each element, within
`K_ABSOLUTE_TOLERANCE`.

## LogWritesTheLogarithmOfEachElement

`log` of the [4] input 0.5, 1, 2, 4 gives the host `std::log` of each element, within
`K_ABSOLUTE_TOLERANCE`.

## PowWritesTheHostPowerAtTheExponent

One case per exponent 2, 0.5 and -0.5: `pow` of the [4] input 0.5, 1, 1.5, 2. The result
equals the host `std::pow` of each element at its exponent, within `K_ABSOLUTE_TOLERANCE`; the
inputs are small, so one float roundoff stays inside the bound.

## PowRefusesAnExponentThatIsNotFinite

`pow` with an infinite exponent returns `InvalidDescription` naming "finite exponent" and leaves
the node count unchanged. A value outside a function's domain is not refused; the kernel's NaN or
infinity stands.
