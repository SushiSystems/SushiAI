# tests/core/graph/lowering/test_rule.cpp


Suites: `Small_LoweringRule`.

Tests `LoweringEntry` and `concat_lowerings` with three file-local tables: a one-row table whose
RELU row has a scratch rule, a two-row table for TANH and SIGMOID without one, and their join.
The rows point at two stand-in rules that record nothing, so a row is told apart by the address
it holds. No device is used.

## TheJoinKeepsTheFirstTableInFront

The join has three rows, checked at compile time. Row 0 is the RELU row of the first table, and
rows 1 and 2 are the TANH and SIGMOID rows of the second, each with the rule it was written with.

## ARowWithoutAScratchRuleHasNone

The joined RELU row keeps its scratch rule. The two rows written without one hold null.

## AScratchRuleAppendsTheBuffersOfItsNode

The scratch rule of the joined RELU row, called on a traced RELU node, appends one
`ScratchRequest`: that node, slot 3, extents [2], float32.
