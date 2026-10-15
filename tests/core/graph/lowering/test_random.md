# tests/core/graph/lowering/test_random.cpp


Suites: `Unit_DropoutOps`.

Tests `GraphBuilder::dropout`: the node it emits, the seed it stores, the operands it refuses
and its row in the lowering table. The accepted cases pass the rate 0.25.

## DropoutWritesItsValueAndTheMaskItSaves

Over a [2, 3, 4] float64 input the call is ok and emits one DROPOUT node that reads the input
and writes the output and then the saved mask. The node holds 0.25 in `K_DROPOUT_RATE_SLOT`.
Both outputs have the shape [2, 3, 4], the type float64 and the lifetime `Activation`, and the
graph has no refusal.

## TheSiteSeedSurvivesItsAttributeSlots

For the seeds 0xFEDCBA9876543210, 0 and all 64 bits set, `Graph::dropout_seed` of the emitted
node's attributes returns the seed that was passed.

## DropoutRefusesOperandsThatDoNotFit

An int32 input fails with `InvalidDescription` and "half, float32 or float64 input". The rates
0, 1 and -0.5 each fail with "rate in (0, 1)". After each the node count is unchanged and
`has_refusal()` is true.

## DropoutHasTheRowOfTheRandomTable

`Graph::K_RANDOM_LOWERINGS` has one row, for DROPOUT. `find_lowering` returns
`Graph::lower_dropout`, `lowering_row_count` is 1 and `is_unlowered` is false.
