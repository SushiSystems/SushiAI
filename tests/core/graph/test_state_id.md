# tests/core/graph/test_state_id.cpp


Suites: `Small_StateID`.

Tests the process-wide counter behind `StateID` and the self-assignment of a `StateIdentity`.
Drawing, copying and moving a holder are pinned by the `Small_Ir` cases of
`tests/core/graph/test_ir.cpp`.

## NoStateIsZeroAndIsNeverDrawn

`K_NO_STATE` is 0, and `draw_state_id` does not return it.

## EachDrawIsAboveTheOneBefore

Three draws in a row return three increasing identities.

## SelfAssignmentKeepsTheIdentity

A holder copy-assigned from itself, then move-assigned from itself, both through a reference,
keeps the identity it drew.
