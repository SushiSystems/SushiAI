# tests/core/graph/test_state_id.cpp


Suites: `Small_StateID`.

Tests `StateIDSource`, the counter behind `StateID`, and the identity a `StateIdentity` draws,
copies and moves. Each case draws from a `StateIDSource` of its own through `StateIdentity::get(source)`,
so none reads or moves the process-wide source `Detail::draw_state_id()` uses.

## NoDrawnIdentityIsNoState

A fresh `StateIDSource` does not draw `K_NO_STATE`.

## EachDrawIsAboveTheOneBefore

Three draws in a row from one source return three increasing identities.

## CopySelfAssignmentKeepsTheIdentity

A holder copy-assigned from itself through a reference keeps the identity it drew.

## MoveSelfAssignmentKeepsTheIdentity

A holder move-assigned from itself through a reference keeps the identity it drew.

## GetDrawsOneStableIdentity

A fresh holder peeks `K_NO_STATE`; `get()` draws an identity other than `K_NO_STATE` and returns
the same one on every later call.

## TwoHoldersNeverShareAnIdentity

The identities two holders draw differ.

## ACopyConstructedHolderDoesNotShareTheSourcesIdentity

A copy starts with `K_NO_STATE` and draws an identity other than the source's, which keeps its own.

## ACopyAssignedHolderDoesNotShareTheSourcesIdentity

A holder assigned from another drops its identity, then draws one that is neither the source's
nor the one it held before.

## AMoveConstructedHolderCarriesTheIdentityAway

A holder move-constructed from a source peeks the source's identity; the source peeks `K_NO_STATE`.

## AMoveAssignedHolderCarriesTheIdentityAway

A holder move-assigned from a source peeks the source's identity; the source peeks `K_NO_STATE`.
