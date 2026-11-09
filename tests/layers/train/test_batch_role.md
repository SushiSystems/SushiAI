# tests/layers/train/test_batch_role.cpp


Suites: `Small_BatchRole`, `Small_BatchRoleRefusal`.

Tests `Train::require_batch_role`, the lookup of the target an objective reads from a batch.

This file needs no device. The targets hold default tensors, because the lookup reads only the
role names.

## labelled_batch

`labelled_batch` returns a batch whose targets are `labels` and then `boxes`.

## AHeldRoleIsReturnedAsAPointerIntoTheBatch

Asking for `boxes` returns the address of the second target of the batch, not a copy, and its
role reads `boxes`.

## RefusalRow

`RefusalRow` holds a batch, the role asked for and the two texts the refusal holds.

## Small_BatchRoleRefusal

`Small_BatchRoleRefusal` runs `AMissingRoleIsRefusedByName` over the rows.

## AMissingRoleIsRefusedByName

Each row fails with InvalidDescription naming the role and the objective. The rows are `mask` asked
of a batch holding `labels` and `boxes`, and `labels` asked of a batch with no targets.
