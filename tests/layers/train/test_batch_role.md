# tests/layers/train/test_batch_role.cpp


Suites: `Small_BatchRole`.

Tests `Train::require_batch_role`, the lookup of the target an objective reads from a batch.

This file needs no device. The targets hold default tensors, because the lookup reads only the
role names.

## labelled_batch

`labelled_batch` returns a batch whose targets are `labels` and then `boxes`.

## AHeldRoleIsReturnedAsAPointerIntoTheBatch

Asking for `boxes` returns the address of the second target of the batch, not a copy, and its
role reads `boxes`.

## AMissingRoleIsRefusedByName

Asking for `mask` fails with InvalidDescription naming the role and the objective. A batch with no
targets refuses `labels` the same way.
