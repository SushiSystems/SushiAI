# tests/layers/train/test_extra_input_copy.cpp

Suites: `Medium_ExtraInputCopy`.

Tests `Train::copy_extra_inputs` of `extra_input_copy.hpp`, which copies the role of each extra input
of a batch into the leaf a graph declared for it. The model is `SushiAI::Test::build_extra_input_model()`
of `sample_models`: the sum of an [2, 2, 3] input and an [2, 2, 3] extra input named `offset`, then
`Flatten` and `Linear{12, 3}`. A batch is [4, 2, 2, 3] float32 inputs, the int32 labels 0 1 2 1 and
the role `offset`, built by `tests/support/extra_input_batches.hpp`. Values are eighths of a whole
number between -1 and 1, which float32 holds exactly, so a copy compares equal. The cases compile a
pool on the engine of the program.

## CopyExtraInputsCopiesTheRoleOfEachLeafIntoItsLeaf

`copy_extra_inputs` recorded and run through `record_and_run` over the batch of the sample model puts
the eighths of the `offset` role into the pool's `offset` leaf, equal to the values written.

## CopyExtraInputsRefusesAMissingRoleAndAnotherShape

A batch with no `offset` role fails with `InvalidDescription` and `'offset'`. A batch whose `offset` is
[4, 5] fails with `Incompatible`, "offset", "[4, 5]" and "[4, 2, 2, 3]". The failure of the recording
drops its tasks, so both refusals come back from `record_and_run`.
