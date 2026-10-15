# tests/layers/train/test_evaluation_step.cpp


Suites: `Medium_CarryingStep`, `Medium_PlainEvaluationStep`.

Tests a `Train::EvaluationStep` that is fed two leaves and carries a value between runs. Both
fixtures need an engine. The model of `Medium_CarryingStep` is a local `Accumulator` whose
`forward` feeds a leaf named `gain`, declares a carried leaf `sum` with the initial value 1 and
returns sum + x * gain. The fixture builds the step with `Train::build_carrying_graph` over a
[1, 3] row and fills x with 1, 2, 3 and the gain with 2, 2, 2. A `static_assert` holds that
`Accumulator` is a `TraceModule`.

## TheStepNamesTheModelInputThenTheFedLeaves

`input_names()` is `K_MODEL_INPUT_NAME` then `gain`, and both `carries_state()` and
`step().has_trailing_work()` are true. `input("gain")` returns a tensor whose second extent is
3. `input("absent")` fails with `InvalidDescription` and a message naming `'absent'`,
`'inputs'` and `'gain'`.

## ARunReadsWhatTheRunBeforeCarried

After one, two and three runs by name, element i of the output is 1 + (i + 1) * 2 * runs within
1e-6. The step compiled once and ran three times.

## AResetRestoresTheInitialValue

After two runs, `reset()` and one more run, the output is what one run gives. The compile count
stays 1.

## ARunInOrderFeedsTheLeavesAsTheStepListsThem

`run_in_order` with x then the gain gives the output of one run by name.

## ARunOfOneTensorIsRefusedByAStepFedTwoLeaves

`run(x)` fails with `Incompatible` and a message containing "1 tensors", "fed 2" and `'gain'`.
The run count stays 0.

## NamedFeedsThatDoNotCoverTheLeavesAreRefused

`run_by_name` fails with `InvalidDescription` three times: for a feed named `scale` ("'scale'",
"no input leaf"), for `gain` given twice ("'gain' twice") and for one feed alone ("1 named
tensors", "fed 2"). The run count stays 0.

## AFeedOfAnotherShapeIsRefusedUnderItsName

A [1, 4] tensor fed as `gain` fails with `Incompatible` and a message containing "gain",
"[1, 4]" and "[1, 3]".

## AnOrdinaryStepFeedsOneLeafAndCarriesNothing

A step built with `Train::build_evaluation_graph` over `NN::Linear{3, 3}` lists one input,
`K_MODEL_INPUT_NAME`. `carries_state()` and `has_trailing_work()` are false, `reset()` is ok and
the run count is 0.
