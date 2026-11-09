# tests/layers/train/test_evaluation_step.cpp


Suites: `Small_CarryingStep`, `Small_CarryingStepNamedFeeds`, `Small_PlainEvaluationStep`,
`Small_EvaluationStep`, `Medium_EvaluationStep`.

Tests a `Train::EvaluationStep`: the feeds it takes, the value it carries between runs and the
step a trainer compiles. The `Small_` suites run on a `HostExecutionBackend`; the `Medium_` suite
runs on the engine of `SaTest::EngineTest`, because its cases run a GELU, a training step and a
batch norm kernel. The model of `Small_CarryingStep` is a local `Accumulator` whose `forward`
feeds a leaf named `gain`, declares a carried leaf `sum` with the initial value 1 and returns
sum + x * gain. The fixture builds the step with `Train::build_carrying_graph` over a [1, 3] row
and fills x with 1, 2, 3 and the gain with 2, 2, 2. A `static_assert` holds that `Accumulator`
is a `TraceModule`.

Every sum of the carrying cases is a whole number far below 2^24, so each float32 operation is
exact and the comparison is equality, with no rounding bound.

## FeedRow

`FeedRow` holds the leaf names a run is fed under and two texts of the refusal it gets.

## TheStepNamesTheModelInputThenTheFedLeaves

`input_names()` is `K_MODEL_INPUT_NAME` then `gain`.

## ACarryingStepReportsItsStateAndItsTrailingWork

`carries_state()` and `step().has_trailing_work()` are both true.

## InputReturnsTheFedLeafOfThatName

`input("gain")` returns a tensor whose second extent is 3.

## InputRefusesAnUnknownNameAndListsTheNames

`input("absent")` fails with `InvalidDescription` and a message naming `'absent'`, `'inputs'` and
`'gain'`.

## ARunReadsWhatTheRunBeforeCarried

After one, two and three runs by name, element i of the output is 1 + (i + 1) * 2 * runs.

## ThreeRunsCompileOnceAndCountThreeRuns

After three runs `compile_count()` is 1 and `run_count()` is 3.

## AResetRestoresTheInitialValue

After two runs, `reset()` and one more run, the output is what one run gives.

## AResetDoesNotCompileAgain

After a run and a reset the compile count is still 1.

## ARunInOrderFeedsTheLeavesAsTheStepListsThem

`run_in_order` with x then the gain gives the output of one run by name.

## ARunOfOneTensorIsRefusedByAStepFedTwoLeaves

`run(x)` fails with `Incompatible` and a message containing "1 tensors", "fed 2" and `'gain'`.
The run count stays 0.

## NamedFeedsThatDoNotCoverTheLeavesAreRefused

`run_by_name` fails with `InvalidDescription`, one row for each cause, and the run count stays 0.
`AnUnknownName` feeds `scale` and the model input ("'scale'", "no input leaf"). `ARepeatedName`
feeds `gain` twice ("'gain' twice"). `AShortList` feeds `gain` alone ("1 named tensors",
"fed 2").

## AFeedOfAnotherShapeIsRefusedUnderItsName

A [1, 4] tensor fed as `gain` fails with `Incompatible` and a message containing "gain",
"[1, 4]" and "[1, 3]".

## AnOrdinaryStepFeedsOneLeaf

A step built with `Train::build_evaluation_graph` over `NN::Linear{3, 3}` lists one input,
`K_MODEL_INPUT_NAME`.

## AnOrdinaryStepCarriesNothing

The same step has `carries_state()` false and `has_trailing_work()` false.

## AResetWithZeroRunsDoesNothing

`reset()` of the same step is ok and the run count stays 0.

## TheStepCompilesOnce

The step is made from an MLP's evaluation graph. `compile_count()` is 1 after `make()` and still
1 after three runs, each written out.

## AnEvaluationStepSeesAWeightTheTrainingStepJustMoved

A training pool and an evaluation pool attach to one state. A host reference computes the logits
of the MLP in double from the four parameters read out of the state, before and after one
training step on the synthetic dataset, and the evaluation output matches it each time. The
reference after the step differs from the one before, so the case cannot pass on unmoved
weights. The bound of a logit is u * (9 M + 17 sum |w2| A) with u the unit roundoff of float32:
the second layer rounds 9 times against M = sum |w2 h| + |b2|, and a hidden unit carries 7
roundings of the first layer against A = sum |w1 x| + |b1|, a GELU slope of at most 1.2 and the 8
operations of the tanh form against |h| <= A. The default policy starts the classifier at zero,
so the reference before the step is zero.

## AProbeRunLeavesBothRunningVectorsUnchanged

The evaluation graph of a batch norm gives its buffers their leaves, and its probe graph is run
once. The bytes of the running mean and the running variance are the same before and after.

## AnInputOfAnotherShapeIsRefusedWithBothShapes

A step compiled for a [4, 6] input refuses a [2, 6] input. The refusal is `Incompatible`, and its
message names "inputs", "[2, 6]" and "[4, 6]".

## AnUnknownOutputNameListsTheNames

`output("boxes")` is refused as `InvalidDescription`. The message names "boxes" and the single
port name, `NN::K_SINGLE_PORT_NAME`.
