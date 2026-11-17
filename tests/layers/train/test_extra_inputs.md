# tests/layers/train/test_extra_inputs.cpp

Suites: `Small_ExtraInputsGraph`, `Small_ExtraFeeds`, `Medium_ExtraInputsRun`.

Tests the extra inputs of a model: the leaf each step builder declares for one, the role that feeds
it, `Train::extra_feeds`, the copy of a training step and the feed of an evaluation step. The model is
`SushiAI::Test::build_extra_input_model()` of `sample_models`: the sum of an [2, 2, 3] input and an
[2, 2, 3] extra input named `offset`, then `Flatten` and `Linear{12, 3}`, publishing `logits`. A batch
is [4, 2, 2, 3] float32 inputs, the int32 labels 0 1 2 1 and the role `offset`. Values are eighths of
a whole number between -1 and 1, which float32 holds exactly, so a copy compares equal.

`Small_ExtraInputsGraph` traces graphs and needs no device. `Small_ExtraFeeds` allocates its tensors on
the engine of a `HostExecutionBackend`. `Medium_ExtraInputsRun` compiles a step on the engine of the
program.

## TheTrainingGraphDeclaresALeafOfTheBatchShapeForEachExtraInput

`build_training_graph` over the sample model lists one extra input leaf: `offset`, a value other
than the model input, of shape [4, 2, 2, 3] and element type float32. `find("offset")` returns it and
`find("absent")` returns `K_NO_VALUE`.

## TheTrainingGraphRequiresTheRoleOfEachExtraInput

The roles the graph requires are two: `labels`, which the cross entropy reads, and `offset`, of the
sample shape [2, 2, 3], float32, with no class count. `offset` is not a leaf of the objective, so
`leaves.has("offset")` is false and the objective's feed does not write it.

## TheProbeEvaluationAndCarryingGraphsDeclareTheLeafToo

The probe, evaluation and carrying graphs of the sample model each list the leaf `offset` of
[4, 2, 2, 3] in `TracedStep::extra_inputs`.

## AModelWithNoExtraInputDeclaresNone

An MLP, which has no `extra_inputs()`, gets no extra leaf, and its graph requires the role `labels`
alone.

## AnExtraInputNamedAsTheModelInputIsRefused

A model graph whose extra input is called `inputs`, the name of the model input in every step, fails
`build_probe_graph` with `InvalidDescription`, `'inputs'` and "model input".

## EachLeafIsPairedWithTheRoleOfItsNameInLeafOrder

Over a batch whose roles are `scale`, `offset` and `labels`, the leaves `offset` and `scale` give two
feeds in that order, the first pointing at the second role of the batch and the second at the first.

## TheFeedsAreAppendedAndWhatTheyHeldIsKept

A feed list that holds `inputs` before the call holds `inputs` and `offset` after it.

## ABatchWithoutTheRoleOfALeafIsRefusedNamingTheLeaf

A leaf `lengths` the batch has no role for fails with `InvalidDescription`, `'lengths'` and "extra
input".

## ATrainingStepCopiesTheRoleIntoTheLeaf

After `run_step` of a trainer over the sample model, with a learning rate of 0, the pool's tensor of
the `offset` leaf holds the eighths of the batch's `offset` role.

## TheExtraInputMovesTheLossAndTheSameBatchRepeatsIt

With a learning rate of 0 the weights stay as the initialiser left them. A step over one offset and a
step over another give two losses that differ; a third step over the first offset gives the first
loss again, bit for bit. The extra input therefore reaches the loss through the graph, and nothing
else moves between the steps.

## AStepRefusesABatchWithoutTheRoleOrWithAnotherShape

A batch with no `offset` role fails with `InvalidDescription` and `'offset'`. A batch whose `offset` is
[4, 5] fails with `Incompatible`, "offset", "[4, 5]" and "[4, 2, 2, 3]", the shape the step compiled.

## AnEvaluationStepReadsTheExtraInputFromTheBatch

`run_batch` over inputs x and offsets o gives the same `logits` as `run_batch` over inputs x + o and
offsets of 0. The model adds the two before its linear layer, and float32 adds the eighths exactly,
so the equality is bit for bit and the weights, which are the same step's, drop out.

## ARunByPositionOfAStepWithAnExtraInputNamesBothLeaves

The names of the evaluation step's inputs are `inputs` and `offset`, in that order. `run` with the
model input alone fails with `Incompatible`, "fed 2" and `'offset'`.

## AnEvaluationStepRefusesABatchWithoutTheExtraInputNamingIt

`run_batch` of the evaluation step over a batch whose `offset` role is missing fails with
`InvalidDescription` and `'offset'`, the input `sa inference` reports when its data omits it.

`Train::copy_extra_inputs` has its own file, `test_extra_input_copy.cpp`.
