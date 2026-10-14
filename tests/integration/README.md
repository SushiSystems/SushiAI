# Integration tests

The integration tier runs on a real `SushiBLAS::Engine`. Each file here checks a claim that
needs a device to answer; the host-only half of the same claim lives in
`tests/unit/`. This document holds the reasoning behind the tests: why a case is
shaped the way it is, which wrong implementation it is built to catch, and the numbers worked
out by hand. The source keeps one line per case and cites this file.

## The SaTest alias

`test_autograd_numeric.cpp`, `test_forward.cpp`, `test_resilience.cpp` and `test_training.cpp`,
among others, declare `namespace SaTest = SushiAI::Test`. Inside a `TEST_F` body the fixture's base
class injects the name `Test`, so `Test` alone resolves to `::testing::Test`. The alias keeps the
helpers in `SushiAI::Test` reachable from a test body.

## test_pixel_objectives.cpp

Suites: `Integration_PixelObjectives`, in `sushiai_vision_tests`.

The logits are one 2 by 3 map of four classes. The model, `LogitMap`, adds a learnable map to an
input of zeros, so the gradient of the map is the gradient at the logits. The mask ignores two of
the six pixels with the value 255 and holds no pixel of class 3. One step runs with SGD at a rate of
0, and the loss and the gradient are compared with a host computation in double within 1e-5.

### TheCrossEntropyAndItsGradientEqualTheHostComputation

The loss equals the mean of `-log p(label)` over the four kept pixels. The gradient has 24 elements:
`(softmax - one_hot) / 4` on a kept pixel and zero on an ignored one.

### TheDiceLossAndItsGradientEqualTheHostComputation

The loss equals the mean over the present classes of `1 - 2 I / (P + Y + eps)`. The gradient is
compared with a central difference of that host loss with a step of 1e-4.

### AWeightedSumOfBothLossesEqualsTheHostComputation

A `SumObjective` of the cross entropy at weight 1 and the Dice loss at weight 0.5. The loss equals
the weighted sum on the host, and the gradient its central difference.

### AMaskWithNoKeptPixelGivesZero

With all six pixels ignored, the cross-entropy loss and every gradient element are zero within 1e-5.

### ALabelOutsideTheClassesStopsTheFeed

A mask that holds 4 at pixel 2 is refused as `Incompatible`. The message contains "holds 4 at pixel
2" and "below 4".

## test_program_runs.cpp

Suites: `Integration_ProgramRuns`, in `sushiai_program_tests`.

The cases exercise the tables of `applications/programs`: the run table and its fallback, the report
table, and the start of a run. The pieces of a run are an MLP of 6 features, 12 hidden units and 4
classes, its classifier graph for a batch of 8, SGD at a rate of 0.1 and a synthetic dataset of 8
batches. A classification run takes 2 epochs.

### TheRunTablePicksARowOrTheFallback

A second row for 'anchor' and a second fallback are each refused as `InvalidDescription` with
"already registered". Without a fallback, the type 'cross_entropy' is refused with a message that
names it and 'anchor'. With both registered, 'anchor' runs the row's stub, which reports a compile
count of 7, and 'cross_entropy' runs the fallback's, which reports 9.

### AClassificationRunTrainsThroughTheFallback

`register_runs` sets a fallback. A 'cross_entropy' run through it reports one compile, and the
optimizer has taken 16 steps. The run prints two losses, and the last is below the first.

### TheReportTableRefusesAnUnknownTask

A second row for 'classification' is refused with "already registered". A report for
'classification' runs. A report for 'pose' is refused as `InvalidDescription`, and the message names
"model.json", 'pose' and 'classification'.

### ARunStartsFromWhatItsFileStates

A run is trained and saved with its run state to a file in the temporary directory. For a spec
without files `describe_start` returns "initialised" and the optimizer stays at step 0. With
`weights` set it returns the path, and the step count stays 0. With `resume` set it returns "resumed
from " and the path, and the step count equals the trained run's 16.

