# tests/vision/importers/test_darknet_recipe.cpp


Suites: `Small_DarknetRecipe`.

Tests `write_darknet_recipe` in
`modules/vision/importers/include/SushiAI/vision/importers/darknet_recipe.hpp`. Each case
parses `.cfg` text, reads it with `read_darknet_training` and writes the recipe. The sample
`[net]` section states `batch=8`, `subdivisions=2`, `momentum=0.9`, `decay=0.0005`,
`learning_rate=0.001`, `burn_in=100`, `max_batches=2000`, `policy=steps`, `steps=1600,1800`
and `scales=.1,.1`. A `[yolo]` section states a mask, one class, `jitter=.3` and an
`ignore_thresh`.

## WritesTheSolverAsSGD

The `optimizer` section has three keys: the type "sgd", a `momentum` of 0.9 and a
`weight_decay` of 0.0005, each within 1e-12. It has no `learning_rate`.

## WritesTheRateAsAStepSchedule

The `schedule` section reads through `read_step_schedule_section` as a base rate of 0.001, 100
warm-up iterations, the steps 1600 and 1800 and the scales 0.1 and 0.1.

## WritesTheSharedHeadOptionsAsTheAnchorObjective

With two heads of `ignore_thresh=.7`, the `objective` section has the type "anchor" and reads
through `read_head_training_section` with an ignore threshold of 0.7 and
`BoxGradient::DarknetCompatible`.

## HoldsTheObjectiveTheOptimizerAndTheSchedule

The recipe's `training` value has the keys `objective`, `optimizer`, `schedule`.

## NotesEveryKeyWithoutAPlace

The recipe has four notes. One holds "line 2", "batch=8", "[net]" and "batch_size"; one
"line 3", "subdivisions=2" and "[net]"; one "line 8", "max_batches=2000" and "epochs"; one
"line 15", "jitter=.3" and "[yolo]".

## LeavesNoNoteWhenEveryKeyHasAPlace

A `.cfg` of `learning_rate=0.01` and a head of `mask=0` and `classes=1` gives a recipe with no
note.

## HeadsThatDifferAreRefused

Heads with `ignore_thresh` .7 and .5 fail with `InvalidDescription`, "head 0 (line 12)",
"head 1 (line 17)", "'ignore_threshold'" and "one set".

## AFileWithoutAHeadIsRefused

The `[net]` section alone fails with `InvalidDescription`, "[yolo]" and "no objective".
