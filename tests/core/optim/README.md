# tests/core/optim

The tests of the module `core/optim`, one file per header of `modules/core/optim/include/`. The suite prefix of a case gives its ctest label.

## test_apply_step_schedule.cpp

Suites: `Unit_ApplyStepSchedule`.

Tests `Optim::apply_step_schedule` and `Optimizer::set_learning_rate`. The fixture needs an
engine and gives each optimizer one float32 parameter of three zeros whose gradient is 2 in
every element. The schedule most cases use has the base rate 0.5 and halves at updates 2 and 3.

### TheRateSetIsTheScheduleAtTheNextUpdate

Over an SGD made with the rate 0.75, four calls of `apply_step_schedule`, each followed by a
step, return 0.5, 0.25, 0.125 and 0.125. `learning_rate()` equals the returned rate after each
call, and `step_count()` ends at 4.

### AnUpdateUsesTheRateThatWasSet

Over three updates, every element of the parameter equals the running sum of minus the applied
rate times the gradient, within 1e-6.

### AdamWTakesTheRateThroughTheSameSeam

An AdamW made with the rate 0.75 reports 0.75 through the `Optimizer` interface. After
`apply_step_schedule` returns 0.5, `options().learning_rate` is 0.5.

### ANegativeRateIsFatal

`set_learning_rate(-0.1)` stops the process with a report containing "learning rate" and
"negative".

### ACosineScheduleTakesTheSameSeam

A `CosineSchedule` with the base rate 0.5, two warm-up updates and four updates in total gives
0.25 at the first update. Held in a `RateSchedule` it gives 0.5 at the second update, and
`learning_rate()` is 0.5. At the third update a `RateSchedule` holding the halving schedule
gives 0.125 and the held cosine gives 0.25.

## test_cosine_schedule.cpp

Suites: `Unit_CosineSchedule`.

Tests `Optim::CosineSchedule::rate_at`, `Optim::validate_cosine_schedule` and the section
reader and writer of `cosine_schedule_section.hpp`. The schedule most cases start from has the
base rate 1, 10 warm-up updates, 110 updates in total and the final rate 0.1. Rates are compared
within 1e-15.

### TheWarmupIsLinear

The rate at iterations 0, 1, 5 and 10 is 0, 0.1, 0.5 and 1.

### TheDecayFollowsHalfACosine

At iterations 35, 60 and 85 the rate is 0.1 + 0.9 * 0.5 * (1 + c), where c is the square root
of 2 over 2, then 0, then minus the square root of 2 over 2. The middle value is 0.55.

### TheRateStaysAtTheFinalOne

The rate at iterations 110 and 1000000 is 0.1.

### WithoutAWarmupTheFirstRateIsTheBaseRate

With `warmup_iterations` 0 the rate at iteration 0 is 1 and the rate at iteration 1 is below 1.

### ADecayOfNoLengthDividesByNothing

With `total_iterations` equal to `warmup_iterations`, `validate_cosine_schedule` is ok, the rate
at iteration 9 is 0.9 and the rate at iteration 10 is 0.1.

### ANegativeIterationIsFatal

`rate_at(-1)` stops the process with a report containing "iteration 0 or later".

### TheCheckRefusesWhatRateAtCannotUse

`validate_cosine_schedule` fails with `InvalidDescription` for a warm-up of -1 ("warm-up",
"negative"), a total of 5 under a warm-up of 10 ("ends at iteration 5", "warm-up of 10"), a
final rate of -0.1 ("final rate", "negative") and a final rate of 2 ("above the base rate").

### ASectionRoundTripsEveryField

A schedule with the base rate 0.0006 and the final rate 0.00006 goes through
`write_cosine_schedule_section`, `Config::write_json`, `parse_json` and
`read_cosine_schedule_section`. The section's `type` is "cosine" and the four fields read back
equal to the ones written.

### AnEmptySectionReadsAsTheDefaults

`read_cosine_schedule_section` over `{}` returns a base rate of 0, a warm-up of 0, a total of 0
and a final rate of 0.

### TheReaderRefusesAndNamesTheKey

The reader fails with `InvalidDescription` for the type `step` ("schedule 'type'", "'step'",
"a cosine schedule states 'cosine'"), the key `steps` ("schedule states the key 'steps'",
"'total_iterations'"), a `total_iterations` of 1.5 ("schedule 'total_iterations'", "whole
number"), a final rate above the base rate ("schedule: ", "above the base rate") and an array
in place of the section ("schedule must be an object").

### TheWriterRefusesARateThatIsNotFinite

`write_cosine_schedule_section` over a base rate of infinity fails with `InvalidDescription`
and a message containing "schedule 'base_rate'" and "finite".

## test_gradient_clip.cpp

Suites: `Unit_GradientClip`.

Tests the gradient clip of `Optim::Optimizer`: `enable_gradient_clip`, `set_clip_norm`, the
`clip_norm` key of `Optim::make_optimizer` and the update a clipped step applies. The fixture
needs an engine. Each slot has a float32 parameter of zeros and a stated gradient, and
`make_sgd` returns an SGD of rate 1 with no momentum and no decay. A parameter is compared
within 1e-6.

### ANormAboveTheThresholdScalesTheUpdate

With the gradient 3, 4, 0, 0 and a clip norm of 1, one step leaves the parameter at -0.6, -0.8,
0, 0. `gradient_clip()` is not null and its `applied_scale()` is 0.2.

### ANormBelowTheThresholdLeavesTheUpdate

With the same gradient and a clip norm of 10, one step leaves the parameter at -3, -4, 0, 0 and
`applied_scale()` is exactly 1.

### TheNormIsTakenOverEverySlot

With the gradients 3, 0 and 0, 4, 0 in two slots and a clip norm of 2.5, one step leaves the
parameters at -1.5, 0 and 0, -2, 0. `gradient_count()` is 2.

### AThresholdSetAtRunTimeIsTheNextUpdates

With the gradient 3, 4 and a clip norm of 5, the first step leaves -3, -4. After
`set_clip_norm(2.5)`, `max_norm()` is 2.5 and the second step leaves -4.5, -6.

### TheMomentumKernelReadsTheFactor

An SGD of rate 1 and momentum 0.5 with a clip norm of 1 leaves the parameter at -0.6, -0.8
after its first step.

### AdamWReadsTheSameFactor

An AdamW of default options over the gradient 3, 4 with a clip norm of 1 reports an
`applied_scale()` of 0.2 after one step.

### AClipAddsItsTasksAndItsAbsenceAddsNone

Over two slots, `record_into_step` returns 2 for an SGD without a clip, whose
`gradient_clip()` is null, and 5 for one with a clip.

### AClipRefusesWhatItCannotCompute

`enable_gradient_clip` fails with `InvalidDescription` and "positive and finite" for the norms
0 and -1, and `gradient_clip()` stays null. On an SGD made with `UpdateKernel::Separate` it
fails with a message containing "UpdateKernel::Fused".

### ANormSetWithoutAClipIsFatal

`set_clip_norm(1.0)` on an optimizer without a clip stops the process with a report containing
"no gradient clip". After a clip is enabled, `set_clip_norm(0.0)` stops it with "clip norm" and
"positive and finite".

### TheKeyClipNormEnablesTheClip

For the types `sgd` and `adamw`, `make_optimizer` with `{"clip_norm": 0.5}` returns an
optimizer whose clip has the `max_norm()` 0.5, and with `{}` one whose `gradient_clip()` is
null. `{"clip_norm": 0}` fails with `InvalidDescription`, "optimizer 'clip_norm'" and "positive
and finite".

## test_rate_schedule.cpp

Suites: `Unit_RateSchedule`.

Tests `Optim::RateSchedule`, the variant of `StepSchedule` and `CosineSchedule`: its section
reader and writer, `rate_at`, `base_rate_of`, `validate_rate_schedule` and
`Optim::read_rate_schedule`. The step section has the base rate 0.5 and one step at 3 of scale
0.5. The cosine section has the base rate 0.25, a warm-up of 2, a total of 10 and the final
rate 0.125. A rate is compared with a tolerance of 0.

### ASectionIsReadByTheTypeItStates

The step section reads as a `StepSchedule` with the base rate 0.5. The cosine section reads as
a `CosineSchedule` with `total_iterations` 10 and the final rate 0.125.

### ASectionWithoutATypeIsAStepSchedule

`{"base_rate": 0.5}` reads as a `StepSchedule`.

### AnUnknownTypeIsRefusedWithTheTypesThereAre

The type `linear` fails with `InvalidDescription`, "schedule 'type' is 'linear'", `'step'` and
`'cosine'`. A `type` of 3 fails with "schedule 'type' must be a string", and the number 7 in
place of the section with "schedule must be an object".

### AKeyOfAnotherTypeIsRefused

`steps` in a cosine section fails with "states the key 'steps'", and `final_rate` in a step
section with "states the key 'final_rate'".

### TheRateIsThatOfTheScheduleHeld

`rate_at` of the step schedule is 0.5 at iteration 2 and 0.25 at iteration 3. `rate_at` of the
cosine schedule is 0.125 at iterations 1 and 10. `base_rate_of` is 0.5 and 0.25.

### AWrittenSectionStatesItsTypeAndReadsBack

`write_rate_schedule_section` of the cosine schedule gives a section whose `type` is "cosine".
It reads back as a `CosineSchedule` with `warmup_iterations` 2.

### ARunReadsACosineScheduleUnderTheSameRules

`read_rate_schedule` returns no schedule for a null section beside an optimizer that states
`learning_rate`, and a `CosineSchedule` for the cosine section beside `{}`. The cosine section
beside a `learning_rate` fails with `InvalidDescription` and "stated twice". A cosine section
without a base rate fails with "'base_rate' must be stated and above zero".

### ARunReadsAStepScheduleAsAStepSchedule

`read_rate_schedule` of the step section beside `{}` returns a `StepSchedule` with the base
rate 0.5.

### AHeldScheduleIsCheckedByItsOwnValidation

`validate_rate_schedule` fails with `InvalidDescription`, "2 steps" and "1 scales" for a held
`StepSchedule` of two steps and one scale. It is ok for a held `CosineSchedule` with the base
rate 0.5 and a total of 10.

## test_run_schedule.cpp

Suites: `Unit_RunSchedule`.

Tests `Optim::read_rate_schedule`, which takes the schedule section and the optimizer section
of a training file. The schedule section of the cases is a step schedule with the base rate
0.00261, a warm-up of 1000 and one step at 400000 of scale 0.1.

### ANullSectionYieldsNoSchedule

A null schedule section beside an optimizer that states `learning_rate` returns no schedule.

### AScheduleIsReadWhenTheOptimizerStatesNoRate

The schedule section beside `{"momentum": 0.9}` returns a `StepSchedule` with the base rate
0.00261, a warm-up of 1000 and one step at 400000.

### ARateStatedTwiceIsRefused

The schedule section beside `{"learning_rate": 0.00261}` fails with `InvalidDescription` and a
message containing "stated twice", `'learning_rate'` and `'base_rate'`.

### AScheduleWithoutABaseRateIsRefused

`{"type": "step"}` fails with `InvalidDescription`, `'base_rate'` and "above zero".

### ABaseRateNotAboveZeroIsRefused

The base rates 0 and -0.01 each fail with `InvalidDescription`, `'base_rate'` and "above zero".

### WhatTheSectionReaderRefusesIsRefused

A section with the key `gamma` fails with `InvalidDescription`, "schedule" and `'gamma'`. A
string in place of the section fails with "schedule" and "object".

## test_run_state.cpp

Suites: `Unit_OptimizerRunState`, `Unit_LossScaleStateRestore`.

Tests `Optimizer::run_state`, `restore_scalars` and `check_state_owner`, and
`Optim::LossScaleState::restore`. The `Unit_OptimizerRunState` fixture needs an engine and
makes the slots `weight` and `bias`, each three float32 zeros with a gradient of ones.

### SGDListsItsMomentumBuffersAndItsStepCount

After one step of an SGD with momentum 0.9, `type_name()` is "sgd" and the run state lists the
tensors `weight.momentum` and `bias.momentum`, the first of three elements. Its one scalar is
`step_count`, a `uint64_t` of 1.

### SGDWithoutMomentumListsNoTensor

The run state of an SGD with momentum 0 has no tensor and one scalar whose value is 0.

### AdamWListsBothMomentsOfEachSlot

`type_name()` is "adamw" and the run state lists `weight.first_moment`, `weight.second_moment`,
`bias.first_moment` and `bias.second_moment`, in that order. The first listed tensor has the
data pointer of `first_moment(0)` and the fourth that of `second_moment(1)`.

### TheStepCountIsRestored

After `restore_scalars` with a step count of 41, `step_count()` is 41, and it is 42 after one
step.

### AnotherOwnerIsRefusedWithBothTypes

On an SGD, `check_state_owner("sgd")` is ok and `check_state_owner("adamw")` fails with
`Incompatible`, `'adamw'` and `'sgd'`.

### AScalarThatDoesNotFitIsRefusedByName

`restore_scalars` on an SGD fails with `Incompatible` four times: for no scalar ("does not
hold", "'step_count'", "'sgd'"), for an added `loss_scale` ("'loss_scale'", "'sgd'", "does not
keep"), for `step_count` given twice ("more than once") and for a `step_count` that is a real
("a real number where a count is kept"). The step count stays 0.

### TakesASavedScaleAndRefusesOneOutOfBounds

`restore(4096.0, 1999)` is ok and sets the scale to 4096 and the count of good steps to 1999.
One `observe(false)` then gives the scale 8192 and the count 0. `restore` fails with
`Incompatible` for the scale 3000 ("power of two"), the scale 0.5 ("bounds") and a count of
2000 ("growth interval"), and the scale stays 8192.

## test_step_schedule.cpp

Suites: `Unit_StepSchedule`.

Tests `Optim::StepSchedule::rate_at` and `Optim::validate_step_schedule`. Every case starts from
the base rate 0.001, and the warm-up cases use 1000 warm-up updates at the default power 4. A
rate is compared with a value the test computes in double, within 1e-15 plus 1e-12 times the
expected value.

### ConstantPolicyKeepsBaseRate

With no warm-up and no step, the rate at iterations 0, 1 and 1000000 is the base rate.

### WarmupRisesAsPowerOfIterationFraction

During the warm-up the rate is the base rate times (iteration / 1000) to the power 4. The case
checks iterations 0, 1, 500 and 999, and the rate at 0 is 0.

### WarmupEndsAtItsLengthWithTheBaseRate

At iterations 1000 and 1001 the rate is the base rate.

### WarmupPowerIsTheExponentOfTheRamp

With `warmup_power` 1 the rate at iteration 250 is a quarter of the base rate. With
`warmup_power` 2 the rate at iteration 500 is a quarter of the base rate.

### StepAppliesAtItsListedIteration

With steps at 400 and 450, each of scale 0.1, the rate is the base rate at 399, a tenth of it
from 400 to 449 and a hundredth of it at 450 and 451.

### WarmupReplacesThePolicyUntilItsLength

With a step at 500 of scale 0.1 inside the warm-up, the rates at 500 and 999 are the plain
warm-up values. At 1000 the rate is a tenth of the base rate.

### StepAtFirstIterationPastWarmupApplies

With a step at 1000 of scale 0.5, the rate at 999 is the warm-up value and the rate at 1000 is
half the base rate.

### ValidateAcceptsEmptyAndAscendingSchedules

`validate_step_schedule` is ok for a default `StepSchedule` and for a warm-up schedule with
steps at 400 and 450 and two scales.

### ValidateRefusesDescendingSteps

Steps listed as 450 then 400 fail with `InvalidDescription`.

### ValidateRefusesRepeatedStep

Steps listed as 400 and 400 fail with `InvalidDescription`.

### ValidateRefusesScalesLongerThanSteps

One step with two scales fails with `InvalidDescription`.

### ValidateRefusesStepsLongerThanScales

Two steps with one scale fail with `InvalidDescription`.

### ValidateRefusesNegativeWarmup

A `warmup_iterations` of -1 fails with `InvalidDescription`.

### ValidateRefusesNegativeWarmupPower

A `warmup_power` of -1 fails with `InvalidDescription`.

## test_step_schedule_section.cpp

Suites: `Unit_StepScheduleSection`.

Tests `Optim::read_step_schedule_section` and `Optim::write_step_schedule_section`. The full
section states `type`, `base_rate`, `warmup_iterations`, `warmup_power`, `steps` and `scales`.
A round trip writes a schedule to a section, the section to JSON text, parses the text and
reads it, and compares every field exactly.

### TheDefaultsRoundTripThroughTheirSection

A default `StepSchedule` round-trips.

### AWarmupAndTwoStepsRoundTripThroughTheirSection

A schedule with the base rate 0.00261, a warm-up of 1000 at the power 2.5, steps at 400000 and
450000 and the scales 0.1 and 0.3 round-trips.

### AFullSectionRoundTripsThroughItsSchedule

The full section, read to a schedule and written back, gives the same JSON text as the section
itself.

### ReadsEveryFieldOfASection

The full section reads as the base rate 0.00261, a warm-up of 1000, the power 4, the steps
400000 and 450000 and the scales 0.1 and 0.1.

### AKeyLeftOutKeepsTheDefaultOfItsField

`{}` and `{"type": "step"}` each read as a default `StepSchedule`. `{"base_rate": 0.01}` reads
as a default schedule whose base rate is 0.01.

### WritesTheTypeAndEveryKeyInOrder

The written section has the keys `type`, `base_rate`, `warmup_iterations`, `warmup_power`,
`steps`, `scales`, in that order. `type` is `K_STEP_SCHEDULE_TYPE`, `warmup_iterations` is 1000
and the second step is 450000.

### ReadsAWholeNumberAsARateAPowerAndAScale

A section with `base_rate` 1, `warmup_power` 2 and the scale 3 reads as 1.0, 2.0 and 3.0.

### RefusesASectionThatIsNotAnObject

An array and a null value each fail with `InvalidDescription`, "schedule" and "an object".

### RefusesAnotherType

The type `cosine` fails with "schedule 'type'", `'cosine'` and `'step'`. A `type` of 1 fails
with "schedule 'type'" and "a string".

### RefusesAnUnknownKey

The key `burn_in` fails with `InvalidDescription` and a message containing "schedule",
`'burn_in'`, "'type', 'base_rate'" and `'scales'`.

### RefusesAValueOfAnotherKind

Five sections fail with `InvalidDescription`: a string `base_rate` ("schedule 'base_rate'", "a
number"), a `warmup_iterations` of 1.5 ("a whole number"), a `steps` of 100 ("an array"), a
step of 20.5 ("schedule 'steps' entry 1", "a whole number") and a scale of "half" ("schedule
'scales' entry 0", "a number").

### RefusesTheSchedulesTheScheduleCheckRefuses

The reader fails with `InvalidDescription` and "schedule" for two steps without scales ("2
steps", "0 scales"), for steps 200 then 100 ("strictly ascend"), for a warm-up of -1 ("-1",
"negative") and for a warm-up power of -2 ("-2", "negative").

### WriterRefusesANumberThatIsNotFinite

The writer fails with `InvalidDescription` and "finite" for an infinite base rate ("schedule
'base_rate'"), an infinite warm-up power ("schedule 'warmup_power'") and a NaN second scale
("schedule 'scales' entry 1").

### WriterRefusesTheSchedulesTheScheduleCheckRefuses

The writer fails with `InvalidDescription` and "schedule" for two steps with one scale ("2
steps", "1 scales") and for steps 450000 then 400000 ("strictly ascend").

## test_optimizer.cpp

Suites: `Unit_ParameterDecay`.

Tests `ParameterSlot::decays` and the `decay_all` option of SGD and AdamW. The fixture needs an
engine and makes two slots of three float32 elements, each starting at 2 with a zero gradient:
`weight`, which decays, and `bias`, which does not. Both optimizers use the rate 0.5 and the
weight decay 0.5, so a decayed parameter is 1.5 after one step. Values are compared within
1e-6.

### ASlotDecaysUnlessItSaysOtherwise

A default `ParameterSlot` has `decays` true.

### SGDDecaysOnlyTheSlotsThatTakeIt

For the kernels `Fused` and `Separate`, one SGD step leaves the weight at 1.5 and the bias
at 2.

### AdamWDecaysOnlyTheSlotsThatTakeIt

For the kernels `Fused` and `Separate`, one AdamW step leaves the weight at 1.5 and the bias
at 2.

### DecayAllDecaysEverySlot

With `decay_all` true, one step of SGD and one step of AdamW each leave both parameters at 1.5.

### TheKeyDecayAllReachesTheOptimizer

For the types `sgd` and `adamw`, an optimizer from `make_optimizer` without the key `decay_all`
leaves the weight at 1.5 and the bias at 2 after one step. With `"decay_all": true` it leaves
both at 1.5.
