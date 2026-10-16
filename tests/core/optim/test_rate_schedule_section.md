# tests/core/optim/test_rate_schedule_section.cpp


Suites: `Small_RateScheduleSection`.

Tests that a schedule section of either type is written under its own type.

This file needs no device. `test_rate_schedule.cpp` covers reading by type, a section without
a type, the refusals of the reader and the cosine round trip, so this file pins the writer's
side for the step type and its refusals.

## AStepScheduleIsWrittenAsStep

A step schedule is written with the type "step" and reads back as a step schedule with its base
rate and its one step.

## ACosineScheduleIsWrittenAsCosine

A cosine schedule is written with the type `K_COSINE_SCHEDULE_TYPE`.

## TheWriterPassesOnTheTypesRefusal

A step schedule with an infinite base rate is refused with `InvalidDescription` naming
"'base_rate'" and "finite", the refusal of the step writer.
