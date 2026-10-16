# tests/core/optim/test_cosine_schedule_section.cpp


Suites: `Small_CosineScheduleSection`.

Tests the type and keys the cosine section writer states and what both sides refuse.

This file needs no device. `test_cosine_schedule.cpp` covers the round trip through JSON text,
the defaults of an empty section and each section the reader refuses, so this file pins the
promises those cases leave out.

## section_from

`section_from` parses JSON text as a section of a training file.

## TheWriterStatesTheTypeAndEveryKey

A written section states `K_COSINE_SCHEDULE_TYPE` as its type and holds `base_rate`,
`warmup_iterations`, `total_iterations` and `final_rate` with the values of the schedule.

## ASectionWithoutItsTypeIsRead

A section with no "type" key reads as a cosine schedule with the rates and length it states.

## TheWriterRefusesWhatTheCheckRefuses

A schedule whose final rate is above its base rate is refused by the writer with
`InvalidDescription` and "above the base rate".
