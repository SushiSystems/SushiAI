# tests/vision/datasets/test_collator.cpp


Suites: `Small_DatasetsCollator` and `Small_DatasetsCollatorBase`.

Tests the interface `Collator` in
`modules/vision/datasets/include/SushiAI/vision/datasets/collator.hpp`. The header declares the
interface only, and the module builds no in-memory collator (T3 of the plan), so the cases drive
a real `BinaryCollator` through a pointer to the base, on a host engine. The destructor case is
the one that needs a stand-in: only a counting task can show that the destructor of the
derived class ran, and it stays as it was until the owner rules on an in-memory `Collator`
(REPORT.md, decision 49). The collators of the module are tested in their own files, such as
`test_classification_collator.cpp`, and the collator each row of the task table builds in
`test_reader_section.cpp`.

## ClassCollator

`ClassCollator` is the fake of the destructor case. Its task is "fake_classes", its one role
is int32 labels over four classes, and `collate` refuses the first sample with no class,
opening the message with the sample's file. Its destructor raises a counter the test owns.

## TheTaskAndTheRolesAreReadThroughTheBase

Through a `const Collator` pointer to a `BinaryCollator` of the positive class 1, `task` is
`K_TASK_BINARY` and `targets` holds one role, `Data::K_ROLE_LABELS`, of `Float32`.

## ASampleTheTaskCannotUseIsRefusedNamingItsFile

The first sample has the class 1 and the second none. `collate`, called through the base, fails
with InvalidDescription, and the message holds "second.png" and then "no class index".

## ASampleTheTaskCanUseIsCollated

With the classes 1 and 0, `collate` through the base succeeds.

## DestroyingThroughTheBaseRunsTheDerivedDestructor

Resetting a `std::unique_ptr` to the base raises the counter once, which the virtual destructor
of `Collator` makes possible.
