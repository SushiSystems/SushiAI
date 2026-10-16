# tests/vision/datasets/test_collator.cpp


Suites: `Small_DatasetsCollator`.

Tests the interface `Collator` in
`modules/vision/datasets/include/SushiAI/vision/datasets/collator.hpp`. The header declares the
interface only, so the cases drive a fake task through a pointer to the base. The collators of
the module are tested in their own files, such as `test_classification_collator.cpp`, and the
collator each row of the task table builds in `test_reader_section.cpp`.

## ClassCollator

`ClassCollator` is the fake. Its task is "fake_classes", its one role is int32 labels over four
classes, and `collate` refuses the first sample with no class, opening the message with the
sample's file. Its destructor raises a counter the test owns.

## TheTaskAndTheRolesAreReadThroughTheBase

Through a `const Collator` pointer, `task` is "fake_classes" and `targets` holds one role,
`Data::K_ROLE_LABELS`, of four classes.

## ASampleTheTaskCannotUseIsRefusedNamingItsFile

Of two samples, the second names file 1 and has no class. `collate` fails with
InvalidDescription, and the message holds "b.png" and then "has no class". Once that sample has
a class, `collate` succeeds.

## DestroyingThroughTheBaseRunsTheDerivedDestructor

Resetting a `std::unique_ptr` to the base raises the counter once, which the virtual destructor
of `Collator` makes possible.
