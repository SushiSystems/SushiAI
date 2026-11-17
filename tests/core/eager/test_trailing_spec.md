# tests/core/eager/test_trailing_spec.cpp

Suites: `Small_TrailingSpec` and `Small_TrailingSpecDifference`.

Tests `TrailingSpec` of `trailing_spec.hpp`, the work a step records behind its regions, as storage
ids. The file needs no device; the recorder a spec holds is read by address only.

## ASpecWithNoCopyAndNoBlendIsPlainWhateverTheCallersRecorder

A spec with no copy and no blend is plain, with and without the caller's recorder: the recorder is
all the step then records.

## ASpecWithACopyIsNotPlain, ASpecWithABlendIsNotPlain

One copy, or one blend, makes the spec not plain.

## TwoSpecsBuiltTheSameWayAreEqual

Two specs with the same recorder, copies and blends in the same order are equal.

## TwoBlendsAreEqualOnlyWhenEveryFieldIsEqual, TwoCopiesAreEqualOnlyWhenBothIdsAreEqual

A blend changes under any of its target, source, momentum or scale; a copy under its source or its
destination.

## AnyChangedPartOfTheWorkMakesTwoSpecsDiffer

One row per change against a reference of two copies and one blend: another caller's recorder, no
recorder, a copy fewer, a copy more, the copies in another order, another blend target and no blend.
