# tests/vision/darknet_training_cases/test_darknet_training.cpp


Suites: `Small_DarknetTrainingCases`.

Tests the parts of
`modules/vision/darknet_training_cases/include/SushiAI/vision/darknet_training_cases/darknet_training.hpp`
that need neither a model file nor a reference dump: `delta_within`, and the failure of
`load_darknet_maps` and `load_darknet_head_dump` for a run that does not exist. The checks of a
training step read the Darknet files of `SA_REFERENCE_DATA_DIR`; the suites of
`tests/vision/importers/test_darknet_training/` run them and skip when the files are absent.

## bounds

`bounds` returns a bound of 0.1 plus 0.01 of the reference on a box entry and of 0.01 on an
objectness or class entry.

## reference_slot

`reference_slot` returns one slot of one class: the box entries 1, 2, 3 and 100, the objectness
0.5 and the class 0.25.

## ABoxEntryIsHeldToItsRelativeBound

A box entry of 101 against 100 is within 0.1 plus 1, and one of 101.5 is not.

## AnotherEntryIsHeldToTheAbsoluteBoundAlone

An objectness of 0.505 against 0.5 is within 0.01. A class of 0.3 against 0.25 is not, since the
relative bound of the box entries does not apply to it.

## SizesThatHoldNoWholeSlotFail

Five values against six, five values against five in slots of six, and a slot length of 0 all
fail.

## TheLoadersOfAnAbsentRunAreNotFound

`load_darknet_maps` of a run no reference data directory holds fails with FileNotFound naming
the file. `load_darknet_head_dump` of [yolo] layer 16 of that run fails the same way, naming the
input of layer 15 it reads first. Both hold whether `SA_REFERENCE_DATA_DIR` is set or not.
