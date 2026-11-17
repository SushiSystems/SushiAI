# tests/vision/datasets: test_batch_inputs.cpp, test_classification_collator.cpp, test_detection_collator.cpp


The scaled inputs and the roles the classification and detection collators write, and the batches they refuse. Every case runs on a host engine.

Suites: `Small_BatchInputs`, `Small_ClassificationCollator`, `Small_ClassificationCollatorTargets`,
`Small_DetectionCollator`, `Small_DetectionCollatorMakeRefusal` and `Small_DetectionCollatorTargets`.

Tests `write_batch_inputs`, `ClassificationCollator` and `DetectionCollator` in
`modules/vision/datasets/include/SushiAI/vision/datasets/`. `Small_BatchInputs` holds a
`HostExecutionBackend`, and the collator suites derive from `Support::CollatorBench`, so their
batches are host tensors and no device or process-wide engine is reached. The collator cases
hold two gray samples of one row and two columns, the first of level 51 from `first.png` and
the second of level 255 from `second.png`, and allocate a batch of two such images.

## InputsAreBytesDividedBy255

`write_batch_inputs` writes 0.2, 0.2, 1.0, 1.0, each within the typed tolerance of a byte divided by
255 in float, 32 times the float unit roundoff.

## TheDefaultNormalisationIsTheByteDividedBy255ForEveryLevel

One gray image holds each level 0 to 255 once. With a default `Config::InputNormalise`,
`write_batch_inputs` writes for every level the float of `level / 255.0f`, equal bit for bit; the
oracle is the division the writer made before a normalisation could be stated.

## ABlueFirstPixelTakesTheScaleTheMeanAndTheDeviation

One RGB pixel of the bytes 2, 4, 8 with `bgr`, scale 0.5, mean (1, 0.5, 0.25) and std (2, 0.5,
0.25) gives 1.5, 3, 3: the bytes read 8, 4, 2, halve to 4, 2, 1, lose the mean to 3, 1.5, 0.75 and
divide by the std. Every number is a power of two, so each value is exact.

## ARedFirstPixelKeepsEachByteOnItsChannel

The same pixel and numbers with the default order give 0, 3, 15: 1, 2, 4 less the mean 0, 1.5,
3.75 over the std.

## ANormalisationOfColoursIsRefusedForAGrayInput

With `bgr` on a one-channel input, `write_batch_inputs` fails with `Incompatible`,
"write_batch_inputs", "1 channels" and "stated for 3".

## AScaleAloneAppliesToAGrayInput

With only scale 0.5 on the gray levels 51 and 255, the inputs 25.5 and 127.5 are exact in float.

## AnImageOfAnotherSizeIsRefused

With the second image 2 by 2, `write_batch_inputs` fails with `Incompatible`,
"write_batch_inputs", "image 1" and "2 by 2".

## ATensorOfAnotherImageCountIsRefused

With one sample and a tensor for two, `write_batch_inputs` fails with `Incompatible`,
"write_batch_inputs" and "holds 2 images".

## MakeRefusesZeroClasses

`ClassificationCollator::make(0)` fails with `InvalidDescription`, "ClassificationCollator" and
"0 classes".

## ClassificationPublishesOneInt32LabelPerSample

A collator of 7 classes reports the task `K_TASK_CLASSIFICATION` and one target: the role
`K_ROLE_LABELS`, an empty sample shape, `Int32` and 7 classes.

## ClassificationWritesTheClassOfEachSample

With the class indices 6 and 0, `collate` writes the labels 6 and 0, and the third input value
is 1.0 within the typed tolerance of a byte divided by 255.

## ClassificationWritesTheInputsByItsNormalisation

A collator built with scale 0.5 writes 25.5 and 127.5 for the gray levels 51 and 255.

## ClassificationNamesTheFileOfASampleWithoutAClass

With no class index on the second sample, `collate` fails with `InvalidDescription`,
"second.png", "ClassificationCollator" and "no class index".

## ClassificationNamesTheFileOfASampleOfAClassOutsideTheSet

With the class index 7 over 7 classes on the second sample, `collate` fails with
`InvalidDescription`, "second.png", "class index 7" and "[0, 7)".

## ABatchWithoutTheRoleIsRefused

Over a batch allocated without targets, `collate` fails with `Incompatible`,
"ClassificationCollator" and "no role 'labels'".

## MakeRefusesACountOfNothing

`Small_DetectionCollatorMakeRefusal`, one row each: `DetectionCollator::make` fails with
`InvalidDescription` for 0 classes ("DetectionCollator", "0 classes") and for 0 objects
("0 objects").

## DetectionPublishesThreeRoles

A collator of 3 classes and 4 objects reports the task `K_TASK_DETECTION` and three targets:
`K_ROLE_BOXES` of shape [4, 4] and `Float32`, `K_ROLE_CLASSES` of shape [4], `Int32` and 3
classes, and `K_ROLE_VALID` of shape [4] and `Float32`.

## DetectionWritesCornersAndPadsWithZero

Every target slot is first filled with 9. With one object of class 2 and the box (0.25, 0.5,
1.5, 1.0) on the second image and two slots per image, `collate` writes that box, the class 2
and the valid flag 1 into the third slot and zero into every other slot.

## DetectionNamesTheFileOfAnImageWithTooManyObjects

Three objects on the second image with two slots fail with `InvalidDescription`,
"second.png", "DetectionCollator", "3 objects" and "at most 2".

## DetectionRefusesAClassOutsideTheSet

An object of class 3 on the first image, over 3 classes, fails with `InvalidDescription`,
"first.png", "object 0", "class 3" and "[0, 3)".
