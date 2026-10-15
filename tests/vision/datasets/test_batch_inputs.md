# tests/vision/datasets: test_batch_inputs.cpp, test_classification_collator.cpp, test_detection_collator.cpp


The scaled inputs and the roles the classification and detection collators write, and the batches they refuse. Needs an engine.

Suites: `Unit_DatasetsCollators`.

Tests `write_batch_inputs`, `ClassificationCollator` and `DetectionCollator` in
`modules/vision/datasets/include/SushiAI/vision/datasets/`. The fixture is an `EngineTest`. It
holds two gray samples of one row and two columns, the first of level 51 from `first.png` and
the second of level 255 from `crowded.png`, and allocates batches of two such images.

## InputsAreBytesDividedBy255

`write_batch_inputs` writes 0.2, 0.2, 1.0, 1.0, each within 1e-6.

## AnImageOfAnotherSizeIsRefused

With the second image 2 by 2, `write_batch_inputs` fails with `Incompatible`,
"write_batch_inputs", "image 1" and "2 by 2".

## ATensorOfAnotherImageCountIsRefused

With one sample and a tensor for two, `write_batch_inputs` fails with `Incompatible`,
"write_batch_inputs" and "holds 2 images".

## ClassificationPublishesLabels

`ClassificationCollator::make(0)` fails with `InvalidDescription`, "ClassificationCollator" and
"0 classes". A collator of 7 classes reports the task `K_TASK_CLASSIFICATION` and one target:
the role `K_ROLE_LABELS`, an empty sample shape, `Int32` and 7 classes.

## ClassificationWritesTheClassOfEachSample

With the class indices 6 and 0, `collate` writes the labels 6 and 0, and the third input value
is 1.0.

## ClassificationNamesTheFileOfASampleWithoutAClass

With no class index on the second sample, `collate` fails with `InvalidDescription`,
"crowded.png", "ClassificationCollator" and "no class index". With the class index 7 over 7
classes it fails with "crowded.png", "class index 7" and "[0, 7)".

## ABatchWithoutTheRoleIsRefused

Over a batch allocated without targets, `collate` fails with `Incompatible`,
"ClassificationCollator" and "no role 'labels'".

## DetectionPublishesThreeRoles

`DetectionCollator::make` fails with `InvalidDescription` for 0 classes ("0 classes") and for
0 objects ("0 objects"). A collator of 3 classes and 4 objects reports the task
`K_TASK_DETECTION` and three targets: `K_ROLE_BOXES` of shape [4, 4] and `Float32`,
`K_ROLE_CLASSES` of shape [4], `Int32` and 3 classes, and `K_ROLE_VALID` of shape [4] and
`Float32`.

## DetectionWritesCornersAndPadsWithZero

Every target slot is first filled with 9. With one object of class 2 and the box (0.25, 0.5,
1.5, 1.0) on the second image and two slots per image, `collate` writes that box, the class 2
and the valid flag 1 into the third slot and zero into every other slot.

## DetectionNamesTheFileOfAnImageWithTooManyObjects

Three objects on the second image with two slots fail with `InvalidDescription`,
"crowded.png", "DetectionCollator", "3 objects" and "at most 2".

## DetectionRefusesAClassOutsideTheSet

An object of class 3 on the first image, over 3 classes, fails with `InvalidDescription`,
"first.png", "object 0", "class 3" and "[0, 3)".
