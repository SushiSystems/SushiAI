# tests/vision/training/test_pixel_targets.cpp


Suites: `Medium_VisionTrainingPixelTargets` and `Small_VisionTrainingPixelTargetsClasses`.

Tests `validate_pixel_labels`, `fill_pixel_targets` and the refusals of `fill_dice_targets`. The
cases allocate INT32 and float32 tensors on the CPU SYCL device through `EngineTest`, so the suite
is medium. A mask is six pixels over three classes with the ignore index 255. Every real leaf
starts at -7, so a leaf the fill skipped shows. The real leaves are compared for equality: each
expected value is a zero or a quarter, exact in float32.

`test_dice_targets.cpp` counts the kept pixels of each class for the Dice leaves.

### TheLabelCheckPassesClassesAndTheIgnoreIndex

A mask of the classes 0 to 2 and the ignore index passes.

### TheLabelCheckNamesAHighLabelsPixel

A label of 3 at pixel 3 fails with `Incompatible` naming "holds 3 at pixel 3", "below 3" and 255.

### TheLabelCheckNamesANegativeLabelsPixel

A label of -1 at pixel 4 fails with `Incompatible` naming "holds -1 at pixel 4".

### TheLabelCheckRefusesAnotherPixelCount

A six-pixel mask checked for seven fails with `Incompatible` naming "6 pixels" and "expects 7".

### TheLabelCheckRefusesAFloatMask

A float mask fails with `Incompatible` naming "mask is".

### AKeptPixelHoldsItsLabelAndSharesTheWeight

The mask 0, 2, 255, 1, 1, 255 gives the labels 0, 2, 255, 1, 1, 255 and the weights 0.25, 0.25, 0,
0.25, 0.25, 0. A kept pixel weighs 1 over the four kept pixels; an ignored pixel keeps the ignore
index, which names no class, with a weight of 0.

### AMaskWithNoKeptPixelGivesZeroWeights

A mask of ignored pixels alone gives six labels of 255 and six weights of 0.

### ARefusedPixelFillChangesNothing

A label of 9, a weight leaf of five pixels and a real tensor given as the indices each fail with
`Incompatible` ("6 pixel labels", "5 pixel weights", "pixel labels are"), and after the three calls
the labels are still -7 and the weights still -7.

### AClassCountBelowOneIsRefusedWithoutReadingTheMask

Suite `Small_VisionTrainingPixelTargetsClasses`. Mask, label and weight tensors that are headers
alone, each claiming six elements and holding no storage, give `fill_pixel_targets` over 0 classes
an `Incompatible` failure naming "at least one class" and "got 0". A refusal that read a label
would dereference the missing storage.

### ARefusedDiceFillChangesNothing

A label of 3, a `class_weights` leaf of two elements and a real `labels` leaf each fail with
`Incompatible`, and after the three calls `labels`, `kept` and `class_pixels` are still -7.
