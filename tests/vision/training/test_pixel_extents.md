# tests/vision/training/test_pixel_extents.cpp


Suites: `Small_PixelExtents`.

Tests the largest `PixelExtents` the per-pixel objectives accept. A pixel objective holds one score
per pixel and class and addresses it by an INT32 index, so batch times height times width times
classes must not exceed 2147483647. No case needs a device.

### MakeRefusesWhatInt32CannotIndex

A batch of 54 images of 512 by 512 pixels over 150 classes holds 2 123 366 400 scores and is
accepted by `PixelCrossEntropyObjective::make` and by `DiceObjective::make`. One image more holds
2 162 688 000 and both fail with `InvalidDescription` naming "55 * 512 * 512 * 150" and
"2147483647". The Dice objective gathers one probability per pixel, so the same bound holds for it.
