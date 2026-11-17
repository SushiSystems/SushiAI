# tests/vision/training/test_pixel_extents.cpp


Suites: `Small_PixelExtents`.

Tests the largest `PixelExtents` the per-pixel objectives accept. A pixel objective reads one label
per pixel through `K_TAKE_ALONG`, so no product of the extents is an index and none is bounded.
No case needs a device.

### MakeAcceptsAProductOfExtentsPastInt32

A batch of 54 images of 512 by 512 pixels over 150 classes holds 2 123 366 400 scores and one
image more holds 2 162 688 000, above 2147483647. `PixelCrossEntropyObjective::make` and
`DiceObjective::make` accept both. Both were refused at 55 while the loss gathered from a table of
`R * C` rows.
