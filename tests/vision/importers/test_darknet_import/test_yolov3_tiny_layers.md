# tests/vision/importers/test_darknet_import/test_yolov3_tiny_layers.cpp


Suites: `Medium_DarknetYolov3TinyLayers`, `Medium_DarknetLayers`, in
`sushiai_vision_tests`. Every case reads a `.cfg` or a dump from the directory `SA_REFERENCE_DATA_DIR`
names, so none is `Small_`.

The file compares the body of a model layer by layer. It covers two models that have a dump of every
layer: `yolov3-tiny` in the run `yolov3_tiny` and `yolov7-tiny` in the run
`yolov7_tiny_layers_plain_pool`. The fixture loads the `.cfg` from the directory
`SA_REFERENCE_DATA_DIR` names and skips, naming the model and the first missing file, when the cfg,
the weights, the input or a layer dump is absent. A [yolo] layer is not compared here; the forward
test of its model covers it.

## TheSectionsAreThoseOfTheDumps

`yolov3-tiny.cfg` has the 24 sections the file lists, in that order, compared as two vectors of
section types.

## TheYoloSectionsAreThoseOfTheParityTable

The [yolo] sections of the cfg are layers 16 and 23, the `yolo_layers` of the parity table.

## TheTranslationHoldsOneRowPerSectionAndTheHeads

The folded translation has 24 entries in `row_of_layer`, and its heads sit at the layers of the
cfg's [yolo] sections.

## AOneLayerRouteSharesTheRowOfItsSource

Layer 17, a one-layer route, has the row of layer 13 (SPEC 2.1).

## TheLayersAreReadAsTheCfgGivesThem

The layers read are the network input for layer 0, layer 13 for layer 17, and layers 19 and 8 for
layer 20.

## EveryLayerMatchesItsDumpWithinTolerance

Parameterised over the two models. The folded model is run with every layer that is not a [yolo]
section observed, and each output is compared with its dump within the layer bounds: 32768 times the float unit roundoff absolute plus 2048 times it relative.
Every layer prints its index, type, extents and largest difference. A failure names the first layer
outside the bound, the layers that layer reads, and how many later layers are outside it too.
