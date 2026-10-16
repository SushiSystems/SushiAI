# tests/vision/darknet_forward_cases/test_darknet_forward.cpp


Suites: `Small_DarknetForward`.

Tests the helpers of
`modules/vision/darknet_forward_cases/include/SushiAI/vision/darknet_forward_cases/darknet_forward.hpp`
that need neither a model file nor a reference dump: the parity table, the layer names, the
tolerance measure, the [yolo] output and the output lookup. The forward pass and the four parity
checks need the Darknet files of `SA_REFERENCE_DATA_DIR`; the suites of
`tests/vision/importers/test_darknet_import/` run them and skip when the files are absent.
`translate_darknet_text` and `sources_of_row` are tested in `test_darknet_model.cpp` of the same
folder.

## one_class

`one_class` returns head options of one class, so a slot holds six entries, under a given coding
and `scale_xy`.

## TheParityTableHoldsSixModelsWithRunsAndHeads

`list_darknet_parity_models` returns six rows with distinct stems, and every row has a positive
side, at least one run and at least one [yolo] layer.

## ALookupReturnsTheRowOfItsStem

`find_darknet_parity_model` of "yolov3-tiny" returns side 416 and the [yolo] layers 16 and 23,
and that of "yolov4" side 608.

## ALayerNamePadsItsIndexToThreeDigits

`name_darknet_layer` gives "layer_016_yolo" and "layer_000_yolo_input", and keeps all four
digits of 1234. `name_observed_layer` of 7 is "layer.7".

## TheGapCountsValuesBeyondTheBoundAndKeepsTheLargest

The values 1, 2.5 and 101 against 1, 2 and 100 differ by 0, 0.5 and 1. Under an absolute bound
of 0.1 two values are outside and the largest is at element 2. Under 0.6 plus 0.02 of the
reference none is outside and the largest gap is 1.

## ANaNIsOutsideEveryBound

A NaN against 2 is outside a bound of 1 plus 1 of the reference and is reported at element 1,
ahead of the gap of 0.5 at element 0.

## TheToleranceCheckFailsOnASizeMismatchOrAValueOutside

`within_tolerance` passes two equal spans at a bound of 0, fails a span of one value against two,
and fails a gap of 1 under a bound of 0.5.

## ALogitHeadIsActivatedAndStretched

A logit slot of ln 3, 0, 0.3, -0.2, 0 and ln 3 at `scale_xy` 2 gives 1, 0.5, 0.3, -0.2, 0.5 and
0.75: x and y take the logistic and then `v * 2 - 0.5`, the sizes stay, and the objectness and
the class take the logistic. Values are compared within 1e-6.

## AnActivatedHeadIsOnlyStretched

Two activated slots at `scale_xy` 2 change only at x and y: 0.75 and 0.25 become 1 and 0, and
0.5 and 0 become 0.5 and -0.5.

## AnOutputIsFoundByItsName

A run that published "layer.3" returns it from `find_output`, and "layer.4" gives null.

## AModelWithoutFilesIsReportedAndRefused

For a stem no reference data directory holds, `find_missing_darknet_file` returns a reason that
names the stem, and `load_darknet_config_file` fails with FileNotFound naming its cfg. Both hold
whether `SA_REFERENCE_DATA_DIR` is set or not.
