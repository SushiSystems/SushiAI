# tests/vision/darknet_forward_cases/test_darknet_forward.cpp


Suites: `Small_DarknetForward`, `Small_DarknetForwardLookup`, `Medium_DarknetForwardTable` and
`Medium_DarknetForwardFiles`.

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

## ALookupReturnsTheRowOfItsStem

One row per stem of the parity table. `find_darknet_parity_model` returns the row of that stem,
with a positive side and at least one [yolo] layer. The table values are not pasted here; the
next case checks them against the cfg files.

## ARowNamesTheSideAndTheYoloLayersItsCfgSpells

`Medium_`, one row per stem. The cfg file of the stem is loaded, and the row's side equals the
`width` of its [net] section and the row's [yolo] layers equal the indices of the cfg's [yolo]
sections. The case skips when the Darknet files are absent.

## ALayerNamePadsItsIndexToThreeDigits

`name_darknet_layer` gives "layer_016_yolo" and "layer_000_yolo_input", and keeps all four
digits of 1234.

## AnObservedLayerIsNamedByItsIndex

`name_observed_layer` of 7 is "layer.7".

## TheGapCountsValuesBeyondAnAbsoluteBound

The values 1, 2.5 and 101 against 1, 2 and 100 differ by 0, 0.5 and 1. Under an absolute bound
of 0.1 two values are outside and the largest is at element 2.

## TheGapKeepsTheLargestWithinAnAbsolutePlusRelativeBound

The same values under 0.6 plus 0.02 of the reference have none outside, and the largest gap is
exactly 1.

## ANaNIsOutsideEveryBound

A NaN against 2 is outside a bound of 1 plus 1 of the reference and is reported at element 1,
ahead of the gap of 0.5 at element 0.

## TheToleranceCheckPassesTwoEqualSpans

`within_tolerance` passes two equal spans at a bound of 0.

## TheToleranceCheckFailsOnASizeMismatch

`within_tolerance` fails a span of one value against two.

## TheToleranceCheckFailsOnAValueOutsideTheBound

`within_tolerance` fails a gap of 1 under a bound of 0.5.

## ALogitHeadIsActivatedAndStretched

A logit slot of ln 3, 0, 0.3, -0.2, 0 and ln 3 at `scale_xy` 2 gives 1, 0.5, 0.3, -0.2, 0.5 and
0.75: x and y take the logistic and then `v * 2 - 0.5`, the sizes stay, and the objectness and
the class take the logistic. Values are compared within `K_YOLO_TOLERANCE`.

## AnActivatedHeadIsOnlyStretched

Two activated slots at `scale_xy` 2 change only at x and y: 0.75 and 0.25 become 1 and 0, and
0.5 and 0 become 0.5 and -0.5.

## AnOutputIsFoundByItsName

A run that published "layer.3" returns it from `find_output`, and "layer.4" gives null.

## AModelWithoutFilesIsReported

For a stem no reference data directory holds, `find_missing_darknet_file` returns a reason that
names the stem. It holds whether `SA_REFERENCE_DATA_DIR` is set or not.

## ALoadOfAStemWithoutACfgIsFileNotFound

`Medium_`, since the load looks for the file. `load_darknet_config_file` of that stem fails with
FileNotFound naming its cfg.
