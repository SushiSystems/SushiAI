# tests/core/config/test_data_normalise.cpp


Suites: `Small_DataNormalise`.

Tests `Config::normalise_data_to_model` on a data block of two keys and a `ModelSpec`. No case
needs a device.

## AModelWithoutANormalisationLeavesTheBlock

The block keeps its two keys and gains no `normalise`.

## ABlockWithoutANormalisationTakesTheModels

The block gains `normalise` as its third key, and the object reads back equal to the model's.

## ABlockStatingTheModelsNormalisationKeepsIt

A block that states an equal object keeps its three keys.

## ABlockStatingAnotherNormalisationIsRefused

A block that states the default under a model that swaps channels fails with `InvalidDescription`,
"'model.json' states a 'normalise'" and "states another".

## ABlockWithAMalformedNormalisationIsRefused

A block object with the key `gamma` fails with `InvalidDescription`, "the data block's
'normalise'" and "does not accept 'gamma'".
