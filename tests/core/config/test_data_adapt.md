# tests/core/config/test_data_adapt.cpp

Suites: `Small_DataAdapt`.

Tests `adapt_data_to_model` of `modules/core/config/include/SushiAI/config/data_adapt.hpp`, which
passes a data block through `fit_data_to_model` and then `normalise_data_to_model`.

## AModelStatingNeitherLeavesTheBlock

A default `ModelSpec` leaves the two-key block with two keys.

## ABlockTakesTheFitThenTheNormalisation

A model with a `resize` fit and the default normalisation gives four keys: the block's own two,
then `fit`, then `normalise`.

## ABlockStatingAnotherFitIsRefused

A block with the `letterbox` fit against a model fitting by `resize` fails with
InvalidDescription naming `'model.json' fits its images by 'resize'` and `another 'fit'`.

## ABlockStatingAnotherNormalisationIsRefused

A block whose `normalise` states a deviation of 2, 4 and 8 against the model's default fails with
InvalidDescription naming `'model.json' states a 'normalise'` and `states another`.
