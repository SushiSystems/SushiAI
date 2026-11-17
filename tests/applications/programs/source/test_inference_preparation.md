# tests/applications/programs/source/test_inference_preparation.cpp

Suite: `Medium_InferencePreparation`, in `sushiai_programs_tests`.

Tests `prepare_inference` of `inference_preparation.hpp` over the `tiny_yolo` fixture, with a weight
file that no file stands behind.

## AWeightFileThatDoesNotExistRefusesTheBuild

A request with no hooks loads its state from the weight file, so the build is refused.

## TheFillHookRunsOnceInPlaceOfTheWeightLoad

A request with a fill hook that initialises the state builds with the missing weight file, and the
hook ran once.

## TheWrapHookServesTheDatasetItReturns

A request whose wrap hook hides the `fit` role builds a `PreparedInference` whose dataset lists no
`fit` target.
