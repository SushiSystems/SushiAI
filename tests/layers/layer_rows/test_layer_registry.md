# tests/layers/layer_rows/test_layer_registry.cpp


Suites: `Small_LayerRowsRegistry`.

Tests `make_layer_registry` and `types_of` in
`modules/layers/layer_rows/include/SushiAI/layer_rows/layer_registry.hpp`. The rows themselves
are tested in `tests/layers/nn/test_layer_registry.cpp`. No case needs a device.

## EachCallBuildsAFreshRegistryOfTheLayerRows

Two calls return registries that both hold the "linear" row and no "no_such_layer" row, have the
same number of layer rows and do not share their storage.

## TypesOfListsEveryRowInOrder

Three hand-made layer rows yield "first", "second", "third", and an empty span yields an empty
list.

## TheListedTypesMatchTheRegistryRows

`types_of` over the built registry lists one type per layer row, and `find_layer` finds each.
