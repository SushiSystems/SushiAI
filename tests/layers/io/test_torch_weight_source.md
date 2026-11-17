# tests/layers/io/test_torch_weight_source.cpp

Suites: `Small_TorchWeightSource`.

Tests `is_torch_file` and `open_torch_source` of `torch_weight_source.hpp`. The file is built in
memory by `weight_fixtures.hpp`: a zip archive whose `data.pkl` entry pickles `{"state_dict": {...}}`
the way `torch.save` does, with one storage entry per tensor: `net.fc.weight` ([2, 3], 1 to 6),
`net.fc.bias` ([2], 0.5 and -0.5) and `net.bn.num_batches_tracked`, one int64 element. Every value
is a small whole number or a binary fraction, so each float32 is exact and compared with `==`.

## TheMatcherTakesAZipArchiveAndNoOtherBytes

`is_torch_file` accepts the zip and refuses the text "hello, weights".

## TheSourceListsOneTensorPerEntryInPickleOrder

With the key `state_dict` the source lists the three tensors in the order the pickle holds them,
with the stored types `float32` and `int64` and the shapes [2, 3] and [2]; it reports the origin the
options state and reads the weight and the bias as the values written.

## ACheckpointRootNeedsTheStateKey

Opened with no state key, the zip's root holds `state_dict`, which is no tensor. The failure is
`InvalidFile`; it names the entry and says the key goes in the name map.

## RefusesAStateKeyTheFileDoesNotHold

The key `model` over the same file fails with `InvalidFile`, naming the origin and the key.
