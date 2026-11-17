# tests/layers/io/test_weight_source.cpp

Suites: `Small_WeightSource`.

Tests the weight source seam, in `modules/layers/io/include/SushiAI/io/weight_source.hpp`, and its
two sources, `torch_weight_source.hpp` and `safetensors_weight_source.hpp`. The files are built in
memory by `weight_fixtures.hpp`: one small model of two tensors, `fc.weight` of shape [2, 3] holding
1 to 6 and `fc.bias` of shape [2] holding 0.5 and -0.5, stored under the names `net.fc.weight` and
`net.fc.bias`.

- `torch_fixture` is a zip archive whose `data.pkl` entry pickles `{"state_dict": {...}}` the way
  `torch.save` does, with one storage entry per tensor, and holds a third tensor,
  `net.bn.num_batches_tracked`, one int64 element.
- `safetensors_fixture` is the 8-byte header length, a JSON header and 32 data bytes. It cannot state
  the counter, since the container holds no int64 here, so it has the two tensors only.

Every value is a small whole number or a binary fraction, so each float32 is exact and compared with
`==`.

## EachMatcherAcceptsOnlyItsOwnContainer

`is_torch_file` accepts the zip and refuses the safetensors bytes; `is_safetensors_file` accepts the
safetensors bytes and refuses the zip, whose first 8 bytes read as a length no file holds.

## SniffingPicksTheSourceOfTheContainer

The registry holds the torch row, then the safetensors row. The zip opens as a source that lists
`net.fc.weight`, `net.fc.bias` and `net.bn.num_batches_tracked`, with stored types `float32` and
`int64`; the other file opens as a source that lists the first two, with the stored type `F32`. The
origin the options state is the origin of the source.

## BothSourcesReadTheSameShapesAndValues

The shapes of the first two tensors are equal in both sources and are [2, 3] and [2]. Both read the
weight as 1 to 6 and the bias as 0.5 and -0.5.

## RefusesBytesOfNeitherContainer

The text "hello, weights" fails with `InvalidFile`, naming the origin and both formats, `'torch'` and
`'safetensors'`.

## TheRegistryRefusesAFormatTwiceAndKeepsTheOrder

Adding the torch row again fails with `InvalidDescription` naming `'torch'`; the rows stay torch,
then safetensors.

The cases that open one container alone are in `test_torch_weight_source.cpp` and
`test_safetensors_weight_source.cpp`.
