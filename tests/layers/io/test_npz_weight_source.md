# tests/layers/io/test_npz_weight_source.cpp

Suites: `Small_NpzWeightSource`.

Tests the `.npz` weight source, `modules/layers/io/include/SushiAI/io/npz_weight_source.hpp`. The
files are built in memory: each tensor of the one small model of `weight_fixtures.hpp`
(`net.fc.weight` of shape [2, 3] holding 1 to 6, `net.fc.bias` of shape [2] holding 0.5 and -0.5)
becomes a version 1.0 `.npy` file laid out as `numpy.save` does, and `zip_of` packs them as
`zipfile.ZipFile(f, "w").writestr("<name>.npy", data)` would. The deflated form holds the same bytes
in one stored DEFLATE block, with the CRC-32 computed in the test. Every value is exact in float32
and compared with `==`.

## StoredAndDeflatedFilesReadTheFixtureModel

Both forms list `net.fc.weight` then `net.fc.bias` (the entry names without `.npy`), with shapes
[2, 3] and [2], the stored type `<f4` and the origin the options state, and read the values of the
fixture.

## SniffingPicksNpzForAnNpzAndTorchForAPth

With the npz row added before the torch row, both matchers hold as expected: `is_npz_file` accepts
both npz forms and refuses the torch fixture, which holds a pickle entry named `data.pkl`; `is_torch_file` still accepts
the torch fixture. Opened through the registry, the npz reads as `<f4` and the torch file as
`float32`, with the same weight shape and values.

## AZipWithAnyOtherEntryOrNoEntryIsNoNpz

A zip with one `.npy` entry and one `data.pkl` entry, a zip with no entry and the text "hello, weights"
are each refused by `is_npz_file`.

## RefusesAnEntryThatIsNoNpyFile

An entry named `.npy` whose bytes are text fails with `InvalidFile`, naming the entry.
