# tests/layers/io/test_safetensors_weight_source.cpp

Suites: `Small_SafetensorsWeightSource`.

Tests `is_safetensors_file` and `open_safetensors_source` of `safetensors_weight_source.hpp`. The
file is built in memory by `weight_fixtures.hpp`: the 8-byte header length, a JSON header and 32 data
bytes of the two tensors `net.fc.weight` ([2, 3], 1 to 6) and `net.fc.bias` ([2], 0.5 and -0.5). Every
value is a small whole number or a binary fraction, so each float32 is exact and compared with `==`.

## TheSourceListsTheTensorsOfTheFileWithTheirValues

The source lists `net.fc.weight` with the stored type `F32` and the shape [2, 3], then
`net.fc.bias` with the shape [2]; it reports the origin the options state and reads the two tensors
as the values written.

## ASafetensorsFileNeedsAFittingLengthAndAnOpeningBrace

A length of 1000 over a 10-byte file, a header that opens with `[`, and a file of the 8 length bytes
alone are each refused.

## SafetensorsIgnoresTheStateKey

The safetensors file opens with the key `model` set and lists its two tensors.
