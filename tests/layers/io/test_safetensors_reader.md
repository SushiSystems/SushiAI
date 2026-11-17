# tests/layers/io/test_safetensors_reader.cpp

Suites: `Small_SafetensorsReader`.

Tests `IO::SafetensorsReader`, in `modules/layers/io/include/SushiAI/io/safetensors_reader.hpp`, on
files written by hand in memory with `file_of`: the 8-byte little-endian header length, the JSON
header, then the data. Every expected offset, length and value is worked out from the bytes the
test writes, so no case reads a file from disk.

## ListsEveryTensorInTheOrderOfItsBytes

A 24-byte data section holds four tensors: `b`, F32 [1, 2], at [0, 8); `a`, F16 [3], at [8, 14); `c`,
BF16 [1], at [14, 16); and `d`, F64 of rank 0, at [16, 24). The header spells `a` first, then `d`,
`b` and `c`, and holds a `__metadata__` entry. The reader lists `b`, `a`, `c`, `d`, the order of
their bytes, with the metadata skipped. Each offset is the 8 bytes of the length field plus the
header length plus the range start, and each length is the range width. The elements read as
float32 are 1.0 and -2.0 for `b`, 1.0, 0.5 and -2.0 for `a` (binary16 `3C00`, `3800`, `C000`), 1.0
for `c` (bfloat16 `3F80`) and 0.5 for `d` (binary64 `3FE0000000000000`); every one is exact, so
the comparison is equality.

## HandsOutTheBytesOfARangeWithoutCopyingThem

`contents` of the one tensor returns a view whose pointer is the address of the file's data start
in the caller's vector, and whose size is the range width.

## RefusesAFileShorterThanTheLengthField

A 3-byte file fails with `InvalidFile`, naming its 3 bytes and the header.

## RefusesAHeaderLongerThanTheFile

A length field of 40 over a file with 2 bytes after it fails with `InvalidFile`: "the header states
40 bytes, and the file holds 2".

## RefusesAHeaderThatIsNoJsonObject

The header `[1,2]` fails with `InvalidFile`, "the header is no JSON object". The header `{"w":`
fails with `InvalidFile`, naming "fixture.safetensors header" as the origin of the parse failure.

## RefusesARangeOutsideTheData

A tensor `w` of 5 F32 elements at [0, 20) over 16 data bytes fails with `InvalidFile`, naming
`w`, the range and the 16 data bytes.

## RefusesAnUnknownDtype

A tensor `w` of dtype `I8` fails with `Unsupported`, naming `w` and `I8`.

## RefusesOverlappingRanges

`x` at [0, 8) and `y` at [4, 12) fail with `InvalidFile`: "tensors 'x' and 'y' overlap", the first
ending at byte 8 and the second starting at 4.

## RefusesARangeThatRunsBackwards

A tensor `w` at [8, 4) fails with `InvalidFile`, naming `w` and the range.

## RefusesARangeThatDoesNotMatchItsShape

Three F32 elements need 12 bytes and the range [0, 8) holds 8; the failure is `InvalidFile`, naming
`w` and both counts.

## RefusesAnEntryThatIsMissingAFieldOrHoldsABadExtent

An entry with no `data_offsets` fails with `InvalidFile`, naming `w` and the field. A shape [0]
fails with `InvalidFile`, naming `w` and the extent 0.
