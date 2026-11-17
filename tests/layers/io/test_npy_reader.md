# tests/layers/io/test_npy_reader.cpp

Suites: `Small_NpyReaderVersion`, `Small_NpyReaderType`, `Small_NpyReaderShape`,
`Small_NpyReaderRefusal`, `Small_NpyReader`.

Tests `IO::NpyReader`, in `modules/layers/io/include/SushiAI/io/npy_reader.hpp`. The files are
built in the test the way `numpy.lib.format` writes them: the magic, the version, the length, the
header dictionary padded with spaces and a line end to a multiple of 64 bytes, then the data. The
header of a float32 array of shape (2, 3) is 59 bytes, so its data starts at 128 in all three
versions (10 + 59 + 1 and 12 + 59 + 1 round up to 128). The values are small whole numbers and
binary fractions, so each float32 is exact and compared with `==`.

## ReadReturnsTheArrayOfEachFormatVersion

Rows over versions 1, 2 and 3 of a float32 array of shape (2, 3) holding 0 to 5. The type, the
extents, `fortran_order`, the offset 128, the length 24 and the six values are as written.

## ReadFloat32WidensEachElementType

Rows over each descriptor, with two elements written as raw words by hand: binary16 `0x3C00` and
`0xC000` are 1 and -2, the IEEE words of 1 and -2 for float32 and float64, `0xFE`, `0xFFFE`,
`0xFFFFFFFE` and `0xFFFFFFFFFFFFFFFE` are -2 in 8 to 64 bits, and the unsigned rows hold 200, 60000,
65536 and 2^40, which float32 holds exactly. The type name and size are as the descriptor gives.

## ReadFloat32TransposesAColumnMajorMatrix

The matrix [[0, 1, 2], [3, 4, 5]] stored column by column as 0, 3, 1, 4, 2, 5 with
`fortran_order` true reads as 0 to 5 row-major.

## ReadFloat32TransposesAColumnMajorCube

A (2, 3, 4) array whose element (i, j, k) is 100 i + 10 j + k, stored with i varying fastest, reads
in row-major order: the value is a formula, so the expected order does not come from the reader.

## ReadReturnsAScalarAsOneElement

The shape `()` has rank 0 and reads as one element, 7.

## ReadNamesTheExtentsOfEachShapeSpelling

Rows over `()`, `(3,)`, `(2, 3)` and `(2, 3, 4)`: the extents are as spelled and the data length is
the element count times 4.

## ReadAcceptsDoubleQuotesAndNoSpaces

The header `{"descr":"<f4","fortran_order":False,"shape":(2,)}` reads as the array it spells.

## ReadAllowsBytesAfterTheData

Eight bytes after the data of a one-element array are left alone: the length stays 4.

## ReadRefusesAFileAndNamesWhy

Rows over an object array `|O`, a big-endian `>f4`, an unread `<c8`, a structured descriptor,
format versions 4.0 and 1.1, a header without `shape`, an unknown key, a repeated key, text after
the dictionary, an unclosed string, a shape with no number, an extent of 0, and 20 bytes of data for
a shape that needs 24. Each fails with the code of its kind, naming the file and the thing at fault.

## ReadRefusesAFileWithoutTheMagic

Thirty-two bytes of `x` fail with `InvalidFile`, naming "magic".

## ReadRefusesAHeaderLongerThanTheFile

A file cut to 40 bytes whose header states 118 fails with `InvalidFile`, "header states".
