# tests/vision/datasets/test_idx_file.cpp


Suites: `Small_DatasetsIdxFile`, `Small_DatasetsIdxFileElementBytes`,
`Small_DatasetsIdxFileMalformed`, `Small_DatasetsIdxFileWrongSize` and `Medium_DatasetsIdxFile`.

Tests `idx_element_bytes`, `read_idx_header` and `read_idx_payload` in
`modules/vision/datasets/include/SushiAI/vision/datasets/idx_file.hpp`. The `Small_` cases read the
file `row.idx` from a `MemorySource`; the `Medium_` cases read files they write under the system's
temporary directory.

## ElementBytesFollowTheFormat

One row each: `idx_element_bytes` is 1 for `UInt8` and `Int8`, 2 for `Int16`, 4 for `Int32` and
`Float32`, and 8 for `Float64`.

## AHeaderGivesTheElementTheExtentsAndThePayloadSize

A file of element code 0x08 with the extents 2, 3, 4 and 24 payload bytes reads as `UInt8`,
those extents, a header of 16 bytes and a payload of 24 bytes.

## AnExtentIsReadMostSignificantByteFirst

A file with the single extent 258 reads that extent as 258.

## ThePayloadSizeCountsTheElementBytes

A file of element code 0x0C with the extent 3 reads as `Int32` with a payload of 12 bytes.

## ThePayloadIsEveryByteAfterTheHeader

`read_idx_payload` over a file of five bytes gives 7, 0, 9, 3, 1.

## AMalformedHeaderIsRefused

One row each: `read_idx_header` fails with `InvalidFile`, "row.idx" and the cause for a file of three
bytes ("leading bytes"), a non-zero second byte ("two zero bytes"), the element code 10 ("element
code 10"), a rank of 0 ("no dimension") and a second extent cut after one byte ("dimension 1 of
2").

## AFileOfAnotherSizeThanItsHeaderDeclaresIsRefused

One row each: a file with the extents 2, 2 and three payload bytes fails with `InvalidFile`,
"declares 4 bytes" and "holds 15 bytes"; a file with the extent 2 and three payload bytes fails with
"declares 2 bytes" and "holds 11 bytes".

## ExtentsThatOverflowAreRefused

A file of element code 0x0E with three extents of 0xFFFFFFFF fails with `InvalidFile`, "row.idx" and
"overflow".

## AnAbsentFileIsNotFound

`read_idx_header` of a path that does not exist fails with `FileNotFound` and "absent.idx". It
reaches the file system, so it is `Medium_`.

## AHeaderAndItsPayloadReadFromAFile

The file-system forms of `read_idx_header` and `read_idx_payload` read a 2 by 2 unsigned-byte file
from disk: extents 2 and 2, payload 1, 2, 3, 4.
