# tests/vision/datasets/test_byte_source.cpp


Suite: `Small_DatasetsByteSource`.

Tests `MemorySource` in `modules/vision/datasets/include/SushiAI/vision/datasets/byte_source.hpp`.
Each case builds a source that holds file A, four bytes in the folder `data`, and file B, two
bytes in the folder below it, and asserts what the source answers. Nothing touches the file
system.

## AStoredFileReadsBack

`read` of file A gives `abcd`, and `read_range` from 1 for 2 bytes gives `bc`.

## KindAndSizeAnswerForWhatWasAdded

`kind_of` is `File` for file A, `Directory` for `data` and for an added empty folder, and
`Missing` for an absent file; `size_of` of file A is 4.

## ListNamesTheEntriesOfAFolder

Listing the files of `data` gives file A at depth `Children`, and file A and file B at depth
`Tree`.

## RemoveDropsTheFileItNames

`remove` of file A is true and `kind_of` of it is then `Missing`, while file B is still a `File`.
A second `remove` of the same path is false.

## AnAbsentPathIsRefused

`read` of an absent file fails with `FileNotFound`, "absent.bin" and the reader name; `size_of`
of the folder `data` fails with `InvalidFile`; `list` of `nowhere` fails with `FileNotFound`,
"nowhere" and the reader name.
