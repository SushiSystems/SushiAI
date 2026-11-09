# tests/vision/datasets/test_byte_sink.cpp


Suite: `Small_DatasetsByteSink`.

Tests `MemorySink` in `modules/vision/datasets/include/SushiAI/vision/datasets/byte_sink.hpp`. Each
case builds a `MemorySource`, a `MemorySink` over it, and reads through the source what the sink
wrote. The file system sink writes files and has no case here.

## AWrittenFileReadsBackFromTheSource

After a write of `abcd` to a file in the folder `cache`, `read` of that file on the source gives
`abcd` and `kind_of` of the folder is `PathKind::Directory`.

## ASecondWriteReplacesTheFirst

After a write of `abcd` and a write of `xy` to one path, `size_of` gives 2.
