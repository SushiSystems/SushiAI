# modules/vision/datasets/include/SushiAI/vision/datasets/byte_sink.hpp


`ByteSink` is the port a writer writes its files through, beside the `ByteSource` it reads. Today
one writer takes it: `write_sample_cache`. A null sink is the file system.

| Call | Does |
| --- | --- |
| `write(file, bytes, reader)` | Writes a file, makes the folders above it and replaces a file of that name; `InvalidFile` opening with the path when it cannot |

`MemorySink` is the in-memory adapter. It is made over a `MemorySource` and adds each file to it, so
a `ByteSource` over the same `MemorySource` reads the file back. The file system adapter is
`Detail::FileSystemSink` in `source/disk.hpp`; it writes `<name>.partial` and renames it, so a file
under its final name is whole.
