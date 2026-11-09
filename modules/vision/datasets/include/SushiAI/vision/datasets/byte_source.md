# modules/vision/datasets/include/SushiAI/vision/datasets/byte_source.hpp


`ByteSource` is the port a format reader reads its files and folders through. Each reader has two
forms, `read_x(options)` and `read_x(options, source)`; the first passes the file system, so a
path-only caller sees no change. A reader that lists a folder or opens several files takes the
port in place of a byte span, because one call does not hold what it needs.

| Call | Returns |
| --- | --- |
| `kind_of(path)` | `PathKind::Missing`, `File`, `Directory` or `Other` |
| `size_of(file, reader)` | The byte count of a regular file |
| `modified_of(file, reader)` | When a regular file last changed, in the source's own ticks; `MemorySource` gives 0 |
| `read(file, reader)` | Every byte of a regular file |
| `read_range(file, offset, count, reader)` | Up to `count` bytes from `offset`; fewer when the file ends first |
| `list(directory, kind, depth, reader)` | The files or folders in or below a directory, sorted by path |

`reader` names the calling reader in a message. The statuses are those the file system form of the
same call returns: `FileNotFound` for a path that names nothing, `InvalidFile` for a path that
names something else or cannot be read.

`MemorySource` is the in-memory adapter. `add_file(path, bytes)` adds a file and `add_directory(path)`
a folder that holds none; a folder also exists when a file lies below it. Paths are compared as
`std::filesystem::path` compares them, so a test writes the root it passes in the options. The
file system adapter is `Detail::FileSystemSource` in `source/disk.hpp`.

`read_coco_json` has no `ByteSource` form. It streams its file through `Config::read_json_events`,
which takes a path only.
