# modules/vision/datasets/include/SushiAI/vision/datasets/idx_file.hpp


An IDX file opens with two zero bytes, an element code and a dimension count, then one 32-bit
extent per dimension with the most significant byte first, then the elements. `read_idx_header`
returns the element type, the extents and the byte count of the elements. It refuses a file
whose size is not the header plus those bytes, and extents whose product passes 64 bits.
`read_idx_payload` returns the element bytes as the file stores them; it converts no byte order.

| Element | Code | Bytes |
| --- | --- | --- |
| `UInt8` | `0x08` | 1 |
| `Int8` | `0x09` | 1 |
| `Int16` | `0x0B` | 2 |
| `Int32` | `0x0C` | 4 |
| `Float32` | `0x0D` | 4 |
| `Float64` | `0x0E` | 8 |

`source/mnist.cpp` still parses its own IDX headers. Its failure messages differ from those of
`read_idx_header`, and it reads both files inside one guarded allocation, so moving it would
change what a caller sees.

## Reading from memory

`read_idx_header(file, source)` and `read_idx_payload(file, header, source)` do what the
one-argument forms do, with the bytes read through a `ByteSource` (`byte_source.hpp`). The header
reads two ranges of the file, its four leading bytes and its extents, and the payload one range,
so no call reads the whole file.
