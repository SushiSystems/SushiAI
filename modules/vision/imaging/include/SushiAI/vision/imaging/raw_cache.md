# modules/vision/imaging/include/SushiAI/vision/imaging/raw_cache.hpp


One file holds one decoded image. A separate tool decodes the dataset once and writes these
files; the library reads them and has no image decoder. The reference's counterpart is one
`.npy` per image at the original resolution (`data/base.py:337-345`). The tool is not part of
this module and is not written.

## Byte layout, version 1

Every multi-byte field is little-endian. The header is 24 bytes; no field is padded.

| Offset | Size | Field | Value |
| --- | --- | --- | --- |
| 0 | 4 | magic | the bytes `S`, `A`, `I`, `M` (`K_RAW_IMAGE_MAGIC` as a word) |
| 4 | 4 | `format_version` | unsigned, 1 |
| 8 | 4 | `height` | unsigned, in [1, 16384] |
| 12 | 4 | `width` | unsigned, in [1, 16384] |
| 16 | 4 | `channels` | unsigned, 1 or 3 |
| 20 | 4 | `channel_order` | unsigned: 0 `Gray`, 1 `RGB`, 2 `BGR` |
| 24 | height * width * channels | pixels | `uint8`, row-major, channels innermost |

`channels` is 1 exactly when `channel_order` is `Gray`. The file ends with the last pixel byte.
The extension is `.saim`.

| Version | Release | Change |
| --- | --- | --- |
| 1 | not released; planned for 0.1.0 | The layout above |

## Reader and writer

The writer serialises field by field and writes only the current version. The reader refuses a
version it does not know and names both versions in the error; there is no older version to
upgrade from yet.

`read_raw_image_header` allocates nothing and rejects every field outside its range.
`read_raw_image_file` also compares the file size with `24 + height * width * channels` before
it resets the destination, so a truncated or padded file costs no allocation. The stream form,
`read_raw_image_pixels`, cannot know a stream's length; its allocation is bounded by
`K_MAX_IMAGE_SIDE` and it throws when the stream ends early.

The reader returns the module's order. A `BGR` file has channels 0 and 2 swapped as it is read,
so a cache written from OpenCV-ordered pixels is usable without rewriting it. A `Gray` file is
read as one channel and stays one channel in the format; the detection pipelines expand it.
