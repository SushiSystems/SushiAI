# codecs

## Owns

The decoding of an image file into an `Imaging::Image`: the detection of the format from the
leading bytes, a JPEG decoder over libjpeg-turbo, a PNG decoder over libpng, and the dispatch
that picks one of them or the raw image reader of `vision/imaging`. Everything is in namespace
`SushiAI::Vision::Codecs`. The design is `docs/design/DATASETS.md`, section "Image decoding".

Every function works on bytes in host memory and returns `uint8` pixels in the layout of
`vision/imaging`: row-major, channels innermost, RGB with red at channel 0. Nothing here reaches
an engine or SYCL. A failure is a `Status` inside a `Result`; no exception leaves the module, and
no libjpeg call ends the process.

## Depends on

- `vision/imaging` and `core` (`PUBLIC_DEPENDS`).
- libjpeg-turbo and libpng (`EXTERNAL_PRIVATE_LINKS`, targets `JPEG::JPEG` and `PNG::PNG`).
  `source/jpeg_decoder.cpp` includes `jpeglib.h` and `jerror.h`; `source/png_decoder.cpp` includes
  `png.h`. No public header names a type of either library. `JCS_EXT_RGB` is a libjpeg-turbo
  extension, so the JPEG decoder does not build against the IJG library.

## Public entry points

| Header | Declares |
| --- | --- |
| `<SushiAI/vision/codecs/decode_options.hpp>` | `ChannelLayout`, `PaletteRead`, `DecodeOptions` |
| `<SushiAI/vision/codecs/image_format.hpp>` | `ImageFormat`, `detect_image_format` |
| `<SushiAI/vision/codecs/jpeg_decoder.hpp>` | `decode_jpeg` |
| `<SushiAI/vision/codecs/png_decoder.hpp>` | `decode_png` |
| `<SushiAI/vision/codecs/decode_image.hpp>` | `decode_image`, `load_image` |

`DecodeOptions` has a header of its own, with no source, so that the two decoders take it without
including the dispatcher.

### DecodeOptions

| Field | Default | Meaning |
| --- | --- | --- |
| `layout` | `ChannelLayout::Stored` | `Stored` gives one channel for a gray file and three for any other; `Gray` gives one; `Color` gives three |
| `palette` | `PaletteRead::Colors` | `Indices` makes a palette PNG yield its stored indices in one channel, whatever `layout` says. A file without a palette ignores the field |

An image has one channel or three. Alpha is never returned, because the pipelines of
`vision/imaging` refuse two and four channels.

### detect_image_format

| Format | Leading bytes |
| --- | --- |
| `JPEG` | `FF D8 FF` |
| `PNG` | `89 50 4E 47 0D 0A 1A 0A` |
| `Raw` | `Imaging::K_RAW_IMAGE_MAGIC`, least significant byte first: `S`, `A`, `I`, `M` |

The signatures are one table in `source/image_format.cpp`. Bytes that end inside a signature are
`Unknown`. The file name plays no part.

### decode_jpeg

| File stores | `Stored` | `Gray` | `Color` |
| --- | --- | --- | --- |
| Grayscale | 1 channel | 1 channel | 3 equal channels |
| YCbCr, RGB | RGB | 1 channel, converted by the library | RGB |
| CMYK, YCCK | RGB | `Unsupported` | RGB |

The library converts YCCK to CMYK and the decoder converts CMYK to RGB with no colour profile:
each channel is `(ink * black + 127) / 255`, where `ink` and `black` are the stored samples when
the file has an Adobe marker, which stores 255 for no ink, and `255 - sample` otherwise. Any other
colour space is `Unsupported`, with libjpeg's number for it. The rules are one table,
`K_COLOR_SPACES`, in `source/jpeg_decoder.cpp`.

libjpeg reports a fatal error by calling a handler that must not return. The decoder's handler
copies the library's message and jumps back with `longjmp`. `run_step` is the only function that
calls `setjmp`; it and the four steps it runs hold pointers and integers alone, so the jump
unwinds libjpeg's frames and no C++ object.

A file that makes libjpeg emit a corrupt-data warning is `InvalidFile` with the text of the first
warning. That covers a file that ends early, which the library would otherwise pad with gray
rows, and a file whose only fault is a missing end marker or stray bytes before a marker.

### decode_png

The decoder uses libpng's simplified read interface, which reports an error through its return
value and needs no `setjmp`. It asks the library for one of five formats:

| File stores | Wanted | Format asked for | Then |
| --- | --- | --- | --- |
| No alpha | 1 channel | `PNG_FORMAT_GRAY` | Read into the image |
| No alpha | 3 channels | `PNG_FORMAT_RGB` | Read into the image |
| Alpha, or a `tRNS` chunk | 1 channel | `PNG_FORMAT_GA` | Read into a buffer; the gray byte of each pixel is copied |
| Alpha, or a `tRNS` chunk | 3 channels | `PNG_FORMAT_RGBA` | Read into a buffer; the three colour bytes of each pixel are copied |
| Palette, `PaletteRead::Indices` | Indices | `PNG_FORMAT_RGBA_COLORMAP` | The indices are read into the image; the colour map is discarded |

A format without alpha would make the library composite a file's colours on a background, so a
file with alpha is read with it and the decoder drops the channel. The colour-map format carries
alpha for the same reason: without it the library asks for a background and composites the
transparent palette entries.

The library expands a palette and samples below 8 bits, reduces 16-bit samples to 8, replicates
gray into RGB, and converts RGB to gray with its own coefficients. A pixel whose three channels
are equal keeps that level in the conversion.

The decoder sets `PNG_IMAGE_FLAG_16BIT_sRGB`: a 16-bit file with no `gAMA` and no `sRGB` chunk is
scaled to 8 bits as it is. Without the flag the library treats such a file as linear and encodes
it to sRGB. A file that carries a `gAMA` chunk other than sRGB's is converted to sRGB by the
library at any bit depth, so its 8-bit levels are not the stored ones. Palette indices are never
converted.

### decode_image and load_image

`decode_image` calls `detect_image_format` and looks the format up in `K_DECODERS`, one table of
format and function in `source/decode_image.cpp`. A new format is a decoder and a row.

A raw image file is read through `read_raw_image_header` and `read_raw_image_pixels` over the
bytes in place. The byte count must equal the header plus the pixels the header declares, and
that is checked before the pixels are allocated. A gray file read as `Color` goes through
`Imaging::expand_gray`. A three-channel file read as `Gray` is `Unsupported`: `vision/imaging` has
no RGB to gray conversion.

`load_image` reads the whole file and calls `decode_image`. Every failure message opens with the
path in quotes, written with forward slashes.

### Statuses

| Code | When |
| --- | --- |
| `InvalidFile` | A decoder rejects the bytes; the message carries the library's text. The raw reader rejects the file. `load_image` is given a directory, or cannot open or read the file |
| `Unsupported` | No signature matches. A JPEG colour space or conversion outside the table. A JPEG or PNG side outside [1, `Imaging::K_MAX_IMAGE_SIDE`], checked before the pixels are allocated. A three-channel raw image asked for as gray |
| `FileNotFound` | `load_image` is given a path that names nothing |
| `OutOfMemory` | An allocation of the module fails, or libjpeg reports `JERR_OUT_OF_MEMORY` |

A raw image file with a side above the limit is `InvalidFile`, because its own format bounds the
field. libpng reports an allocation failure as text, so it comes out as `InvalidFile`.

A message opens with the function that failed: `decode_jpeg: `, `decode_png: `, `decode_image: `
or, after the path, `load_image: `.

## Known limits

- A PNG cut after its last row of pixel data can decode: the simplified interface stops reading
  there and does not check the end of the stream.
- CMYK to RGB uses no ICC profile.
- An embedded ICC profile and EXIF orientation are not applied in either format.
- Nothing in this module has run. The sources and the tests pass `clang -fsyntax-only`; no test
  has been executed.

## Tests

`modules/vision/codecs/tests/`, compiled into `sushiai_vision_tests`. No test needs a device and
none reads a file from the repository. Each builds its input in memory: PNG files with
`png_image_write_to_memory`, JPEG files with libjpeg's compressor at quality 100 and no
subsampling, raw image files with `Imaging::write_raw_image`. The files `load_image` reads are
written under the system's temporary directory and removed.

| File | Suite | Covers |
| --- | --- | --- |
| `test_codecs_image_format.cpp` | `Unit_CodecsImageFormat` | Each signature, signatures cut short or changed by one byte, garbage and no bytes |
| `test_codecs_png.cpp` | `Unit_CodecsPng` | Gray, RGB, RGBA, gray-alpha, palette and 16-bit files compared exactly; each layout; palette indices; cut and damaged files; the side limit |
| `test_codecs_jpeg.cpp` | `Unit_CodecsJpeg` | Gray, YCbCr, RGB, CMYK and YCCK files within 3 levels per channel; each layout; every proper prefix of a file; the side limit |
| `test_codecs_decode_image.cpp` | `Unit_CodecsDecodeImage` | The dispatch to each decoder, the raw image path and its refusals, unknown bytes, and `load_image` on a missing path, a directory and files of each format |

The JPEG tolerance is 3 levels, absolute, per channel. The test images are blocks of 8 by 8
pixels of one colour, which the transform reproduces exactly at quality 100, so the difference is
the rounding of the colour conversion.

The 16-bit PNG cases use the samples 0 and 65535 alone. The simplified writer marks 16-bit samples
as linear, and the reader then encodes every level between the two ends to sRGB. The path of a
16-bit file without a `gAMA` chunk has no test, because that writer cannot produce one.
