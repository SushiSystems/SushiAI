# tests/vision/imaging/test_raw_cache.cpp


Suites: `Small_ImagingRawCache`, `Medium_ImagingRawCache` and the parameterised `Small_ImagingRawCacheRoundTrip`, `Small_ImagingRawCacheVersion`, `Small_ImagingRawCacheShortHeader`, `Small_ImagingRawCacheExtent` and `Small_ImagingRawCacheChannels`.

Tests the raw image file: its bytes, its header checks and its file reader and writer.

This file needs no device. The `Small_` suites cover
`modules/vision/imaging/include/SushiAI/vision/imaging/raw_cache.hpp`. A file is a 24-byte header of six little-endian
32-bit words (magic, version, height, width, channel count, channel order) followed by the
pixels. Most cases read and write string streams. The `Medium_ImagingRawCache` cases, which hold the death test and the file boundary,
write files in a directory of their own under the system's temporary directory, which
`ScratchDirectory` creates empty and removes when the case ends.

## K_RGB

`K_RGB` is 1, the channel order word of an RGB file.

## append_u32

`append_u32` appends a value to a byte string as four little-endian bytes.

## header_bytes and layout_bytes

`header_bytes` returns the 24 header bytes for six given fields. `layout_bytes` returns them
for `K_RAW_IMAGE_MAGIC`, `K_RAW_IMAGE_VERSION` and a given height, width, channel count and
order.

## make_image

`make_image` returns an image whose byte at offset i is 7 * i + 3 modulo 256.

## bytes_of

`bytes_of` returns a copy of an image's bytes.

## written_bytes

`written_bytes` returns the bytes `write_raw_image` produces for an image.

## expect_untouched

`expect_untouched` expects an image to be 1 by 1 with one channel still, the destination a
failed read must not replace.

## ScratchDirectory

`ScratchDirectory` owns one directory under `std::filesystem::temp_directory_path()`. It
removes what the path held and creates the directory in its constructor, and removes it in its
destructor. `write` puts bytes in a file of the directory and `file` returns a path inside it.
The class cannot be copied or moved.

## RoundTripRow and HeaderRefusalRow

`RoundTripRow` holds an image shape and a channel order. `HeaderRefusalRow` holds stream bytes and the substrings a refusal names; `expect_header_refused` checks that `read_raw_image_header` refuses the bytes as an invalid file. `version_rows`, `short_header_rows`, `extent_rows` and `channel_rows` build the rows of the four header families from `layout_bytes` and `header_bytes`.

## WriterEmitsTheDocumentedBytes

A 1 by 2 RGB image is written as "SAIM", the words 1, 1, 2, 3, 1 and the six pixel bytes. The
case lists all thirty bytes.

## ImageRoundTripsThroughAStream

Suite `Small_ImagingRawCacheRoundTrip`. A 5 by 7 RGB image and a 4 by 3 one-channel image, written and read back, have the header fields of their shape and `ChannelOrder`, a `pixel_bytes()` equal to the byte count (105 for the first) and the same bytes.

## BGRFileIsWrittenAsGivenAndReadAsRGB

Written with `ChannelOrder::BGR`, the pixel bytes 1 to 6 are stored as given. Read back they
are 3, 2, 1, 6, 5, 4: the reader swaps the first and third channel so the image is RGB.

## ReadReusesADestinationOfAnotherSize

A destination that was 9 by 9 with one channel becomes 2 by 2 with three channels and holds the
file's bytes.

## HeaderRejectsAWrongMagic

A header that starts with the PNG signature bytes returns `InvalidFile` with "magic" and "not a raw image
file".

## HeaderRejectsAVersionOtherThanOne

Suite `Small_ImagingRawCacheVersion`. Versions 2 and 0 return `InvalidFile` with "format version" and the version, and "reads version 1".

## HeaderRejectsAStreamShorterThanTheHeader

Suite `Small_ImagingRawCacheShortHeader`. A stream of 23 bytes returns `InvalidFile` with "header ended" and "23 of 24", and an empty stream with "header ended" and "0 of 24".

## HeaderRejectsAnExtentOutsideTheRange

Suite `Small_ImagingRawCacheExtent`. A height or a width of 0 returns `InvalidFile` with "height 0" or "width 0" and "16384"; a height or a width of 16385 with the extent and "16384"; extents of 0xFFFFFFFF with "height 4294967295", since the extent is checked before any product is formed; a width of 0x80000001 with "width 2147483649", since the word is read as unsigned.

## HeaderAcceptsBothSidesAtTheLimit

A 16384 by 16384 RGB header is accepted, and `pixel_bytes()` is 805306368.

## HeaderRejectsAChannelCountOrOrderTheFormatDoesNotAllow

Suite `Small_ImagingRawCacheChannels`. A channel order of 3 returns `InvalidFile` with "channel order 3" and "unknown". Three channels with order 0, one channel with order 1, four channels and zero channels with order 2 fail with the channel count and, but for four channels, the channel order.

## PixelsRejectAStreamThatEndsEarly

A stream one byte short of its pixels returns `InvalidFile` with "pixels ended" and "11 of 12".

## PixelsRejectAHeaderTheReaderDidNotValidate

`read_raw_image_pixels` given a hand-built header of height 0 returns `InvalidFile` with "height 0" and leaves
the destination as it was.

## WriterStopsTheProcessOnAnEmptyView

Suite `Medium_ImagingRawCache`. A default view breaks the precondition of `write_raw_image`, which stops the process with
"write_raw_image: the image to write has no pixels".

## WriterRejectsAChannelCountThatDoesNotFitTheOrder

One channel with `RGB` or `BGR` and three channels with `Gray` fail with the channel count,
and nothing is written.

## WriterRejectsASideAboveTheLimit

A view 16385 px wide returns `InvalidDescription` with "width 16385" and nothing is written.

## WriterReportsAFailedStream

Writing to a stream whose bad bit is set returns `InvalidFile` with "stream failed".

## FileWriterReplacesALongerFile

Suite `Medium_ImagingRawCache`. Writing a 1 by 1 image over a file that held an 8 by 8 one leaves a file of 24 + 3 bytes that
reads back as the 1 by 1 image.

## FileWriterLeavesTheFileAloneWhenItRejectsTheImage

Suite `Medium_ImagingRawCache`. A write that returns `InvalidDescription` for its channel count leaves an existing four-byte file at four bytes. The
image is checked before the file is opened.

## FileWriterRejectsAPathItCannotOpen

Suite `Medium_ImagingRawCache`. A path inside a directory that does not exist returns `InvalidFile` with "cannot open".

## FileReaderRejectsAMissingFile

Suite `Medium_ImagingRawCache`. A missing file returns `FileNotFound` with "cannot open" and the file name, and the destination is untouched.

## FileReaderRejectsTruncatedPixelsBeforeResettingTheDestination

Suite `Medium_ImagingRawCache`. A file one byte short returns `InvalidFile` with "holds 35 bytes" and "declares 36", and the destination is
untouched.

## FileReaderRejectsTrailingBytesBeforeResettingTheDestination

Suite `Medium_ImagingRawCache`. A file with one byte after its pixels returns `InvalidFile` with "holds 37 bytes" and "declares 36", and the
destination is untouched.

## FileReaderAllocatesNothingForALargeImageTheFileDoesNotHold

Suite `Medium_ImagingRawCache`. A file that is only a header declaring 16384 by 16384 pixels returns `InvalidFile` with "holds 24 bytes" and
"declares 805306392". The destination is untouched, so the reader compares the file size with
the header before it allocates.
