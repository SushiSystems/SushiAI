# tests/vision/imaging/test_raw_cache.cpp


Suites: `Small_ImagingRawCache`.

Tests the raw image file: its bytes, its header checks and its file reader and writer.

This file needs no device. `Small_ImagingRawCache` covers
`modules/vision/imaging/include/SushiAI/vision/imaging/raw_cache.hpp`. A file is a 24-byte header of six little-endian
32-bit words (magic, version, height, width, channel count, channel order) followed by the
pixels. Most cases read and write string streams. The cases from ImageRoundTripsThroughAFile on
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

## WriterEmitsTheDocumentedBytes

A 1 by 2 RGB image is written as "SAIM", the words 1, 1, 2, 3, 1 and the six pixel bytes. The
case lists all thirty bytes.

## HeaderSizeAndMagicMatchTheDocumentedLayout

`K_RAW_IMAGE_HEADER_BYTES` is 24 and the first four header bytes are "SAIM".

## RGBImageRoundTripsThroughAStream

A 5 by 7 RGB image written and read back has the header fields 5, 7, 3 and
`ChannelOrder::RGB`, `pixel_bytes()` of 105 and the same bytes.

## GrayImageRoundTripsThroughAStream

A 4 by 3 one-channel image round-trips with `ChannelOrder::Gray`.

## BGRFileIsWrittenAsGivenAndReadAsRGB

Written with `ChannelOrder::BGR`, the pixel bytes 1 to 6 are stored as given. Read back they
are 3, 2, 1, 6, 5, 4: the reader swaps the first and third channel so the image is RGB.

## ReadReusesADestinationOfAnotherSize

A destination that was 9 by 9 with one channel becomes 2 by 2 with three channels and holds the
file's bytes.

## HeaderRejectsAWrongMagic

A header that starts with the PNG signature bytes throws with "magic" and "not a raw image
file".

## HeaderRejectsANewerVersionAndNamesBoth

Version 2 throws with "format version 2" and "reads version 1".

## HeaderRejectsVersionZero

Version 0 throws with "format version 0" and "reads version 1".

## HeaderRejectsATruncatedHeader

A stream of 23 bytes throws with "header ended" and "23 of 24".

## HeaderRejectsAnEmptyStream

An empty stream throws with "header ended" and "0 of 24".

## HeaderRejectsAZeroHeight

A height of 0 throws with "height 0" and "16384".

## HeaderRejectsAZeroWidth

A width of 0 throws with "width 0" and "16384".

## HeaderRejectsAHeightAboveTheLimit

A height of 16385 throws with "height 16385".

## HeaderRejectsAWidthAboveTheLimit

A width of 16385 throws with "width 16385".

## HeaderAcceptsBothSidesAtTheLimit

A 16384 by 16384 RGB header is accepted, and `pixel_bytes()` is 805306368.

## HeaderRejectsExtentsWhoseProductWouldOverflow

Extents of 0xFFFFFFFF throw with "height 4294967295". The extent is checked before any product
is formed.

## HeaderRejectsAnExtentWithTheSignBitSet

A width of 0x80000001 throws with "width 2147483649". The word is read as unsigned and is not
taken for a negative number.

## HeaderRejectsAnUnknownChannelOrder

A channel order of 3 throws with "channel order 3" and "unknown".

## HeaderRejectsThreeChannelsInAGrayFile

Three channels with order 0 throw with "channel count 3" and "channel order 0".

## HeaderRejectsOneChannelInAnRGBFile

One channel with order 1 throws with "channel count 1" and "channel order 1".

## HeaderRejectsFourChannels

Four channels throw with "channel count 4".

## HeaderRejectsZeroChannels

Zero channels with order 2 throw with "channel count 0" and "channel order 2".

## PixelsRejectAStreamThatEndsEarly

A stream one byte short of its pixels throws with "pixels ended" and "11 of 12".

## PixelsRejectAHeaderTheReaderDidNotValidate

`read_raw_image_pixels` given a hand-built header of height 0 throws with "height 0" and leaves
the destination as it was.

## WriterRejectsAnEmptyView

A default view throws with "no pixels" and nothing is written.

## WriterRejectsAChannelCountThatDoesNotFitTheOrder

One channel with `RGB` or `BGR` and three channels with `Gray` throw with the channel count,
and nothing is written.

## WriterRejectsASideAboveTheLimit

A view 16385 px wide throws with "width 16385" and nothing is written.

## WriterReportsAFailedStream

Writing to a stream whose bad bit is set throws with "stream failed".

## ImageRoundTripsThroughAFile

A 6 by 4 RGB image written with `write_raw_image_file` gives a file of 24 + 72 bytes, and
`read_raw_image_file` returns the same extents and bytes.

## FileWriterReplacesALongerFile

Writing a 1 by 1 image over a file that held an 8 by 8 one leaves a file of 24 + 3 bytes that
reads back as the 1 by 1 image.

## FileWriterLeavesTheFileAloneWhenItRejectsTheImage

A write that throws for its channel count leaves an existing four-byte file at four bytes. The
image is checked before the file is opened.

## FileWriterRejectsAPathItCannotOpen

A path inside a directory that does not exist throws with "cannot open".

## FileReaderRejectsAMissingFile

A missing file throws with "cannot open" and the file name, and the destination is untouched.

## FileReaderRejectsATruncatedHeader

A file of 10 bytes throws with "header ended" and "10 of 24", and the destination is untouched.

## FileReaderRejectsTruncatedPixelsBeforeResettingTheDestination

A file one byte short throws with "holds 35 bytes" and "declares 36", and the destination is
untouched.

## FileReaderRejectsTrailingBytesBeforeResettingTheDestination

A file with one byte after its pixels throws with "holds 37 bytes" and "declares 36", and the
destination is untouched.

## FileReaderAllocatesNothingForALargeImageTheFileDoesNotHold

A file that is only a header declaring 16384 by 16384 pixels throws with "holds 24 bytes" and
"declares 805306392". The destination is untouched, so the reader compares the file size with
the header before it allocates.

## FileReaderRejectsAMalformedHeaderBeforeResettingTheDestination

A file whose header has extents of 0xFFFFFFFF throws with "height 4294967295", and the
destination is untouched.
