# tests/vision/codecs

The tests of the module `vision/codecs`, one file per header of `modules/vision/codecs/include/`. The suite prefix of a case gives its ctest label.

## test_decode_image.cpp

Suites: `Unit_CodecsDecodeImage`.

Tests `decode_image` and `load_image` in
`modules/vision/codecs/include/SushiAI/vision/codecs/decode_image.hpp`. The inputs are a 2 by 2
RGB PNG that libpng encodes, an 8 by 8 gray JPEG of level 140 that libjpeg encodes at quality
100, and raw image files that `write_raw_image` writes. A level decoded from the JPEG may
differ from 140 by 3. The `Load` cases write their files into a directory under the system's
temporary directory and remove it.

### PNGBytesReachThePNGDecoder

The PNG decodes to a 2 by 2 image of three channels whose bytes are the encoded samples.

### JPEGBytesReachTheJPEGDecoder

The JPEG decodes to an 8 by 8 image of one channel, every byte within 3 of 140.

### OptionsReachTheSelectedDecoder

With `ChannelLayout::Color` the gray JPEG decodes to three channels. With
`ChannelLayout::Gray` the RGB PNG decodes to one.

### RawRGBFileDecodesToItsStoredSamples

A raw RGB file decodes to a 2 by 2 image of three channels holding the samples written, with
the default options and with `ChannelLayout::Color`.

### RawBGRFileDecodesToRGB

A raw file written in BGR order decodes with the first and third byte of every pixel swapped:
0, 0, 255, 0, 255, 0, 255, 0, 0, 30, 20, 10.

### RawGrayFileDecodesToOneChannelOrToThreeEqualOnes

A raw gray file of 3, 60, 190, 255 decodes to those four bytes in one channel, by default and
with `ChannelLayout::Gray`. With `ChannelLayout::Color` it decodes to three channels, each
level repeated three times.

### RawRGBFileIsRefusedAsGray

A raw RGB file read with `ChannelLayout::Gray` fails with `Unsupported`, "decode_image: ",
"3 channels" and "gray".

### RawFileCutShortIsInvalid

A raw file one byte short fails with `InvalidFile`, "holds 35 bytes" and "declares 36". Its
first 10 bytes fail with `InvalidFile`, "decode_image: " and "10 of 24".

### RawFileWithTrailingBytesIsInvalid

A raw file with one byte appended fails with `InvalidFile`, "holds 37 bytes" and "declares 36".

### RawFileWithASideAboveTheLimitIsInvalidBeforeAnyAllocation

A raw file whose height bytes are set to 16385 fails with `InvalidFile`, "height 16385" and
"16384".

### BytesOfNoKnownFormatAreUnsupported

`decode_image` fails with `Unsupported` and "decode_image: " for 64 bytes of 0x5A ("64 bytes"),
for a GIF signature and for an empty span ("0 bytes").

### SignatureWithNothingBehindItFailsInItsDecoder

Each signature alone fails with `InvalidFile`: the three JPEG bytes with "decode_jpeg: ", the
eight PNG bytes with "decode_png: ", and the letters SAIM with "decode_image: " and "4 of 24".

### LoadReportsAMissingPathAndNamesIt

`load_image` of a file that does not exist fails with `FileNotFound`, "absent.png" and
"load_image: ".

### LoadDecodesAPNGFile

The PNG written to `image.png` loads as three channels holding the encoded samples.

### LoadDecodesAJPEGFileWhateverItsExtension

The JPEG written to a file named `image.png` loads with `ChannelLayout::Color` as 8 by 8 by 3
bytes, each within 3 of 140.

### LoadDecodesARawImageFile

The raw gray file written to `image.saim` loads as one channel holding 3, 60, 190, 255.

### LoadNamesThePathOfAFileItsDecoderRejects

The first 40 bytes of the PNG, written to `cut.png`, fail with `InvalidFile`, "cut.png" and
"decode_png: ".

### LoadNamesThePathOfAFileOfNoKnownFormat

A file `notes.txt` holding "hello" fails with `Unsupported`, "notes.txt" and "decode_image: ".
An empty `empty.png` fails with `Unsupported`, "empty.png" and "0 bytes".

### LoadRefusesADirectory

`load_image` of the scratch directory itself fails with `InvalidFile`, the directory's name
and "regular file".

## test_image_format.cpp

Suites: `Unit_CodecsImageFormat`.

Tests `detect_image_format` in
`modules/vision/codecs/include/SushiAI/vision/codecs/image_format.hpp`.

### JPEGIsDetectedFromItsThreeLeadingBytes

The bytes FF D8 FF give `ImageFormat::JPEG`.

### JPEGIsDetectedWhateverFollowsItsSignature

FF D8 FF followed by E0 00 10, and followed by E1 12 34, both give `ImageFormat::JPEG`.

### PNGIsDetectedFromItsEightByteSignature

The eight signature bytes give `ImageFormat::PNG`, and so do the same bytes with one zero
byte appended.

### RawImageIsDetectedFromTheCacheMagic

The four bytes of `K_RAW_IMAGE_MAGIC`, least significant first, give `ImageFormat::Raw`.

### RawImageIsDetectedFromTheLettersSAIM

The bytes 'S', 'A', 'I', 'M', 1, 0, 0, 0 give `ImageFormat::Raw`.

### NoBytesAreUnknown

An empty span gives `ImageFormat::Unknown`.

### GarbageIsUnknown

The text "not an img", sixteen zero bytes and the GIF signature each give
`ImageFormat::Unknown`.

### SignatureCutShortIsUnknown

The first two bytes of the JPEG signature, the first seven of the PNG signature and the first
three of SAIM each give `ImageFormat::Unknown`.

### SignatureWithOneByteChangedIsUnknown

The PNG signature with its last byte set to 0x0B, the bytes FF D8 00 and the letters SAIN each
give `ImageFormat::Unknown`.

## test_jpeg_decoder.cpp

Suites: `Unit_CodecsJpeg`.

Tests `decode_jpeg` in `modules/vision/codecs/include/SushiAI/vision/codecs/jpeg_decoder.hpp`
on files the cases encode in memory with libjpeg at quality 100 and without subsampling. A
quadrant image is 16 by 16 pixels with one colour per 8 by 8 quadrant, and a decoded channel
may differ from its colour by 3 levels.

### GrayFileDecodesToOneChannel

The gray quadrant file of levels 10, 100, 180 and 250 decodes to one channel holding those
levels.

### GrayFileDecodesToThreeEqualChannelsWhenColorIsAsked

With `ChannelLayout::Color` the gray file decodes to three channels holding the gray levels,
and the three bytes of every pixel are equal.

### ColorFileDecodesToRGB

A YCbCr file of four RGB colours decodes with the default options to three channels holding
those colours.

### ColorFileDecodesToRGBWhenColorIsAsked

The same file decodes to the same colours with `ChannelLayout::Color`.

### ColorFileOfNeutralPixelsDecodesToTheirLevelWhenGrayIsAsked

A YCbCr file whose pixels have three equal channels of 10, 100, 180 and 250 decodes with
`ChannelLayout::Gray` to one channel holding those levels.

### RGBFileWithoutAColorTransformDecodesToRGB

A file stored in the RGB colour space decodes to three channels holding its colours.

### CMYKFileDecodesToRGB

A file stored as CMYK, where 255 means no ink, decodes to three channels: (255, 255, 255, 255)
to white, (0, 255, 255, 255) to (0, 255, 255), (255, 255, 255, 0) to black and
(128, 255, 64, 255) to (128, 255, 64).

### YCCKFileDecodesToRGB

The same CMYK pixels stored as YCCK decode with `ChannelLayout::Color` to the same four RGB
colours.

### CMYKFileIsRefusedAsGrayAndTheColorSpaceIsNamed

With `ChannelLayout::Gray` the CMYK file fails with `Unsupported`, "decode_jpeg: ", "CMYK" and
"gray", and the YCCK file fails with `Unsupported`, "decode_jpeg: ", "YCCK" and "gray".

### FileCutInHalfIsInvalid

The first half of the colour file fails with `InvalidFile` and "decode_jpeg: ".

### FileWithoutItsEndMarkerIsInvalidAndCarriesTheLibraryMessage

The gray file without its last two bytes fails with `InvalidFile`,
"decode_jpeg: the data is corrupt: " and "JPEG".

### EveryProperPrefixOfAFileFails

`decode_jpeg` returns no image for every prefix of the gray file, from zero bytes to one byte
short of the whole.

### NoBytesAreInvalid

An empty span fails with `InvalidFile` and "decode_jpeg: ".

### BytesThatAreNotAJPEGAreInvalid

64 bytes of 0x5A fail with `InvalidFile` and "decode_jpeg: ".

### DecoderIsUsableAfterAFailure

After 64 bytes of 0x5A fail with `InvalidFile`, the gray file decodes to its four levels in
one channel.

### SideAtTheLimitDecodes

A gray file `K_MAX_IMAGE_SIDE` pixels wide and one high decodes to an image of that width, a
height of 1 and one channel.

### WidthAboveTheLimitIsRefused

A file 16385 pixels wide and one high fails with `Unsupported`, "1x16385" and "16384".

### HeightAboveTheLimitIsRefused

A file one pixel wide and 16385 high fails with `Unsupported`, "16385x1" and "16384".

## test_png_decoder.cpp

Suites: `Unit_CodecsPng`.

Tests `decode_png` in `modules/vision/codecs/include/SushiAI/vision/codecs/png_decoder.hpp` on
files the cases encode in memory with libpng's simplified writer. The gray file is 2 rows by 3
columns of 0, 17, 128, 200, 254, 255. The palette file is 2 by 4 with the indices 3, 0, 1, 2,
2, 2, 0, 3 over the palette red, green, blue and (40, 40, 40). The pattern file is a 16 by 16
RGB image of generated bytes.

### GrayFileDecodesToItsStoredSamples

The gray file decodes to 2 rows, 3 columns and one channel holding its samples.

### GrayFileDecodesToThreeEqualChannelsWhenColorIsAsked

With `ChannelLayout::Color` the gray file decodes to 18 bytes in three channels, each pixel
holding its gray sample three times.

### ColorFileDecodesToItsStoredSamplesInRGBOrder

A 2 by 2 RGB file decodes to three channels whose bytes are the samples encoded.

### ColorFileOfGrayPixelsDecodesToThoseLevelsWhenGrayIsAsked

An RGB file whose pixels have three equal channels of 0, 90, 200 and 255 decodes with
`ChannelLayout::Gray` to one channel holding 0, 90, 200, 255.

### AlphaIsDroppedAndTheColorsKeepTheirStoredValues

An RGBA file with the alphas 0, 128, 255 and 1 decodes to three channels holding the stored
red, green and blue of each pixel.

### GrayAlphaFileDecodesToItsStoredGrayLevels

A gray-alpha file of the levels 5, 100, 180 and 250 with the alphas 0, 60, 255 and 128
decodes to one channel holding 5, 100, 180, 250.

### GrayAlphaFileDecodesToThreeEqualChannelsWhenColorIsAsked

A gray-alpha file of the levels 5 and 250 decodes with `ChannelLayout::Color` to 5, 5, 5, 250,
250, 250.

### PaletteFileYieldsItsStoredIndicesWhenIndicesAreAsked

With `PaletteRead::Indices` and `ChannelLayout::Stored` the palette file decodes to 2 rows, 4
columns and one channel holding its indices.

### PaletteIndicesComeOutInOneChannelWhateverTheLayout

With `PaletteRead::Indices` the palette file decodes to one channel holding its indices under
`ChannelLayout::Color` and under `ChannelLayout::Gray`.

### PaletteFileDecodesToItsPaletteColorsByDefault

With the default options the palette file decodes to three channels, each pixel holding the
palette entry its index names.

### PaletteFileReadAsGrayIsAConversionAndNotItsIndices

With `ChannelLayout::Gray` and palette colours the palette file decodes to one channel of
eight bytes. The first and the last pixel, both of index 3, hold 40.

### IndicesOptionLeavesAFileWithoutAPaletteToItsLayout

The gray file read with `PaletteRead::Indices` decodes to one channel holding its samples.

### SixteenBitGrayFileDecodesToEightBits

A 16-bit gray file of 0, 65535, 65535, 0, 65535, 0 decodes to 2 rows, 3 columns and one
channel holding 0, 255, 255, 0, 255, 0.

### SixteenBitColorFileDecodesToEightBitRGB

A 16-bit RGB file of a red, a green, a blue and a black pixel at 65535 decodes to three
channels holding 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0.

### FileCutInHalfIsInvalid

The first half of the pattern file fails with `InvalidFile` and "decode_png: ".

### FileCutInsideItsHeaderIsInvalid

The first 0, 7, 8 and 20 bytes of the pattern file each fail with `InvalidFile` and
"decode_png: ".

### EveryPrefixOfAFileFailsOrDecodesToTheFullSize

Every prefix of the pattern file shorter than the file either fails with `InvalidFile` or
decodes to an image of 16 by 16 by 3 bytes.

### BytesThatAreNotAPNGAreInvalid

64 bytes of 0x5A fail with `InvalidFile` and "decode_png: ".

### FileWithADamagedChunkIsInvalid

The pattern file with one bit of byte 19 flipped fails with `InvalidFile` and "decode_png: ".

### SideAtTheLimitDecodes

A gray file `K_MAX_IMAGE_SIDE` pixels wide and one high decodes to an image of that width and
a height of 1 holding its samples.

### WidthAboveTheLimitIsRefused

A file 16385 pixels wide and one high fails with `Unsupported`, "1x16385" and "16384".

### HeightAboveTheLimitIsRefused

A file one pixel wide and 16385 high fails with `Unsupported`, "16385x1" and "16384".
