# tests/vision/imaging/test_image.cpp


Suites: `Small_ImagingImage`, `Medium_ImagingImage`.

Tests the owned image, the view over foreign pixels and the checks both share.

This file needs no device. `Small_ImagingImage` covers `modules/vision/imaging/include/SushiAI/vision/imaging/image.hpp`:
`ImageSize`, `ImageView`, `Image`, `overlaps` and `check_source`. Pixels are bytes in
height, width, channel order.

Eleven `static_assert` lines at file scope hold the parts that are constant expressions:
equality of `ImageSize`, `is_valid_side`, `is_valid_size` and `is_valid_channel_count` at both
ends of their ranges and one past each, and `row_bytes()` and `byte_count()` of a view.

## write_offsets and its checks

`write_offsets` writes each byte's own offset, modulo 251, into an image. `expect_offsets`
expects an image to hold those bytes, `expect_every_byte` expects every byte to hold one value,
`expect_bytes` expects an image to hold a given vector of bytes, and `expect_row_starts_at`
expects row y of an image, of a const reference to it and of its view to start at an offset.

## ImageSizeComparesHeightAndWidth

Two sizes are equal when both sides are, and 3 by 4 differs from 4 by 3.

## AConstructedImageReportsItsExtents

An image of 4 by 6 with three channels reports those extents, 18 row bytes and 72 bytes, and
its data pointer is not null.

## AConstructedImageIsZero

Every byte of a constructed image is 0.

## TheLargestSideAndChannelCountAreAccepted

An image of `K_MAX_IMAGE_SIDE` by 1 and one of 1 by `K_MAX_IMAGE_SIDE` with `K_MAX_CHANNELS`
channels are constructed and report their byte counts.

## ConstructionRejectsASideOutsideItsBounds

Heights of 0, -2 and 16385 and widths of 0, -7 and 16385 stop the process with a message
containing "Image:" and the size. The test is in `Medium_ImagingImage` because a death test starts
a child process.

## ConstructionRejectsAChannelCountOutsideItsBounds

Channel counts of 0, -1 and 5 stop the process with a message containing "Image:" and the count;
the first also contains "[1, 4]". The test is in `Medium_ImagingImage`.

## RowsStartOneRowOfBytesApart

`row(y)` of the image, of a const reference to it and of its view all equal the data pointer
plus 15 * y for y of 0, 1 and 2 on a 5 px wide image of three channels.

## AByteWrittenThroughARowIsReadAtItsHwcOffset

A byte written at row 2, column 1, channel 2 of a 3 by 4 image is byte 29 of the data, and
every other byte is 0, compared as a whole image.

## FillSetsEveryByte

`fill(114)` sets every byte to 114.

## FillOnAnEmptyImageDoesNothing

`fill` on a default image leaves the byte count at 0.

## ResetGivesTheImageNewExtents

`reset` to 5 by 3 with four channels gives that size and channel count, 12 row bytes and 60 bytes.

## EveryByteOfTheResetExtentIsWritable

After `reset` to 5 by 3 with four channels, every byte can be written and read back.

## ResetOnADefaultImageAllocates

`reset` on a default image gives a data pointer that is not null and 18 bytes.

## ResetKeepsTheAllocationWhenItIsLargeEnough

An 8 by 8 image of three channels keeps its data pointer through a reset to 6 bytes and through
one to 192 bytes, the count it was constructed with.

## ResetRejectsTheBoundsTheConstructorRejects

`reset` stops the process for a height of 0, a width of 16385 and five channels, with a message
containing "Image::reset". The test is in `Medium_ImagingImage`.

## ARejectedResetLeavesTheImageAsItWas

After a `reset` that stops the process for a channel count of 0, the image has its size, channel
count, data pointer and bytes. The test is in `Medium_ImagingImage`.

## AViewOfAnImageDescribesTheSamePixels

`view()` has the image's data pointer, size, channel count, row bytes and byte count.

## AViewOverForeignBytesAddressesThemInHwcOrder

A view over a twelve-byte array as 2 by 3 with two channels has 6 row bytes, and row 1,
column 2, channel 1 is byte 11.

## ACopyOwnsItsOwnPixels

A copy has another data pointer, and filling the copy leaves the original's bytes as they
were.

## AllocatedBytesKeepsTheLargestSizeTheImageHad

After a reset from 48 bytes to 4, `byte_count()` is 4 and `allocated_bytes()` is 48.

## OverlapsFindsTheWholeImageAndAViewInsideIt

`overlaps` is true for the image's own view, for a view of two of its rows and for a view of
its last byte.

## OverlapsIsFalseForBytesOfAnotherContainer

A view over an array the image does not own does not overlap it.

## OverlapsCountsTheAllocationBeyondTheCurrentSize

A view of the last row of a 4 by 4 image still overlaps it after the image is reset to one
byte. The test is against the allocation and not the current size.

## OverlapsIsFalseForSeparateOrEmptyOperands

`overlaps` is false for two separate images, for a default view, for an unallocated image, and
for views with a height of 0 or no channels.

## CheckSourceStopsOnAViewWithoutPixels

`check_source` stops the process with the caller's name as prefix and "no pixels" for a default
view. The check_source tests are in `Medium_ImagingImage`.

## CheckSourceStopsOnASourceSizeOutsideItsBounds

A view with a height of 0 and one with a width of 16385 stop the process with the caller's name,
"source size" and the size.

## CheckSourceStopsOnASourceChannelCountOutsideItsBounds

A view of five channels stops the process with "source channel count of 5" and "[1, 4]".

## CheckSourceStopsOnASourceTheDestinationHolds

The destination's own view and a view of two of its rows stop the process with "destination
holds the source's pixels".
