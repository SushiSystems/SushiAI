# tests/vision/imaging/test_gray.cpp


Suites: `Small_ImagingGray`, `Medium_ImagingGray`.

Tests the expansion of a one-channel image to three equal channels.

This file needs no device. `Small_ImagingGray` covers `expand_gray` of
`modules/vision/imaging/include/SushiAI/vision/imaging/gray.hpp`.

## make_counted

`make_counted` returns an image whose byte at offset i is 17 * i + 3 modulo 256. `expect_bytes`
expects an image to hold exactly the listed bytes.

## ExpandCopiesEachLevelIntoThreeChannels

A 2 by 3 grey image of the levels 3, 20, 37, 54, 71 and 88 expands to a 2 by 3 image of
`K_RGB_CHANNELS` channels, and all three channels of every pixel hold the pixel's grey level.

## ExpandRewritesAReusedDestination

A destination that was 9 by 9 with four channels and filled with 255 becomes 2 by 2 with three
channels, and every byte holds its pixel's grey level (3, 20, 37, 54).

## ExpandRejectsASourceThatIsNotGrey

Sources of three and of four channels fail with "expand_gray" and the channel count.

## ExpandStopsOnASourceWithoutPixels

An empty view stops the process with "no pixels". The test is in `Medium_ImagingGray` because a
death test starts a child process.

## ExpandStopsOnASourceInTheDestination

The destination's own view, and a view of two of its rows, stop the process with "destination
holds the source's pixels".
