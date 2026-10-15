# tests/vision/imaging/test_gray.cpp


Suites: `Unit_ImagingGray`.

Tests the expansion of a one-channel image to three equal channels.

This file needs no device. `Unit_ImagingGray` covers `expand_gray` of
`modules/vision/imaging/include/SushiAI/vision/imaging/gray.hpp`.

## make_counted

`make_counted` returns an image whose byte at offset i is 17 * i + 3 modulo 256.

## ExpandCopiesEachLevelIntoThreeChannels

A 3 by 5 grey image expands to a 3 by 5 image of `K_RGB_CHANNELS` channels, and all three
channels of every pixel hold the pixel's grey level.

## ExpandRewritesAReusedDestination

A destination that was 9 by 9 with four channels and filled with 255 becomes 2 by 2 with three
channels, and every byte holds its pixel's grey level.

## ExpandRejectsASourceThatIsNotGrey

Sources of three and of four channels throw with "expand_gray" and the channel count.
`K_GRAY_CHANNELS` is 1.

## ExpandRejectsASourceWithoutPixelsAndOneInTheDestination

An empty view throws with "no pixels". The destination's own view, and a view of two of its
rows, throw with "destination holds the source's pixels". The image keeps its size, channel
count and bytes.
