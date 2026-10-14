/**************************************************************************/
/* test_png_decoder.cpp                                                   */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/


/**
 * @file test_png_decoder.cpp
 * @brief Tests the PNG decoder on files the test encodes in memory with libpng's writer.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <png.h>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/codecs/decode_options.hpp>
#include <SushiAI/vision/codecs/png_decoder.hpp>
#include <SushiAI/vision/imaging/image.hpp>

using namespace SushiAI::Vision::Codecs;
using SushiAI::StatusCode;
using SushiAI::Vision::Imaging::Image;
using SushiAI::Vision::Imaging::K_MAX_IMAGE_SIDE;

namespace
{
    /** @brief Holds the layout of the samples handed to encode_png. */
    struct PNGSource
    {
        /** @brief Holds the PNG_FORMAT_* value of the samples. */
        png_uint_32 format;

        /** @brief Counts the columns. */
        std::uint32_t width;

        /** @brief Counts the rows. */
        std::uint32_t height;

        /** @brief Points at the samples, row-major, in the layout format names. */
        const void* samples;

        /** @brief Points at the colour map of a palette format, or is null. */
        const void* colormap = nullptr;

        /** @brief Counts the entries of the colour map. */
        std::uint32_t colormap_entries = 0;
    };

    /** @brief Returns the bytes of a PNG file libpng's simplified writer makes from samples. */
    std::vector<std::uint8_t> encode_png(const PNGSource& source)
    {
        png_image control{};
        control.version = PNG_IMAGE_VERSION;
        control.width = source.width;
        control.height = source.height;
        control.format = source.format;
        control.colormap_entries = source.colormap_entries;

        png_alloc_size_t byte_count = 0;
        if (png_image_write_to_memory(&control, nullptr, &byte_count, 0, source.samples, 0,
                                      source.colormap) == 0)
        {
            ADD_FAILURE() << "libpng could not size the test image: " << control.message;
            return {};
        }

        std::vector<std::uint8_t> encoded(byte_count);
        if (png_image_write_to_memory(&control, encoded.data(), &byte_count, 0, source.samples, 0,
                                      source.colormap) == 0)
        {
            ADD_FAILURE() << "libpng could not encode the test image: " << control.message;
            return {};
        }

        encoded.resize(byte_count);
        return encoded;
    }

    /** @brief Returns a copy of an image's bytes. */
    std::vector<std::uint8_t> bytes_of(const Image& image)
    {
        return std::vector<std::uint8_t>(image.data(), image.data() + image.byte_count());
    }

    /** @brief Returns options that ask for one layout and palette colours. */
    DecodeOptions layout_options(ChannelLayout layout)
    {
        DecodeOptions options;
        options.layout = layout;
        return options;
    }

    /** @brief Returns options that ask for palette indices and one layout. */
    DecodeOptions index_options(ChannelLayout layout)
    {
        DecodeOptions options = layout_options(layout);
        options.palette = PaletteRead::Indices;
        return options;
    }

    /** @brief Holds the samples of a 2 by 3 gray image, row-major. */
    const std::vector<std::uint8_t> K_GRAY_SAMPLES = {0, 17, 128, 200, 254, 255};

    /** @brief Returns a 2 by 3 gray PNG of K_GRAY_SAMPLES. */
    std::vector<std::uint8_t> gray_png()
    {
        return encode_png(PNGSource{PNG_FORMAT_GRAY, 3, 2, K_GRAY_SAMPLES.data()});
    }

    /** @brief Holds the samples of a 2 by 2 RGB image whose every pixel differs per channel. */
    const std::vector<std::uint8_t> K_RGB_SAMPLES = {255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30};

    /** @brief Holds the four entries of the test palette as RGB triples. */
    const std::vector<std::uint8_t> K_PALETTE = {255, 0, 0, 0, 255, 0, 0, 0, 255, 40, 40, 40};

    /** @brief Holds the palette indices of a 2 by 4 image, row-major. */
    const std::vector<std::uint8_t> K_INDICES = {3, 0, 1, 2, 2, 2, 0, 3};

    /** @brief Returns a 2 by 4 palette PNG of K_INDICES over K_PALETTE. */
    std::vector<std::uint8_t> palette_png()
    {
        return encode_png(
            PNGSource{PNG_FORMAT_RGB_COLORMAP, 4, 2, K_INDICES.data(), K_PALETTE.data(), 4});
    }

    /** @brief Returns a 16 by 16 RGB PNG of bytes a linear congruential generator draws. */
    std::vector<std::uint8_t> pattern_png()
    {
        std::vector<std::uint8_t> samples(16 * 16 * 3);
        std::uint32_t state = 20261011u;
        for (std::uint8_t& sample : samples)
        {
            state = state * 1664525u + 1013904223u;
            sample = static_cast<std::uint8_t>(state >> 24);
        }

        return encode_png(PNGSource{PNG_FORMAT_RGB, 16, 16, samples.data()});
    }
} // namespace

TEST(Unit_CodecsPng, GrayFileDecodesToItsStoredSamples)
{
    SA_ASSERT_OK_AND_ASSIGN(const Image image, decode_png(gray_png(), DecodeOptions{}));

    EXPECT_EQ(image.height(), 2);
    EXPECT_EQ(image.width(), 3);
    EXPECT_EQ(image.channels(), 1);
    EXPECT_EQ(bytes_of(image), K_GRAY_SAMPLES);
}

TEST(Unit_CodecsPng, GrayFileDecodesToThreeEqualChannelsWhenColorIsAsked)
{
    SA_ASSERT_OK_AND_ASSIGN(const Image image,
                            decode_png(gray_png(), layout_options(ChannelLayout::Color)));

    ASSERT_EQ(image.channels(), 3);
    ASSERT_EQ(image.byte_count(), 18u);
    for (std::size_t pixel = 0; pixel < K_GRAY_SAMPLES.size(); ++pixel)
        for (std::size_t channel = 0; channel < 3; ++channel)
            EXPECT_EQ(image.data()[3 * pixel + channel], K_GRAY_SAMPLES[pixel]) << pixel;
}

TEST(Unit_CodecsPng, ColorFileDecodesToItsStoredSamplesInRGBOrder)
{
    const std::vector<std::uint8_t> file =
        encode_png(PNGSource{PNG_FORMAT_RGB, 2, 2, K_RGB_SAMPLES.data()});

    SA_ASSERT_OK_AND_ASSIGN(const Image image, decode_png(file, DecodeOptions{}));

    EXPECT_EQ(image.height(), 2);
    EXPECT_EQ(image.width(), 2);
    EXPECT_EQ(image.channels(), 3);
    EXPECT_EQ(bytes_of(image), K_RGB_SAMPLES);
}

TEST(Unit_CodecsPng, ColorFileOfGrayPixelsDecodesToThoseLevelsWhenGrayIsAsked)
{
    const std::vector<std::uint8_t> samples = {0, 0, 0, 90, 90, 90, 200, 200, 200, 255, 255, 255};
    const std::vector<std::uint8_t> file =
        encode_png(PNGSource{PNG_FORMAT_RGB, 2, 2, samples.data()});

    SA_ASSERT_OK_AND_ASSIGN(const Image image,
                            decode_png(file, layout_options(ChannelLayout::Gray)));

    EXPECT_EQ(image.channels(), 1);
    // libpng passes a pixel with three equal channels through its gray conversion unchanged.
    EXPECT_EQ(bytes_of(image), (std::vector<std::uint8_t>{0, 90, 200, 255}));
}

TEST(Unit_CodecsPng, AlphaIsDroppedAndTheColorsKeepTheirStoredValues)
{
    const std::vector<std::uint8_t> samples = {200, 100, 50, 0,   10, 20,  30,  128,
                                               90,  80,  70, 255, 0,  255, 255, 1};
    const std::vector<std::uint8_t> file =
        encode_png(PNGSource{PNG_FORMAT_RGBA, 2, 2, samples.data()});

    SA_ASSERT_OK_AND_ASSIGN(const Image image, decode_png(file, DecodeOptions{}));

    EXPECT_EQ(image.channels(), 3);
    EXPECT_EQ(bytes_of(image),
              (std::vector<std::uint8_t>{200, 100, 50, 10, 20, 30, 90, 80, 70, 0, 255, 255}));
}

TEST(Unit_CodecsPng, GrayAlphaFileDecodesToItsStoredGrayLevels)
{
    const std::vector<std::uint8_t> samples = {5, 0, 100, 60, 180, 255, 250, 128};
    const std::vector<std::uint8_t> file =
        encode_png(PNGSource{PNG_FORMAT_GA, 2, 2, samples.data()});

    SA_ASSERT_OK_AND_ASSIGN(const Image image, decode_png(file, DecodeOptions{}));

    EXPECT_EQ(image.channels(), 1);
    EXPECT_EQ(bytes_of(image), (std::vector<std::uint8_t>{5, 100, 180, 250}));
}

TEST(Unit_CodecsPng, GrayAlphaFileDecodesToThreeEqualChannelsWhenColorIsAsked)
{
    const std::vector<std::uint8_t> samples = {5, 0, 250, 128};
    const std::vector<std::uint8_t> file =
        encode_png(PNGSource{PNG_FORMAT_GA, 2, 1, samples.data()});

    SA_ASSERT_OK_AND_ASSIGN(const Image image,
                            decode_png(file, layout_options(ChannelLayout::Color)));

    EXPECT_EQ(image.channels(), 3);
    EXPECT_EQ(bytes_of(image), (std::vector<std::uint8_t>{5, 5, 5, 250, 250, 250}));
}

TEST(Unit_CodecsPng, PaletteFileYieldsItsStoredIndicesWhenIndicesAreAsked)
{
    SA_ASSERT_OK_AND_ASSIGN(const Image image,
                            decode_png(palette_png(), index_options(ChannelLayout::Stored)));

    EXPECT_EQ(image.height(), 2);
    EXPECT_EQ(image.width(), 4);
    EXPECT_EQ(image.channels(), 1);
    EXPECT_EQ(bytes_of(image), K_INDICES);
}

TEST(Unit_CodecsPng, PaletteIndicesComeOutInOneChannelWhateverTheLayout)
{
    SA_ASSERT_OK_AND_ASSIGN(const Image as_color,
                            decode_png(palette_png(), index_options(ChannelLayout::Color)));
    SA_ASSERT_OK_AND_ASSIGN(const Image as_gray,
                            decode_png(palette_png(), index_options(ChannelLayout::Gray)));

    EXPECT_EQ(as_color.channels(), 1);
    EXPECT_EQ(bytes_of(as_color), K_INDICES);
    EXPECT_EQ(as_gray.channels(), 1);
    EXPECT_EQ(bytes_of(as_gray), K_INDICES);
}

TEST(Unit_CodecsPng, PaletteFileDecodesToItsPaletteColorsByDefault)
{
    SA_ASSERT_OK_AND_ASSIGN(const Image image, decode_png(palette_png(), DecodeOptions{}));

    ASSERT_EQ(image.channels(), 3);
    ASSERT_EQ(image.byte_count(), 3 * K_INDICES.size());
    for (std::size_t pixel = 0; pixel < K_INDICES.size(); ++pixel)
        for (std::size_t channel = 0; channel < 3; ++channel)
            EXPECT_EQ(image.data()[3 * pixel + channel], K_PALETTE[3 * K_INDICES[pixel] + channel])
                << "pixel " << pixel << ", channel " << channel;
}

TEST(Unit_CodecsPng, PaletteFileReadAsGrayIsAConversionAndNotItsIndices)
{
    SA_ASSERT_OK_AND_ASSIGN(const Image image,
                            decode_png(palette_png(), layout_options(ChannelLayout::Gray)));

    ASSERT_EQ(image.channels(), 1);
    ASSERT_EQ(image.byte_count(), K_INDICES.size());
    // Entry 3 of K_PALETTE is the gray level 40, which the conversion keeps.
    EXPECT_EQ(image.data()[0], 40);
    EXPECT_EQ(image.data()[7], 40);
}

TEST(Unit_CodecsPng, IndicesOptionLeavesAFileWithoutAPaletteToItsLayout)
{
    SA_ASSERT_OK_AND_ASSIGN(const Image image,
                            decode_png(gray_png(), index_options(ChannelLayout::Stored)));

    EXPECT_EQ(image.channels(), 1);
    EXPECT_EQ(bytes_of(image), K_GRAY_SAMPLES);
}

TEST(Unit_CodecsPng, SixteenBitGrayFileDecodesToEightBits)
{
    const std::vector<std::uint16_t> samples = {0, 65535, 65535, 0, 65535, 0};
    const std::vector<std::uint8_t> file =
        encode_png(PNGSource{PNG_FORMAT_LINEAR_Y, 3, 2, samples.data()});

    SA_ASSERT_OK_AND_ASSIGN(const Image image, decode_png(file, DecodeOptions{}));

    EXPECT_EQ(image.height(), 2);
    EXPECT_EQ(image.width(), 3);
    EXPECT_EQ(image.channels(), 1);
    // The writer marks 16-bit samples linear, so only 0 and 65535 keep their place in 8 bits.
    EXPECT_EQ(bytes_of(image), (std::vector<std::uint8_t>{0, 255, 255, 0, 255, 0}));
}

TEST(Unit_CodecsPng, SixteenBitColorFileDecodesToEightBitRGB)
{
    const std::vector<std::uint16_t> samples = {65535, 0, 0, 0, 65535, 0, 0, 0, 65535, 0, 0, 0};
    const std::vector<std::uint8_t> file =
        encode_png(PNGSource{PNG_FORMAT_LINEAR_RGB, 2, 2, samples.data()});

    SA_ASSERT_OK_AND_ASSIGN(const Image image, decode_png(file, DecodeOptions{}));

    EXPECT_EQ(image.channels(), 3);
    EXPECT_EQ(bytes_of(image),
              (std::vector<std::uint8_t>{255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0}));
}

TEST(Unit_CodecsPng, FileCutInHalfIsInvalid)
{
    const std::vector<std::uint8_t> file = pattern_png();
    ASSERT_GT(file.size(), 200u);
    const SushiRuntime::span<const std::uint8_t> half(file.data(), file.size() / 2);

    SA_EXPECT_FAILURE(decode_png(half, DecodeOptions{}), StatusCode::InvalidFile, "decode_png: ");
}

TEST(Unit_CodecsPng, FileCutInsideItsHeaderIsInvalid)
{
    const std::vector<std::uint8_t> file = pattern_png();
    ASSERT_GT(file.size(), 20u);

    for (std::size_t count : {std::size_t{0}, std::size_t{7}, std::size_t{8}, std::size_t{20}})
    {
        const SushiRuntime::span<const std::uint8_t> prefix(file.data(), count);
        SA_EXPECT_FAILURE(decode_png(prefix, DecodeOptions{}), StatusCode::InvalidFile,
                          "decode_png: ")
            << count << " bytes";
    }
}

TEST(Unit_CodecsPng, EveryPrefixOfAFileFailsOrDecodesToTheFullSize)
{
    const std::vector<std::uint8_t> file = pattern_png();

    for (std::size_t count = 0; count < file.size(); ++count)
    {
        const SushiRuntime::span<const std::uint8_t> prefix(file.data(), count);
        const SushiAI::Result<Image> decoded = decode_png(prefix, DecodeOptions{});

        // libpng's simplified read stops at the last row, so a cut after it can pass.
        if (decoded.has_value())
            EXPECT_EQ(decoded.value().byte_count(), 16u * 16u * 3u) << count << " bytes";
        else
            EXPECT_EQ(decoded.status().code(), StatusCode::InvalidFile) << count << " bytes";
    }
}

TEST(Unit_CodecsPng, BytesThatAreNotAPNGAreInvalid)
{
    const std::vector<std::uint8_t> garbage(64, 0x5A);

    SA_EXPECT_FAILURE(decode_png(garbage, DecodeOptions{}), StatusCode::InvalidFile,
                      "decode_png: ");
}

TEST(Unit_CodecsPng, FileWithADamagedChunkIsInvalid)
{
    std::vector<std::uint8_t> file = pattern_png();
    ASSERT_GT(file.size(), 30u);
    // Bytes 16 to 23 hold the width and the height of IHDR, which its checksum covers.
    file[19] ^= 0x01;

    SA_EXPECT_FAILURE(decode_png(file, DecodeOptions{}), StatusCode::InvalidFile, "decode_png: ");
}

TEST(Unit_CodecsPng, SideAtTheLimitDecodes)
{
    const std::vector<std::uint8_t> samples(static_cast<std::size_t>(K_MAX_IMAGE_SIDE), 9);
    const std::vector<std::uint8_t> file = encode_png(PNGSource{
        PNG_FORMAT_GRAY, static_cast<std::uint32_t>(K_MAX_IMAGE_SIDE), 1, samples.data()});

    SA_ASSERT_OK_AND_ASSIGN(const Image image, decode_png(file, DecodeOptions{}));

    EXPECT_EQ(image.height(), 1);
    EXPECT_EQ(image.width(), K_MAX_IMAGE_SIDE);
    EXPECT_EQ(bytes_of(image), samples);
}

TEST(Unit_CodecsPng, WidthAboveTheLimitIsRefused)
{
    const std::uint32_t side = static_cast<std::uint32_t>(K_MAX_IMAGE_SIDE) + 1;
    const std::vector<std::uint8_t> samples(side, 9);
    const std::vector<std::uint8_t> file =
        encode_png(PNGSource{PNG_FORMAT_GRAY, side, 1, samples.data()});

    SA_EXPECT_FAILURE(decode_png(file, DecodeOptions{}), StatusCode::Unsupported, "1x16385",
                      "16384");
}

TEST(Unit_CodecsPng, HeightAboveTheLimitIsRefused)
{
    const std::uint32_t side = static_cast<std::uint32_t>(K_MAX_IMAGE_SIDE) + 1;
    const std::vector<std::uint8_t> samples(side, 9);
    const std::vector<std::uint8_t> file =
        encode_png(PNGSource{PNG_FORMAT_GRAY, 1, side, samples.data()});

    SA_EXPECT_FAILURE(decode_png(file, DecodeOptions{}), StatusCode::Unsupported, "16385x1",
                      "16384");
}
