/**************************************************************************/
/* test_jpeg_decoder.cpp                                                  */
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
 * @file test_jpeg_decoder.cpp
 * @brief Tests the JPEG decoder on files the test encodes in memory with libjpeg's writer.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <jpeglib.h>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/codecs/decode_options.hpp>
#include <SushiAI/vision/codecs/jpeg_decoder.hpp>
#include <SushiAI/vision/imaging/image.hpp>

using namespace SushiAI::Vision::Codecs;
using SushiAI::StatusCode;
using SushiAI::Vision::Imaging::Image;
using SushiAI::Vision::Imaging::K_MAX_IMAGE_SIDE;

namespace
{
    /** @brief Holds the absolute difference per channel a decoded level may have, in levels. */
    constexpr int K_JPEG_TOLERANCE = 3;

    /** @brief Holds the side of a quadrant image: four blocks of 8 by 8 pixels. */
    constexpr std::uint32_t K_QUADRANT_SIDE = 16;

    /** @brief Holds the samples of one pixel; the leading entries are the ones in use. */
    using Pixel = std::array<std::uint8_t, 4>;

    /** @brief Holds one pixel per quadrant: top left, top right, bottom left, bottom right. */
    using Quadrants = std::array<Pixel, 4>;

    /** @brief Holds the layout of the samples handed to encode_jpeg. */
    struct JPEGSource
    {
        /** @brief Names the colour space of the samples. */
        J_COLOR_SPACE space;

        /** @brief Names the colour space the file stores. */
        J_COLOR_SPACE stored;

        /** @brief Counts the samples per pixel. */
        int components;

        /** @brief Counts the columns. */
        std::uint32_t width;

        /** @brief Counts the rows. */
        std::uint32_t height;

        /** @brief Points at the samples, row-major. */
        const std::uint8_t* samples;
    };

    /**
     * @brief Returns the bytes of a JPEG file libjpeg's writer makes from samples.
     *
     * The quality is 100 and no component is subsampled, so a block of one colour decodes to
     * that colour within the rounding of the colour conversion.
     */
    std::vector<std::uint8_t> encode_jpeg(const JPEGSource& source)
    {
        jpeg_compress_struct info;
        jpeg_error_mgr errors;
        info.err = jpeg_std_error(&errors);
        jpeg_create_compress(&info);

        unsigned char* buffer = nullptr;
        unsigned long byte_count = 0;
        jpeg_mem_dest(&info, &buffer, &byte_count);

        info.image_width = source.width;
        info.image_height = source.height;
        info.input_components = source.components;
        info.in_color_space = source.space;
        jpeg_set_defaults(&info);
        jpeg_set_colorspace(&info, source.stored);
        jpeg_set_quality(&info, 100, TRUE);
        for (int component = 0; component < info.num_components; ++component)
        {
            info.comp_info[component].h_samp_factor = 1;
            info.comp_info[component].v_samp_factor = 1;
        }

        jpeg_start_compress(&info, TRUE);
        const std::size_t row_bytes =
            static_cast<std::size_t>(source.width) * static_cast<std::size_t>(source.components);
        while (info.next_scanline < info.image_height)
        {
            JSAMPROW row = const_cast<JSAMPROW>(source.samples + info.next_scanline * row_bytes);
            (void)jpeg_write_scanlines(&info, &row, 1);
        }

        jpeg_finish_compress(&info);
        jpeg_destroy_compress(&info);

        std::vector<std::uint8_t> encoded(buffer, buffer + byte_count);
        std::free(buffer);
        return encoded;
    }

    /** @brief Returns the quadrant of a pixel of a quadrant image, numbered as in Quadrants. */
    std::size_t quadrant_of(std::uint32_t y, std::uint32_t x)
    {
        const std::uint32_t half = K_QUADRANT_SIDE / 2;
        return (y < half ? 0u : 2u) + (x < half ? 0u : 1u);
    }

    /** @brief Returns the samples of a quadrant image with the given samples per pixel. */
    std::vector<std::uint8_t> quadrant_samples(const Quadrants& quadrants, int components)
    {
        std::vector<std::uint8_t> samples;
        samples.reserve(K_QUADRANT_SIDE * K_QUADRANT_SIDE * static_cast<std::size_t>(components));
        for (std::uint32_t y = 0; y < K_QUADRANT_SIDE; ++y)
            for (std::uint32_t x = 0; x < K_QUADRANT_SIDE; ++x)
                for (int component = 0; component < components; ++component)
                    samples.push_back(
                        quadrants[quadrant_of(y, x)][static_cast<std::size_t>(component)]);

        return samples;
    }

    /** @brief Returns a JPEG quadrant image stored in the colour space of its samples. */
    std::vector<std::uint8_t> quadrant_jpeg(const Quadrants& quadrants, J_COLOR_SPACE space,
                                            J_COLOR_SPACE stored, int components)
    {
        const std::vector<std::uint8_t> samples = quadrant_samples(quadrants, components);
        return encode_jpeg(
            JPEGSource{space, stored, components, K_QUADRANT_SIDE, K_QUADRANT_SIDE,
                       samples.data()});
    }

    /** @brief Holds four gray levels, one per quadrant. */
    constexpr Quadrants K_GRAY_QUADRANTS = {{{10}, {100}, {180}, {250}}};

    /** @brief Holds four RGB colours, one per quadrant, each with three different channels. */
    constexpr Quadrants K_RGB_QUADRANTS = {
        {{200, 30, 40}, {20, 180, 60}, {30, 50, 220}, {240, 230, 10}}};

    /** @brief Holds four RGB colours of equal channels, one per quadrant. */
    constexpr Quadrants K_NEUTRAL_QUADRANTS = {
        {{10, 10, 10}, {100, 100, 100}, {180, 180, 180}, {250, 250, 250}}};

    /** @brief Holds four CMYK pixels as an Adobe writer stores them, 255 meaning no ink. */
    constexpr Quadrants K_CMYK_QUADRANTS = {
        {{255, 255, 255, 255}, {0, 255, 255, 255}, {255, 255, 255, 0}, {128, 255, 64, 255}}};

    /** @brief Holds the RGB colours K_CMYK_QUADRANTS decode to. */
    constexpr Quadrants K_CMYK_AS_RGB = {
        {{255, 255, 255}, {0, 255, 255}, {0, 0, 0}, {128, 255, 64}}};

    /** @brief Returns the gray quadrant image as a JPEG file. */
    std::vector<std::uint8_t> gray_jpeg()
    {
        return quadrant_jpeg(K_GRAY_QUADRANTS, JCS_GRAYSCALE, JCS_GRAYSCALE, 1);
    }

    /** @brief Returns a quadrant image of RGB samples as a YCbCr JPEG file. */
    std::vector<std::uint8_t> color_jpeg(const Quadrants& quadrants)
    {
        return quadrant_jpeg(quadrants, JCS_RGB, JCS_YCbCr, 3);
    }

    /** @brief Returns options that ask for one layout. */
    DecodeOptions layout_options(ChannelLayout layout)
    {
        DecodeOptions options;
        options.layout = layout;
        return options;
    }

    /**
     * @brief Checks that an image is a quadrant image within K_JPEG_TOLERANCE of the colours.
     * @param expected One pixel per quadrant; its first channels entries are compared.
     */
    void expect_quadrants(const Image& image, const Quadrants& expected, std::int32_t channels)
    {
        ASSERT_EQ(image.height(), static_cast<std::int32_t>(K_QUADRANT_SIDE));
        ASSERT_EQ(image.width(), static_cast<std::int32_t>(K_QUADRANT_SIDE));
        ASSERT_EQ(image.channels(), channels);

        for (std::uint32_t y = 0; y < K_QUADRANT_SIDE; ++y)
        {
            const std::uint8_t* row = image.row(static_cast<std::int32_t>(y));
            for (std::uint32_t x = 0; x < K_QUADRANT_SIDE; ++x)
                for (std::int32_t channel = 0; channel < channels; ++channel)
                {
                    const std::size_t at = static_cast<std::size_t>(channel);
                    EXPECT_NEAR(row[x * static_cast<std::uint32_t>(channels) + at],
                                expected[quadrant_of(y, x)][at], K_JPEG_TOLERANCE)
                        << "row " << y << ", column " << x << ", channel " << channel;
                }
        }
    }

    /** @brief Returns a quadrant set whose pixels repeat one channel of another set three times. */
    Quadrants repeated(const Quadrants& gray)
    {
        Quadrants colors{};
        for (std::size_t i = 0; i < gray.size(); ++i)
            colors[i] = Pixel{gray[i][0], gray[i][0], gray[i][0], 0};

        return colors;
    }
} // namespace

TEST(Small_CodecsJpeg, GrayFileDecodesToOneChannel)
{
    SA_ASSERT_OK_AND_ASSIGN(const Image image, decode_jpeg(gray_jpeg(), DecodeOptions{}));

    expect_quadrants(image, K_GRAY_QUADRANTS, 1);
}

TEST(Small_CodecsJpeg, GrayFileDecodesToThreeEqualChannelsWhenColorIsAsked)
{
    SA_ASSERT_OK_AND_ASSIGN(const Image image,
                            decode_jpeg(gray_jpeg(), layout_options(ChannelLayout::Color)));

    expect_quadrants(image, repeated(K_GRAY_QUADRANTS), 3);
    for (std::size_t pixel = 0; pixel < image.byte_count() / 3; ++pixel)
    {
        EXPECT_EQ(image.data()[3 * pixel], image.data()[3 * pixel + 1]) << pixel;
        EXPECT_EQ(image.data()[3 * pixel], image.data()[3 * pixel + 2]) << pixel;
    }
}

TEST(Small_CodecsJpeg, ColorFileDecodesToRGB)
{
    SA_ASSERT_OK_AND_ASSIGN(const Image image,
                            decode_jpeg(color_jpeg(K_RGB_QUADRANTS), DecodeOptions{}));

    expect_quadrants(image, K_RGB_QUADRANTS, 3);
}

TEST(Small_CodecsJpeg, ColorFileDecodesToRGBWhenColorIsAsked)
{
    SA_ASSERT_OK_AND_ASSIGN(const Image image, decode_jpeg(color_jpeg(K_RGB_QUADRANTS),
                                                           layout_options(ChannelLayout::Color)));

    expect_quadrants(image, K_RGB_QUADRANTS, 3);
}

TEST(Small_CodecsJpeg, ColorFileOfNeutralPixelsDecodesToTheirLevelWhenGrayIsAsked)
{
    SA_ASSERT_OK_AND_ASSIGN(const Image image, decode_jpeg(color_jpeg(K_NEUTRAL_QUADRANTS),
                                                           layout_options(ChannelLayout::Gray)));

    // The luma of a pixel with three equal channels is that level.
    expect_quadrants(image, K_NEUTRAL_QUADRANTS, 1);
}

TEST(Small_CodecsJpeg, RGBFileWithoutAColorTransformDecodesToRGB)
{
    const std::vector<std::uint8_t> file = quadrant_jpeg(K_RGB_QUADRANTS, JCS_RGB, JCS_RGB, 3);

    SA_ASSERT_OK_AND_ASSIGN(const Image image, decode_jpeg(file, DecodeOptions{}));

    expect_quadrants(image, K_RGB_QUADRANTS, 3);
}

TEST(Small_CodecsJpeg, CMYKFileDecodesToRGB)
{
    const std::vector<std::uint8_t> file = quadrant_jpeg(K_CMYK_QUADRANTS, JCS_CMYK, JCS_CMYK, 4);

    SA_ASSERT_OK_AND_ASSIGN(const Image image, decode_jpeg(file, DecodeOptions{}));

    expect_quadrants(image, K_CMYK_AS_RGB, 3);
}

TEST(Small_CodecsJpeg, YCCKFileDecodesToRGB)
{
    const std::vector<std::uint8_t> file = quadrant_jpeg(K_CMYK_QUADRANTS, JCS_CMYK, JCS_YCCK, 4);

    SA_ASSERT_OK_AND_ASSIGN(const Image image,
                            decode_jpeg(file, layout_options(ChannelLayout::Color)));

    expect_quadrants(image, K_CMYK_AS_RGB, 3);
}

TEST(Small_CodecsJpeg, CMYKFileIsRefusedAsGrayAndTheColorSpaceIsNamed)
{
    const std::vector<std::uint8_t> cmyk = quadrant_jpeg(K_CMYK_QUADRANTS, JCS_CMYK, JCS_CMYK, 4);
    const std::vector<std::uint8_t> ycck = quadrant_jpeg(K_CMYK_QUADRANTS, JCS_CMYK, JCS_YCCK, 4);

    SA_EXPECT_FAILURE(decode_jpeg(cmyk, layout_options(ChannelLayout::Gray)),
                      StatusCode::Unsupported, "decode_jpeg: ", "CMYK", "gray");
    SA_EXPECT_FAILURE(decode_jpeg(ycck, layout_options(ChannelLayout::Gray)),
                      StatusCode::Unsupported, "decode_jpeg: ", "YCCK", "gray");
}

TEST(Small_CodecsJpeg, FileCutInHalfIsInvalid)
{
    const std::vector<std::uint8_t> file = color_jpeg(K_RGB_QUADRANTS);
    const SushiRuntime::span<const std::uint8_t> half(file.data(), file.size() / 2);

    SA_EXPECT_FAILURE(decode_jpeg(half, DecodeOptions{}), StatusCode::InvalidFile,
                      "decode_jpeg: ");
}

TEST(Small_CodecsJpeg, FileWithoutItsEndMarkerIsInvalidAndCarriesTheLibraryMessage)
{
    const std::vector<std::uint8_t> file = gray_jpeg();
    ASSERT_GT(file.size(), 2u);
    const SushiRuntime::span<const std::uint8_t> cut(file.data(), file.size() - 2);

    SA_EXPECT_FAILURE(decode_jpeg(cut, DecodeOptions{}), StatusCode::InvalidFile,
                      "decode_jpeg: the data is corrupt: ", "JPEG");
}

TEST(Small_CodecsJpeg, EveryProperPrefixOfAFileFails)
{
    const std::vector<std::uint8_t> file = gray_jpeg();

    for (std::size_t count = 0; count < file.size(); ++count)
    {
        const SushiRuntime::span<const std::uint8_t> prefix(file.data(), count);
        const SushiAI::Result<Image> decoded = decode_jpeg(prefix, DecodeOptions{});

        EXPECT_FALSE(decoded.has_value()) << count << " of " << file.size() << " bytes";
    }
}

TEST(Small_CodecsJpeg, NoBytesAreInvalid)
{
    SA_EXPECT_FAILURE(decode_jpeg(SushiRuntime::span<const std::uint8_t>(), DecodeOptions{}),
                      StatusCode::InvalidFile, "decode_jpeg: ");
}

TEST(Small_CodecsJpeg, BytesThatAreNotAJPEGAreInvalid)
{
    const std::vector<std::uint8_t> garbage(64, 0x5A);

    SA_EXPECT_FAILURE(decode_jpeg(garbage, DecodeOptions{}), StatusCode::InvalidFile,
                      "decode_jpeg: ");
}

TEST(Small_CodecsJpeg, DecoderIsUsableAfterAFailure)
{
    const std::vector<std::uint8_t> garbage(64, 0x5A);
    SA_EXPECT_FAILURE(decode_jpeg(garbage, DecodeOptions{}), StatusCode::InvalidFile,
                      "decode_jpeg: ");

    SA_ASSERT_OK_AND_ASSIGN(const Image image, decode_jpeg(gray_jpeg(), DecodeOptions{}));

    expect_quadrants(image, K_GRAY_QUADRANTS, 1);
}

TEST(Small_CodecsJpeg, SideAtTheLimitDecodes)
{
    const std::vector<std::uint8_t> samples(static_cast<std::size_t>(K_MAX_IMAGE_SIDE), 77);
    const std::vector<std::uint8_t> file =
        encode_jpeg(JPEGSource{JCS_GRAYSCALE, JCS_GRAYSCALE, 1,
                               static_cast<std::uint32_t>(K_MAX_IMAGE_SIDE), 1, samples.data()});

    SA_ASSERT_OK_AND_ASSIGN(const Image image, decode_jpeg(file, DecodeOptions{}));

    EXPECT_EQ(image.height(), 1);
    EXPECT_EQ(image.width(), K_MAX_IMAGE_SIDE);
    EXPECT_EQ(image.channels(), 1);
}

TEST(Small_CodecsJpeg, WidthAboveTheLimitIsRefused)
{
    const std::uint32_t side = static_cast<std::uint32_t>(K_MAX_IMAGE_SIDE) + 1;
    const std::vector<std::uint8_t> samples(side, 77);
    const std::vector<std::uint8_t> file =
        encode_jpeg(JPEGSource{JCS_GRAYSCALE, JCS_GRAYSCALE, 1, side, 1, samples.data()});

    SA_EXPECT_FAILURE(decode_jpeg(file, DecodeOptions{}), StatusCode::Unsupported, "1x16385",
                      "16384");
}

TEST(Small_CodecsJpeg, HeightAboveTheLimitIsRefused)
{
    const std::uint32_t side = static_cast<std::uint32_t>(K_MAX_IMAGE_SIDE) + 1;
    const std::vector<std::uint8_t> samples(side, 77);
    const std::vector<std::uint8_t> file =
        encode_jpeg(JPEGSource{JCS_GRAYSCALE, JCS_GRAYSCALE, 1, 1, side, samples.data()});

    SA_EXPECT_FAILURE(decode_jpeg(file, DecodeOptions{}), StatusCode::Unsupported, "16385x1",
                      "16384");
}
