/**************************************************************************/
/* test_decode_image.cpp                                                  */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/


/**
 * @file test_decode_image.cpp
 * @brief Tests the dispatch on an image file's leading bytes and the load of a file from disk.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <jpeglib.h>
#include <png.h>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <support/tolerance.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/scratch_files/scratch_directory.hpp>
#include <SushiAI/vision/codecs/decode_image.hpp>
#include <SushiAI/vision/codecs/decode_options.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/raw_cache.hpp>
#include "jpeg_levels.hpp"

using namespace SushiAI::Vision::Codecs;
using SushiAI::StatusCode;
using SushiAI::Test::ScratchDirectory;
using SushiAI::Test::every_level_within_jpeg_tolerance;
using SushiAI::Vision::Imaging::ChannelOrder;
using SushiAI::Vision::Imaging::Image;
using SushiAI::Vision::Imaging::ImageSize;

namespace
{
    /** @brief Holds the gray level of every pixel of gray_jpeg. */
    constexpr std::uint8_t K_JPEG_LEVEL = 140;

    /** @brief Holds the samples of a 2 by 2 RGB image whose every pixel differs per channel. */
    const std::vector<std::uint8_t> K_RGB_SAMPLES = {255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30};

    /** @brief Holds the samples of a 2 by 2 gray image. */
    const std::vector<std::uint8_t> K_GRAY_SAMPLES = {3, 60, 190, 255};

    /** @brief Returns a 2 by 2 RGB PNG of K_RGB_SAMPLES, encoded by libpng's simplified writer. */
    std::vector<std::uint8_t> rgb_png()
    {
        png_image control{};
        control.version = PNG_IMAGE_VERSION;
        control.width = 2;
        control.height = 2;
        control.format = PNG_FORMAT_RGB;

        std::vector<std::uint8_t> encoded(1024);
        png_alloc_size_t byte_count = encoded.size();
        if (png_image_write_to_memory(&control, encoded.data(), &byte_count, 0,
                                      K_RGB_SAMPLES.data(), 0, nullptr) == 0)
        {
            ADD_FAILURE() << "libpng could not encode the test image: " << control.message;
            return {};
        }

        encoded.resize(byte_count);
        return encoded;
    }

    /** @brief Returns an 8 by 8 gray JPEG of K_JPEG_LEVEL at quality 100, encoded by libjpeg. */
    std::vector<std::uint8_t> gray_jpeg()
    {
        jpeg_compress_struct info;
        jpeg_error_mgr errors;
        info.err = jpeg_std_error(&errors);
        jpeg_create_compress(&info);

        unsigned char* buffer = nullptr;
        unsigned long byte_count = 0;
        jpeg_mem_dest(&info, &buffer, &byte_count);

        info.image_width = 8;
        info.image_height = 8;
        info.input_components = 1;
        info.in_color_space = JCS_GRAYSCALE;
        jpeg_set_defaults(&info);
        jpeg_set_quality(&info, 100, TRUE);

        jpeg_start_compress(&info, TRUE);
        JSAMPLE samples[8];
        for (JSAMPLE& sample : samples)
            sample = K_JPEG_LEVEL;

        while (info.next_scanline < info.image_height)
        {
            JSAMPROW row = samples;
            (void)jpeg_write_scanlines(&info, &row, 1);
        }

        jpeg_finish_compress(&info);
        jpeg_destroy_compress(&info);

        std::vector<std::uint8_t> encoded(buffer, buffer + byte_count);
        std::free(buffer);
        return encoded;
    }

    /** @brief Returns the bytes of a raw image file of the given samples. */
    std::vector<std::uint8_t> raw_file(const std::vector<std::uint8_t>& samples,
                                       std::int32_t channels, ChannelOrder order)
    {
        Image image(ImageSize{2, 2}, channels);
        for (std::size_t i = 0; i < samples.size(); ++i)
            image.data()[i] = samples[i];

        std::ostringstream out(std::ios::binary);
        SA_EXPECT_OK(SushiAI::Vision::Imaging::write_raw_image(out, image.view(), order));
        const std::string text = out.str();
        return std::vector<std::uint8_t>(text.begin(), text.end());
    }

    /** @brief Returns a copy of an image's bytes. */
    std::vector<std::uint8_t> bytes_of(const Image& image)
    {
        return std::vector<std::uint8_t>(image.data(), image.data() + image.byte_count());
    }

    /** @brief Returns options that ask for one layout. */
    DecodeOptions layout_options(ChannelLayout layout)
    {
        DecodeOptions options;
        options.layout = layout;
        return options;
    }

    /** @brief Creates the directory and writes bytes to a file in it; returns the file's path. */
    std::filesystem::path write_into(const ScratchDirectory& directory, const char* name,
                                     const std::vector<std::uint8_t>& bytes)
    {
        std::filesystem::create_directories(directory.path());
        const std::filesystem::path file = directory.path() / name;
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
        return file;
    }
} // namespace

TEST(Small_CodecsDecodeImage, PNGBytesReachThePNGDecoder)
{
    SA_ASSERT_OK_AND_ASSIGN(const Image image, decode_image(rgb_png(), DecodeOptions{}));

    EXPECT_EQ(image.height(), 2);
    EXPECT_EQ(image.width(), 2);
    EXPECT_EQ(image.channels(), 3);
    EXPECT_EQ(bytes_of(image), K_RGB_SAMPLES);
}

TEST(Small_CodecsDecodeImage, JPEGBytesReachTheJPEGDecoder)
{
    SA_ASSERT_OK_AND_ASSIGN(const Image image, decode_image(gray_jpeg(), DecodeOptions{}));

    ASSERT_EQ(image.height(), 8);
    ASSERT_EQ(image.width(), 8);
    ASSERT_EQ(image.channels(), 1);
    EXPECT_TRUE(every_level_within_jpeg_tolerance(image, K_JPEG_LEVEL));
}

TEST(Small_CodecsDecodeImage, OptionsReachTheSelectedDecoder)
{
    SA_ASSERT_OK_AND_ASSIGN(const Image from_jpeg,
                            decode_image(gray_jpeg(), layout_options(ChannelLayout::Color)));
    SA_ASSERT_OK_AND_ASSIGN(const Image from_png,
                            decode_image(rgb_png(), layout_options(ChannelLayout::Gray)));

    EXPECT_EQ(from_jpeg.channels(), 3);
    EXPECT_EQ(from_png.channels(), 1);
}

TEST(Small_CodecsDecodeImage, RawRGBFileDecodesToItsStoredSamples)
{
    const std::vector<std::uint8_t> file = raw_file(K_RGB_SAMPLES, 3, ChannelOrder::RGB);

    SA_ASSERT_OK_AND_ASSIGN(const Image stored, decode_image(file, DecodeOptions{}));
    SA_ASSERT_OK_AND_ASSIGN(const Image color,
                            decode_image(file, layout_options(ChannelLayout::Color)));

    EXPECT_EQ(stored.height(), 2);
    EXPECT_EQ(stored.width(), 2);
    EXPECT_EQ(stored.channels(), 3);
    EXPECT_EQ(bytes_of(stored), K_RGB_SAMPLES);
    EXPECT_EQ(bytes_of(color), K_RGB_SAMPLES);
}

TEST(Small_CodecsDecodeImage, RawBGRFileDecodesToRGB)
{
    const std::vector<std::uint8_t> file = raw_file(K_RGB_SAMPLES, 3, ChannelOrder::BGR);

    SA_ASSERT_OK_AND_ASSIGN(const Image image, decode_image(file, DecodeOptions{}));

    EXPECT_EQ(bytes_of(image),
              (std::vector<std::uint8_t>{0, 0, 255, 0, 255, 0, 255, 0, 0, 30, 20, 10}));
}

TEST(Small_CodecsDecodeImage, RawGrayFileKeepsOneChannelInTheStoredAndGrayLayouts)
{
    const std::vector<std::uint8_t> file = raw_file(K_GRAY_SAMPLES, 1, ChannelOrder::Gray);

    SA_ASSERT_OK_AND_ASSIGN(const Image stored, decode_image(file, DecodeOptions{}));
    SA_ASSERT_OK_AND_ASSIGN(const Image gray,
                            decode_image(file, layout_options(ChannelLayout::Gray)));

    EXPECT_EQ(stored.channels(), 1);
    EXPECT_EQ(bytes_of(stored), K_GRAY_SAMPLES);
    EXPECT_EQ(gray.channels(), 1);
    EXPECT_EQ(bytes_of(gray), K_GRAY_SAMPLES);
}

TEST(Small_CodecsDecodeImage, RawGrayFileDecodesToThreeEqualChannelsWhenColorIsAsked)
{
    const std::vector<std::uint8_t> file = raw_file(K_GRAY_SAMPLES, 1, ChannelOrder::Gray);

    SA_ASSERT_OK_AND_ASSIGN(const Image color,
                            decode_image(file, layout_options(ChannelLayout::Color)));

    EXPECT_EQ(color.channels(), 3);
    EXPECT_EQ(bytes_of(color), (std::vector<std::uint8_t>{3, 3, 3, 60, 60, 60, 190, 190, 190,
                                                          255, 255, 255}));
}

TEST(Small_CodecsDecodeImage, RawRGBFileIsRefusedAsGray)
{
    const std::vector<std::uint8_t> file = raw_file(K_RGB_SAMPLES, 3, ChannelOrder::RGB);

    SA_EXPECT_FAILURE(decode_image(file, layout_options(ChannelLayout::Gray)),
                      StatusCode::Unsupported, "decode_image: ", "3 channels", "gray");
}

TEST(Small_CodecsDecodeImage, RawFileWithItsPixelBytesCutShortIsInvalid)
{
    const std::vector<std::uint8_t> file = raw_file(K_RGB_SAMPLES, 3, ChannelOrder::RGB);
    const SushiRuntime::span<const std::uint8_t> short_pixels(file.data(), file.size() - 1);

    SA_EXPECT_FAILURE(decode_image(short_pixels, DecodeOptions{}), StatusCode::InvalidFile,
                      "holds 35 bytes", "declares 36");
}

TEST(Small_CodecsDecodeImage, RawFileWithItsHeaderCutShortIsInvalid)
{
    const std::vector<std::uint8_t> file = raw_file(K_RGB_SAMPLES, 3, ChannelOrder::RGB);
    const SushiRuntime::span<const std::uint8_t> short_header(file.data(), 10);

    SA_EXPECT_FAILURE(decode_image(short_header, DecodeOptions{}), StatusCode::InvalidFile,
                      "decode_image: ", "10 of 24");
}

TEST(Small_CodecsDecodeImage, RawFileWithTrailingBytesIsInvalid)
{
    std::vector<std::uint8_t> file = raw_file(K_RGB_SAMPLES, 3, ChannelOrder::RGB);
    file.push_back(0);

    SA_EXPECT_FAILURE(decode_image(file, DecodeOptions{}), StatusCode::InvalidFile,
                      "holds 37 bytes", "declares 36");
}

TEST(Small_CodecsDecodeImage, RawFileWithASideAboveTheLimitIsInvalidBeforeAnyAllocation)
{
    std::vector<std::uint8_t> file = raw_file(K_RGB_SAMPLES, 3, ChannelOrder::RGB);
    // Bytes 8 to 11 hold the height, least significant byte first: 0x4001 is 16385.
    file[8] = 0x01;
    file[9] = 0x40;

    SA_EXPECT_FAILURE(decode_image(file, DecodeOptions{}), StatusCode::InvalidFile,
                      "height 16385", "16384");
}

namespace
{
    /** @brief Holds bytes no decoder owns and the words of the refusal they get. */
    struct UnknownRow
    {
        std::string name;                 /**< The row's name. */
        std::vector<std::uint8_t> bytes;  /**< The bytes handed to decode_image. */
        std::string detail;               /**< A phrase the refusal holds. */
    };

    /** @brief Runs the refusal of unknown bytes over each row. */
    class Small_CodecsDecodeImageUnknown : public ::testing::TestWithParam<UnknownRow>
    {
    };

    /** @brief Holds a signature with nothing behind it and the refusal it gets. */
    struct SignatureRow
    {
        std::string name;                 /**< The row's name. */
        std::vector<std::uint8_t> bytes;  /**< The signature alone. */
        std::string owner;                /**< The function whose message the refusal starts with. */
        std::string detail;               /**< A second phrase the refusal holds. */
    };

    /** @brief Runs the refusal of a bare signature over each row. */
    class Small_CodecsDecodeImageSignature : public ::testing::TestWithParam<SignatureRow>
    {
    };

    /** @brief Holds a file that no decoder takes, written under a name, and what its load says. */
    struct UnreadableRow
    {
        std::string name;                 /**< The row's name. */
        std::string file;                 /**< The name the file is written under. */
        std::vector<std::uint8_t> bytes;  /**< The file's content. */
        std::string detail;               /**< A phrase the refusal holds. */
    };

    /** @brief Runs the load of a file of no known format over each row. */
    class Medium_CodecsLoadImageUnknown : public ::testing::TestWithParam<UnreadableRow>
    {
    };
} // namespace

/** @brief Tests that bytes of no known format are Unsupported, naming the function. */
TEST_P(Small_CodecsDecodeImageUnknown, BytesOfNoKnownFormatAreUnsupported)
{
    const UnknownRow& row = GetParam();

    SA_EXPECT_FAILURE(decode_image(row.bytes, DecodeOptions{}), StatusCode::Unsupported,
                      "decode_image: ", row.detail);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_CodecsDecodeImageUnknown,
    ::testing::Values(UnknownRow{"Garbage", std::vector<std::uint8_t>(64, 0x5A), "64 bytes"},
                      UnknownRow{"Gif", {'G', 'I', 'F', '8', '9', 'a', 0, 0}, "decode_image: "},
                      UnknownRow{"NoBytes", {}, "0 bytes"}),
    [](const ::testing::TestParamInfo<UnknownRow>& info) { return info.param.name; });

/** @brief Tests that a signature with nothing behind it is InvalidFile, from its owner. */
TEST_P(Small_CodecsDecodeImageSignature, ASignatureWithNothingBehindItIsInvalid)
{
    const SignatureRow& row = GetParam();

    SA_EXPECT_FAILURE(decode_image(row.bytes, DecodeOptions{}), StatusCode::InvalidFile, row.owner,
                      row.detail);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_CodecsDecodeImageSignature,
    ::testing::Values(SignatureRow{"Jpeg", {0xFF, 0xD8, 0xFF}, "decode_jpeg: ", ""},
                      SignatureRow{"Png",
                                   {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A},
                                   "decode_png: ",
                                   ""},
                      SignatureRow{"Raw", {'S', 'A', 'I', 'M'}, "decode_image: ", "4 of 24"}),
    [](const ::testing::TestParamInfo<SignatureRow>& info) { return info.param.name; });

TEST(Medium_CodecsLoadImage, LoadReportsAMissingPathAndNamesIt)
{
    const ScratchDirectory directory("sushiai_unit_codecs_missing");

    SA_EXPECT_FAILURE(load_image(directory.path() / "absent.png", DecodeOptions{}),
                      StatusCode::FileNotFound, "absent.png", "load_image: ");
}

TEST(Medium_CodecsLoadImage, LoadDecodesAPNGFile)
{
    const ScratchDirectory directory("sushiai_unit_codecs_load_png");
    const std::filesystem::path path = write_into(directory, "image.png", rgb_png());

    SA_ASSERT_OK_AND_ASSIGN(const Image image, load_image(path, DecodeOptions{}));

    EXPECT_EQ(image.channels(), 3);
    EXPECT_EQ(bytes_of(image), K_RGB_SAMPLES);
}

TEST(Medium_CodecsLoadImage, LoadDecodesAJPEGFileWhateverItsExtension)
{
    const ScratchDirectory directory("sushiai_unit_codecs_load_jpeg");
    const std::filesystem::path path = write_into(directory, "image.png", gray_jpeg());

    SA_ASSERT_OK_AND_ASSIGN(const Image image,
                            load_image(path, layout_options(ChannelLayout::Color)));

    ASSERT_EQ(image.channels(), 3);
    ASSERT_EQ(image.byte_count(), 8u * 8u * 3u);
    EXPECT_TRUE(every_level_within_jpeg_tolerance(image, K_JPEG_LEVEL));
}

TEST(Medium_CodecsLoadImage, LoadDecodesARawImageFile)
{
    const ScratchDirectory directory("sushiai_unit_codecs_load_raw");
    const std::filesystem::path path =
        write_into(directory, "image.saim", raw_file(K_GRAY_SAMPLES, 1, ChannelOrder::Gray));

    SA_ASSERT_OK_AND_ASSIGN(const Image image, load_image(path, DecodeOptions{}));

    EXPECT_EQ(image.channels(), 1);
    EXPECT_EQ(bytes_of(image), K_GRAY_SAMPLES);
}

TEST(Medium_CodecsLoadImage, LoadNamesThePathOfAFileItsDecoderRejects)
{
    const ScratchDirectory directory("sushiai_unit_codecs_load_corrupt");
    std::vector<std::uint8_t> bytes = rgb_png();
    ASSERT_GT(bytes.size(), 40u);
    bytes.resize(40);
    const std::filesystem::path path = write_into(directory, "cut.png", bytes);

    SA_EXPECT_FAILURE(load_image(path, DecodeOptions{}), StatusCode::InvalidFile, "cut.png",
                      "decode_png: ");
}

/** @brief Tests that the load of a file of no known format is Unsupported, naming the path. */
TEST_P(Medium_CodecsLoadImageUnknown, LoadNamesThePathOfAFileOfNoKnownFormat)
{
    const UnreadableRow& row = GetParam();
    const ScratchDirectory directory("sushiai_unit_codecs_load_unknown");
    const std::filesystem::path path = write_into(directory, row.file.c_str(), row.bytes);

    SA_EXPECT_FAILURE(load_image(path, DecodeOptions{}), StatusCode::Unsupported, row.file,
                      "decode_image: ", row.detail);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Medium_CodecsLoadImageUnknown,
    ::testing::Values(UnreadableRow{"Text", "notes.txt", {'h', 'e', 'l', 'l', 'o'}, "5 bytes"},
                      UnreadableRow{"Empty", "empty.png", {}, "0 bytes"}),
    [](const ::testing::TestParamInfo<UnreadableRow>& info) { return info.param.name; });

TEST(Medium_CodecsLoadImage, LoadRefusesADirectory)
{
    const ScratchDirectory directory("sushiai_unit_codecs_load_directory");

    SA_EXPECT_FAILURE(load_image(directory.path(), DecodeOptions{}), StatusCode::InvalidFile,
                      "sushiai_unit_codecs_load_directory", "regular file");
}
