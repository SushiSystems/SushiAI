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
#include <system_error>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/codecs/decode_image.hpp>
#include <SushiAI/vision/codecs/decode_options.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/raw_cache.hpp>

using namespace SushiAI::Vision::Codecs;
using SushiAI::StatusCode;
using SushiAI::Vision::Imaging::ChannelOrder;
using SushiAI::Vision::Imaging::Image;
using SushiAI::Vision::Imaging::ImageSize;

namespace
{
    /** @brief Holds the absolute difference a level decoded from a JPEG may have, in levels. */
    constexpr int K_JPEG_TOLERANCE = 3;

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
        SushiAI::Vision::Imaging::write_raw_image(out, image.view(), order);
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

    /** @brief Owns a directory under the system's temporary directory for one test. */
    class ScratchDirectory final
    {
        public:
            /** @brief Creates the directory empty. */
            explicit ScratchDirectory(const char* name)
                : path_(std::filesystem::temp_directory_path() / name)
            {
                std::filesystem::remove_all(path_);
                std::filesystem::create_directories(path_);
            }

            /** @brief Removes the directory and what it holds. */
            ~ScratchDirectory()
            {
                std::error_code ignored;
                std::filesystem::remove_all(path_, ignored);
            }

            /** @brief Forbids a copy, which would remove the directory twice. */
            ScratchDirectory(const ScratchDirectory&) = delete;

            /** @brief Forbids copy assignment. */
            ScratchDirectory& operator=(const ScratchDirectory&) = delete;

            /** @brief Forbids a move, which would leave two owners of one path. */
            ScratchDirectory(ScratchDirectory&&) = delete;

            /** @brief Forbids move assignment. */
            ScratchDirectory& operator=(ScratchDirectory&&) = delete;

            /** @brief Writes bytes to a file in the directory and returns its path. */
            std::filesystem::path write(const char* name,
                                        const std::vector<std::uint8_t>& bytes) const
            {
                const std::filesystem::path file = path_ / name;
                std::ofstream out(file, std::ios::binary | std::ios::trunc);
                out.write(reinterpret_cast<const char*>(bytes.data()),
                          static_cast<std::streamsize>(bytes.size()));
                return file;
            }

            /** @brief Returns the path a file of the given name has in the directory. */
            std::filesystem::path file(const char* name) const { return path_ / name; }

            /** @brief Returns the directory's own path. */
            const std::filesystem::path& path() const noexcept { return path_; }

        private:
            /// @brief Holds the directory's path.
            std::filesystem::path path_;
    };
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
    for (std::size_t i = 0; i < image.byte_count(); ++i)
        EXPECT_NEAR(image.data()[i], K_JPEG_LEVEL, K_JPEG_TOLERANCE) << i;
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

TEST(Small_CodecsDecodeImage, RawGrayFileDecodesToOneChannelOrToThreeEqualOnes)
{
    const std::vector<std::uint8_t> file = raw_file(K_GRAY_SAMPLES, 1, ChannelOrder::Gray);

    SA_ASSERT_OK_AND_ASSIGN(const Image stored, decode_image(file, DecodeOptions{}));
    SA_ASSERT_OK_AND_ASSIGN(const Image gray,
                            decode_image(file, layout_options(ChannelLayout::Gray)));
    SA_ASSERT_OK_AND_ASSIGN(const Image color,
                            decode_image(file, layout_options(ChannelLayout::Color)));

    EXPECT_EQ(stored.channels(), 1);
    EXPECT_EQ(bytes_of(stored), K_GRAY_SAMPLES);
    EXPECT_EQ(bytes_of(gray), K_GRAY_SAMPLES);
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

TEST(Small_CodecsDecodeImage, RawFileCutShortIsInvalid)
{
    const std::vector<std::uint8_t> file = raw_file(K_RGB_SAMPLES, 3, ChannelOrder::RGB);
    const SushiRuntime::span<const std::uint8_t> short_pixels(file.data(), file.size() - 1);
    const SushiRuntime::span<const std::uint8_t> short_header(file.data(), 10);

    SA_EXPECT_FAILURE(decode_image(short_pixels, DecodeOptions{}), StatusCode::InvalidFile,
                      "holds 35 bytes", "declares 36");
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

TEST(Small_CodecsDecodeImage, BytesOfNoKnownFormatAreUnsupported)
{
    const std::vector<std::uint8_t> garbage(64, 0x5A);
    const std::vector<std::uint8_t> gif = {'G', 'I', 'F', '8', '9', 'a', 0, 0};

    SA_EXPECT_FAILURE(decode_image(garbage, DecodeOptions{}), StatusCode::Unsupported,
                      "decode_image: ", "64 bytes");
    SA_EXPECT_FAILURE(decode_image(gif, DecodeOptions{}), StatusCode::Unsupported,
                      "decode_image: ");
    SA_EXPECT_FAILURE(decode_image(SushiRuntime::span<const std::uint8_t>(), DecodeOptions{}),
                      StatusCode::Unsupported, "decode_image: ", "0 bytes");
}

TEST(Small_CodecsDecodeImage, SignatureWithNothingBehindItFailsInItsDecoder)
{
    const std::vector<std::uint8_t> jpeg = {0xFF, 0xD8, 0xFF};
    const std::vector<std::uint8_t> png = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    const std::vector<std::uint8_t> raw = {'S', 'A', 'I', 'M'};

    SA_EXPECT_FAILURE(decode_image(jpeg, DecodeOptions{}), StatusCode::InvalidFile,
                      "decode_jpeg: ");
    SA_EXPECT_FAILURE(decode_image(png, DecodeOptions{}), StatusCode::InvalidFile, "decode_png: ");
    SA_EXPECT_FAILURE(decode_image(raw, DecodeOptions{}), StatusCode::InvalidFile,
                      "decode_image: ", "4 of 24");
}

TEST(Small_CodecsDecodeImage, LoadReportsAMissingPathAndNamesIt)
{
    const ScratchDirectory directory("sushiai_unit_codecs_missing");

    SA_EXPECT_FAILURE(load_image(directory.file("absent.png"), DecodeOptions{}),
                      StatusCode::FileNotFound, "absent.png", "load_image: ");
}

TEST(Small_CodecsDecodeImage, LoadDecodesAPNGFile)
{
    const ScratchDirectory directory("sushiai_unit_codecs_load_png");
    const std::filesystem::path path = directory.write("image.png", rgb_png());

    SA_ASSERT_OK_AND_ASSIGN(const Image image, load_image(path, DecodeOptions{}));

    EXPECT_EQ(image.channels(), 3);
    EXPECT_EQ(bytes_of(image), K_RGB_SAMPLES);
}

TEST(Small_CodecsDecodeImage, LoadDecodesAJPEGFileWhateverItsExtension)
{
    const ScratchDirectory directory("sushiai_unit_codecs_load_jpeg");
    const std::filesystem::path path = directory.write("image.png", gray_jpeg());

    SA_ASSERT_OK_AND_ASSIGN(const Image image,
                            load_image(path, layout_options(ChannelLayout::Color)));

    ASSERT_EQ(image.channels(), 3);
    ASSERT_EQ(image.byte_count(), 8u * 8u * 3u);
    for (std::size_t i = 0; i < image.byte_count(); ++i)
        EXPECT_NEAR(image.data()[i], K_JPEG_LEVEL, K_JPEG_TOLERANCE) << i;
}

TEST(Small_CodecsDecodeImage, LoadDecodesARawImageFile)
{
    const ScratchDirectory directory("sushiai_unit_codecs_load_raw");
    const std::filesystem::path path =
        directory.write("image.saim", raw_file(K_GRAY_SAMPLES, 1, ChannelOrder::Gray));

    SA_ASSERT_OK_AND_ASSIGN(const Image image, load_image(path, DecodeOptions{}));

    EXPECT_EQ(image.channels(), 1);
    EXPECT_EQ(bytes_of(image), K_GRAY_SAMPLES);
}

TEST(Small_CodecsDecodeImage, LoadNamesThePathOfAFileItsDecoderRejects)
{
    const ScratchDirectory directory("sushiai_unit_codecs_load_corrupt");
    std::vector<std::uint8_t> bytes = rgb_png();
    ASSERT_GT(bytes.size(), 40u);
    bytes.resize(40);
    const std::filesystem::path path = directory.write("cut.png", bytes);

    SA_EXPECT_FAILURE(load_image(path, DecodeOptions{}), StatusCode::InvalidFile, "cut.png",
                      "decode_png: ");
}

TEST(Small_CodecsDecodeImage, LoadNamesThePathOfAFileOfNoKnownFormat)
{
    const ScratchDirectory directory("sushiai_unit_codecs_load_unknown");
    const std::filesystem::path text = directory.write("notes.txt", {'h', 'e', 'l', 'l', 'o'});
    const std::filesystem::path empty = directory.write("empty.png", {});

    SA_EXPECT_FAILURE(load_image(text, DecodeOptions{}), StatusCode::Unsupported, "notes.txt",
                      "decode_image: ");
    SA_EXPECT_FAILURE(load_image(empty, DecodeOptions{}), StatusCode::Unsupported, "empty.png",
                      "0 bytes");
}

TEST(Small_CodecsDecodeImage, LoadRefusesADirectory)
{
    const ScratchDirectory directory("sushiai_unit_codecs_load_directory");

    SA_EXPECT_FAILURE(load_image(directory.path(), DecodeOptions{}), StatusCode::InvalidFile,
                      "sushiai_unit_codecs_load_directory", "regular file");
}
