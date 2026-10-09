/**************************************************************************/
/* test_imaging_raw_cache.cpp                                             */
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
 * @file test_imaging_raw_cache.cpp
 * @brief Tests the raw image file's byte layout, its round trip and every malformed file.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/raw_cache.hpp>
#include "../common/test_throws.hpp"

using namespace SushiAI::Vision::Imaging;

namespace
{
    /** @brief Holds the channel_order word of an Rgb file. */
    constexpr std::uint32_t K_RGB = 1u;

    /** @brief Appends a value to a byte string as four little-endian bytes. */
    void append_u32(std::string& bytes, std::uint32_t value)
    {
        for (int i = 0; i < 4; ++i)
            bytes.push_back(static_cast<char>((value >> (8 * i)) & 0xFFu));
    }

    /** @brief Returns the 24 header bytes of a file with the given six fields. */
    std::string header_bytes(std::uint32_t magic, std::uint32_t version, std::uint32_t height,
                             std::uint32_t width, std::uint32_t channels, std::uint32_t order)
    {
        std::string bytes;
        for (std::uint32_t field : {magic, version, height, width, channels, order})
            append_u32(bytes, field);

        return bytes;
    }

    /** @brief Returns the header bytes of a version 1 file with the given layout. */
    std::string layout_bytes(std::uint32_t height, std::uint32_t width, std::uint32_t channels,
                             std::uint32_t order)
    {
        return header_bytes(K_RAW_IMAGE_MAGIC, K_RAW_IMAGE_VERSION, height, width, channels, order);
    }

    /** @brief Returns an image whose byte at offset i is (7 * i + 3) modulo 256. */
    Image make_image(ImageSize size, std::int32_t channels)
    {
        Image image(size, channels);
        for (std::size_t i = 0; i < image.byte_count(); ++i)
            image.data()[i] = static_cast<std::uint8_t>((7 * i + 3) & 0xFFu);

        return image;
    }

    /** @brief Returns a copy of an image's bytes. */
    std::vector<std::uint8_t> bytes_of(const Image& image)
    {
        return std::vector<std::uint8_t>(image.data(), image.data() + image.byte_count());
    }

    /** @brief Returns the bytes write_raw_image produces for an image. */
    std::string written_bytes(const Image& image, ChannelOrder order)
    {
        std::ostringstream out(std::ios::binary);
        write_raw_image(out, image.view(), order);
        return out.str();
    }

    /** @brief Checks that an image still has the one gray pixel a failed read must not replace. */
    void expect_untouched(const Image& destination)
    {
        EXPECT_EQ(destination.height(), 1);
        EXPECT_EQ(destination.width(), 1);
        EXPECT_EQ(destination.channels(), 1);
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
            std::filesystem::path write(const char* name, const std::string& bytes) const
            {
                const std::filesystem::path file = path_ / name;
                std::ofstream out(file, std::ios::binary | std::ios::trunc);
                out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
                return file;
            }

            /** @brief Returns the path a file of the given name has in the directory. */
            std::filesystem::path file(const char* name) const { return path_ / name; }

        private:
            /// @brief Holds the directory's path.
            std::filesystem::path path_;
    };
} // namespace

TEST(Unit_ImagingRawCache, WriterEmitsTheDocumentedBytes)
{
    Image image(ImageSize{1, 2}, 3);
    const std::uint8_t pixels[6] = {10, 20, 30, 40, 50, 60};
    for (std::size_t i = 0; i < 6; ++i)
        image.data()[i] = pixels[i];

    const std::string bytes = written_bytes(image, ChannelOrder::Rgb);

    const std::string expected = {
        'S', 'A', 'I', 'M', 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0,
        10,  20,  30,  40,  50, 60,
    };
    EXPECT_EQ(bytes, expected);
}

TEST(Unit_ImagingRawCache, HeaderSizeAndMagicMatchTheDocumentedLayout)
{
    EXPECT_EQ(K_RAW_IMAGE_HEADER_BYTES, 24u);
    EXPECT_EQ(layout_bytes(1, 1, 3, K_RGB).substr(0, 4), "SAIM");
}

TEST(Unit_ImagingRawCache, RgbImageRoundTripsThroughAStream)
{
    const Image image = make_image(ImageSize{5, 7}, 3);
    std::istringstream in(written_bytes(image, ChannelOrder::Rgb), std::ios::binary);

    const RawImageHeader header = read_raw_image_header(in);
    Image read;
    read_raw_image_pixels(in, header, read);

    EXPECT_EQ(header.size.height, 5);
    EXPECT_EQ(header.size.width, 7);
    EXPECT_EQ(header.channels, 3);
    EXPECT_EQ(header.order, ChannelOrder::Rgb);
    EXPECT_EQ(header.pixel_bytes(), 105u);
    EXPECT_EQ(read.height(), 5);
    EXPECT_EQ(read.width(), 7);
    EXPECT_EQ(read.channels(), 3);
    EXPECT_EQ(bytes_of(read), bytes_of(image));
}

TEST(Unit_ImagingRawCache, GrayImageRoundTripsThroughAStream)
{
    const Image image = make_image(ImageSize{4, 3}, 1);
    std::istringstream in(written_bytes(image, ChannelOrder::Gray), std::ios::binary);

    const RawImageHeader header = read_raw_image_header(in);
    Image read;
    read_raw_image_pixels(in, header, read);

    EXPECT_EQ(header.order, ChannelOrder::Gray);
    EXPECT_EQ(read.channels(), 1);
    EXPECT_EQ(bytes_of(read), bytes_of(image));
}

TEST(Unit_ImagingRawCache, BgrFileIsWrittenAsGivenAndReadAsRgb)
{
    Image image(ImageSize{1, 2}, 3);
    const std::uint8_t pixels[6] = {1, 2, 3, 4, 5, 6};
    for (std::size_t i = 0; i < 6; ++i)
        image.data()[i] = pixels[i];

    const std::string bytes = written_bytes(image, ChannelOrder::Bgr);
    std::istringstream in(bytes, std::ios::binary);
    const RawImageHeader header = read_raw_image_header(in);
    Image read;
    read_raw_image_pixels(in, header, read);

    EXPECT_EQ(bytes.substr(K_RAW_IMAGE_HEADER_BYTES), std::string({1, 2, 3, 4, 5, 6}));
    EXPECT_EQ(header.order, ChannelOrder::Bgr);
    EXPECT_EQ(bytes_of(read), (std::vector<std::uint8_t>{3, 2, 1, 6, 5, 4}));
}

TEST(Unit_ImagingRawCache, ReadReusesADestinationOfAnotherSize)
{
    const Image image = make_image(ImageSize{2, 2}, 3);
    std::istringstream in(written_bytes(image, ChannelOrder::Rgb), std::ios::binary);
    Image read = make_image(ImageSize{9, 9}, 1);

    const RawImageHeader header = read_raw_image_header(in);
    read_raw_image_pixels(in, header, read);

    EXPECT_EQ(read.height(), 2);
    EXPECT_EQ(read.width(), 2);
    EXPECT_EQ(read.channels(), 3);
    EXPECT_EQ(bytes_of(read), bytes_of(image));
}

TEST(Unit_ImagingRawCache, HeaderRejectsAWrongMagic)
{
    std::istringstream in(header_bytes(0x474E5089u, K_RAW_IMAGE_VERSION, 1, 1, 3, K_RGB));

    SA_EXPECT_ERROR((void)read_raw_image_header(in), "magic", "not a raw image file");
}

TEST(Unit_ImagingRawCache, HeaderRejectsANewerVersionAndNamesBoth)
{
    std::istringstream in(header_bytes(K_RAW_IMAGE_MAGIC, 2, 1, 1, 3, K_RGB));

    SA_EXPECT_ERROR((void)read_raw_image_header(in), "format version 2", "reads version 1");
}

TEST(Unit_ImagingRawCache, HeaderRejectsVersionZero)
{
    std::istringstream in(header_bytes(K_RAW_IMAGE_MAGIC, 0, 1, 1, 3, K_RGB));

    SA_EXPECT_ERROR((void)read_raw_image_header(in), "format version 0", "reads version 1");
}

TEST(Unit_ImagingRawCache, HeaderRejectsATruncatedHeader)
{
    std::istringstream in(layout_bytes(1, 1, 3, K_RGB).substr(0, 23));

    SA_EXPECT_ERROR((void)read_raw_image_header(in), "header ended", "23 of 24");
}

TEST(Unit_ImagingRawCache, HeaderRejectsAnEmptyStream)
{
    std::istringstream in{std::string()};

    SA_EXPECT_ERROR((void)read_raw_image_header(in), "header ended", "0 of 24");
}

TEST(Unit_ImagingRawCache, HeaderRejectsAZeroHeight)
{
    std::istringstream in(layout_bytes(0, 4, 3, K_RGB));

    SA_EXPECT_ERROR((void)read_raw_image_header(in), "height 0", "16384");
}

TEST(Unit_ImagingRawCache, HeaderRejectsAZeroWidth)
{
    std::istringstream in(layout_bytes(4, 0, 3, K_RGB));

    SA_EXPECT_ERROR((void)read_raw_image_header(in), "width 0", "16384");
}

TEST(Unit_ImagingRawCache, HeaderRejectsAHeightAboveTheLimit)
{
    std::istringstream in(layout_bytes(16385, 4, 3, K_RGB));

    SA_EXPECT_ERROR((void)read_raw_image_header(in), "height 16385", "16384");
}

TEST(Unit_ImagingRawCache, HeaderRejectsAWidthAboveTheLimit)
{
    std::istringstream in(layout_bytes(4, 16385, 3, K_RGB));

    SA_EXPECT_ERROR((void)read_raw_image_header(in), "width 16385", "16384");
}

TEST(Unit_ImagingRawCache, HeaderAcceptsBothSidesAtTheLimit)
{
    std::istringstream in(layout_bytes(16384, 16384, 3, K_RGB));

    const RawImageHeader header = read_raw_image_header(in);

    EXPECT_EQ(header.size.height, 16384);
    EXPECT_EQ(header.size.width, 16384);
    EXPECT_EQ(header.pixel_bytes(), std::size_t{805306368});
}

TEST(Unit_ImagingRawCache, HeaderRejectsExtentsWhoseProductWouldOverflow)
{
    std::istringstream in(layout_bytes(0xFFFFFFFFu, 0xFFFFFFFFu, 3, K_RGB));

    SA_EXPECT_ERROR((void)read_raw_image_header(in), "height 4294967295");
}

TEST(Unit_ImagingRawCache, HeaderRejectsAnExtentWithTheSignBitSet)
{
    std::istringstream in(layout_bytes(4, 0x80000001u, 3, K_RGB));

    SA_EXPECT_ERROR((void)read_raw_image_header(in), "width 2147483649");
}

TEST(Unit_ImagingRawCache, HeaderRejectsAnUnknownChannelOrder)
{
    std::istringstream in(layout_bytes(4, 4, 3, 3));

    SA_EXPECT_ERROR((void)read_raw_image_header(in), "channel order 3", "unknown");
}

TEST(Unit_ImagingRawCache, HeaderRejectsThreeChannelsInAGrayFile)
{
    std::istringstream in(layout_bytes(4, 4, 3, 0));

    SA_EXPECT_ERROR((void)read_raw_image_header(in), "channel count 3", "channel order 0");
}

TEST(Unit_ImagingRawCache, HeaderRejectsOneChannelInAnRgbFile)
{
    std::istringstream in(layout_bytes(4, 4, 1, K_RGB));

    SA_EXPECT_ERROR((void)read_raw_image_header(in), "channel count 1", "channel order 1");
}

TEST(Unit_ImagingRawCache, HeaderRejectsFourChannels)
{
    std::istringstream in(layout_bytes(4, 4, 4, K_RGB));

    SA_EXPECT_ERROR((void)read_raw_image_header(in), "channel count 4");
}

TEST(Unit_ImagingRawCache, HeaderRejectsZeroChannels)
{
    std::istringstream in(layout_bytes(4, 4, 0, 2));

    SA_EXPECT_ERROR((void)read_raw_image_header(in), "channel count 0", "channel order 2");
}

TEST(Unit_ImagingRawCache, PixelsRejectAStreamThatEndsEarly)
{
    const std::string bytes = written_bytes(make_image(ImageSize{2, 2}, 3), ChannelOrder::Rgb);
    std::istringstream in(bytes.substr(0, bytes.size() - 1), std::ios::binary);
    const RawImageHeader header = read_raw_image_header(in);
    Image read;

    SA_EXPECT_ERROR(read_raw_image_pixels(in, header, read), "pixels ended", "11 of 12");
}

TEST(Unit_ImagingRawCache, PixelsRejectAHeaderTheReaderDidNotValidate)
{
    std::istringstream in(std::string(64, '\0'), std::ios::binary);
    RawImageHeader header;
    header.size = ImageSize{0, 4};
    header.channels = 3;
    Image destination(ImageSize{1, 1}, 1);

    SA_EXPECT_ERROR(read_raw_image_pixels(in, header, destination), "height 0");
    expect_untouched(destination);
}

TEST(Unit_ImagingRawCache, WriterRejectsAnEmptyView)
{
    std::ostringstream out(std::ios::binary);

    SA_EXPECT_ERROR(write_raw_image(out, ImageView(), ChannelOrder::Rgb), "no pixels");
    EXPECT_TRUE(out.str().empty());
}

TEST(Unit_ImagingRawCache, WriterRejectsAChannelCountThatDoesNotFitTheOrder)
{
    const Image gray = make_image(ImageSize{2, 2}, 1);
    const Image color = make_image(ImageSize{2, 2}, 3);
    std::ostringstream out(std::ios::binary);

    SA_EXPECT_ERROR(write_raw_image(out, gray.view(), ChannelOrder::Rgb), "channel count 1");
    SA_EXPECT_ERROR(write_raw_image(out, gray.view(), ChannelOrder::Bgr), "channel count 1");
    SA_EXPECT_ERROR(write_raw_image(out, color.view(), ChannelOrder::Gray), "channel count 3");
    EXPECT_TRUE(out.str().empty());
}

TEST(Unit_ImagingRawCache, WriterRejectsASideAboveTheLimit)
{
    const std::uint8_t pixel[3] = {0, 0, 0};
    const ImageView view(pixel, ImageSize{1, K_MAX_IMAGE_SIDE + 1}, 3);
    std::ostringstream out(std::ios::binary);

    SA_EXPECT_ERROR(write_raw_image(out, view, ChannelOrder::Rgb), "width 16385");
    EXPECT_TRUE(out.str().empty());
}

TEST(Unit_ImagingRawCache, WriterReportsAFailedStream)
{
    const Image image = make_image(ImageSize{2, 2}, 3);
    std::ostringstream out(std::ios::binary);
    out.setstate(std::ios::badbit);

    SA_EXPECT_ERROR(write_raw_image(out, image.view(), ChannelOrder::Rgb), "stream failed");
}

TEST(Unit_ImagingRawCache, ImageRoundTripsThroughAFile)
{
    const ScratchDirectory directory("sushiai_unit_raw_cache_round_trip");
    const std::filesystem::path path = directory.file("image.saim");
    const Image image = make_image(ImageSize{6, 4}, 3);

    write_raw_image_file(path, image.view(), ChannelOrder::Rgb);
    Image read;
    read_raw_image_file(path, read);

    EXPECT_EQ(std::filesystem::file_size(path), K_RAW_IMAGE_HEADER_BYTES + 72u);
    EXPECT_EQ(read.height(), 6);
    EXPECT_EQ(read.width(), 4);
    EXPECT_EQ(read.channels(), 3);
    EXPECT_EQ(bytes_of(read), bytes_of(image));
}

TEST(Unit_ImagingRawCache, FileWriterReplacesALongerFile)
{
    const ScratchDirectory directory("sushiai_unit_raw_cache_replace");
    const std::filesystem::path path = directory.file("image.saim");
    const Image small = make_image(ImageSize{1, 1}, 3);

    write_raw_image_file(path, make_image(ImageSize{8, 8}, 3).view(), ChannelOrder::Rgb);
    write_raw_image_file(path, small.view(), ChannelOrder::Rgb);
    Image read;
    read_raw_image_file(path, read);

    EXPECT_EQ(std::filesystem::file_size(path), K_RAW_IMAGE_HEADER_BYTES + 3u);
    EXPECT_EQ(bytes_of(read), bytes_of(small));
}

TEST(Unit_ImagingRawCache, FileWriterLeavesTheFileAloneWhenItRejectsTheImage)
{
    const ScratchDirectory directory("sushiai_unit_raw_cache_reject");
    const std::filesystem::path path = directory.write("image.saim", "kept");
    const Image gray = make_image(ImageSize{2, 2}, 1);

    SA_EXPECT_ERROR(write_raw_image_file(path, gray.view(), ChannelOrder::Rgb), "channel count");
    EXPECT_EQ(std::filesystem::file_size(path), 4u);
}

TEST(Unit_ImagingRawCache, FileWriterRejectsAPathItCannotOpen)
{
    const ScratchDirectory directory("sushiai_unit_raw_cache_no_parent");
    const std::filesystem::path path = directory.file("absent") / "image.saim";
    const Image image = make_image(ImageSize{2, 2}, 3);

    SA_EXPECT_ERROR(write_raw_image_file(path, image.view(), ChannelOrder::Rgb), "cannot open");
}

TEST(Unit_ImagingRawCache, FileReaderRejectsAMissingFile)
{
    const ScratchDirectory directory("sushiai_unit_raw_cache_missing");
    Image destination(ImageSize{1, 1}, 1);

    SA_EXPECT_ERROR(read_raw_image_file(directory.file("absent.saim"), destination),
                    "cannot open", "absent.saim");
    expect_untouched(destination);
}

TEST(Unit_ImagingRawCache, FileReaderRejectsATruncatedHeader)
{
    const ScratchDirectory directory("sushiai_unit_raw_cache_short_header");
    const std::filesystem::path path =
        directory.write("image.saim", layout_bytes(2, 2, 3, K_RGB).substr(0, 10));
    Image destination(ImageSize{1, 1}, 1);

    SA_EXPECT_ERROR(read_raw_image_file(path, destination), "header ended", "10 of 24");
    expect_untouched(destination);
}

TEST(Unit_ImagingRawCache, FileReaderRejectsTruncatedPixelsBeforeResettingTheDestination)
{
    const ScratchDirectory directory("sushiai_unit_raw_cache_short_pixels");
    const std::string bytes = written_bytes(make_image(ImageSize{2, 2}, 3), ChannelOrder::Rgb);
    const std::filesystem::path path =
        directory.write("image.saim", bytes.substr(0, bytes.size() - 1));
    Image destination(ImageSize{1, 1}, 1);

    SA_EXPECT_ERROR(read_raw_image_file(path, destination), "holds 35 bytes", "declares 36");
    expect_untouched(destination);
}

TEST(Unit_ImagingRawCache, FileReaderRejectsTrailingBytesBeforeResettingTheDestination)
{
    const ScratchDirectory directory("sushiai_unit_raw_cache_trailing");
    const std::string bytes = written_bytes(make_image(ImageSize{2, 2}, 3), ChannelOrder::Rgb);
    const std::filesystem::path path = directory.write("image.saim", bytes + std::string(1, '\0'));
    Image destination(ImageSize{1, 1}, 1);

    SA_EXPECT_ERROR(read_raw_image_file(path, destination), "holds 37 bytes", "declares 36");
    expect_untouched(destination);
}

TEST(Unit_ImagingRawCache, FileReaderAllocatesNothingForALargeImageTheFileDoesNotHold)
{
    const ScratchDirectory directory("sushiai_unit_raw_cache_declared_large");
    const std::filesystem::path path =
        directory.write("image.saim", layout_bytes(16384, 16384, 3, K_RGB));
    Image destination(ImageSize{1, 1}, 1);

    SA_EXPECT_ERROR(read_raw_image_file(path, destination), "holds 24 bytes",
                    "declares 805306392");
    expect_untouched(destination);
}

TEST(Unit_ImagingRawCache, FileReaderRejectsAMalformedHeaderBeforeResettingTheDestination)
{
    const ScratchDirectory directory("sushiai_unit_raw_cache_bad_header");
    const std::filesystem::path path =
        directory.write("image.saim", layout_bytes(0xFFFFFFFFu, 0xFFFFFFFFu, 3, K_RGB));
    Image destination(ImageSize{1, 1}, 1);

    SA_EXPECT_ERROR(read_raw_image_file(path, destination), "height 4294967295");
    expect_untouched(destination);
}
