/**************************************************************************/
/* test_raw_cache.cpp                                                     */
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
 * @file test_raw_cache.cpp
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
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/raw_cache.hpp>

using namespace SushiAI::Vision::Imaging;

namespace
{
    using SushiAI::StatusCode;
    using SushiAI::Test::require_value;
    /** @brief Holds the channel_order word of an RGB file. */
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
        SA_EXPECT_OK(write_raw_image(out, image.view(), order));
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

    /** @brief Describes an image shape and the channel order it is written and read with. */
    struct RoundTripRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Holds the image size. */
        ImageSize size;

        /** @brief Holds the channel count. */
        std::int32_t channels;

        /** @brief Holds the channel order the file is written with. */
        ChannelOrder order;
    };

    /** @brief Runs each stream round-trip case. */
    class Small_ImagingRawCacheRoundTrip : public ::testing::TestWithParam<RoundTripRow>
    {
    };

    /** @brief Describes the bytes of a header the reader refuses and the notes the message names. */
    struct HeaderRefusalRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Holds the stream bytes. */
        std::string bytes;

        /** @brief Holds the first substring the message names. */
        std::string first;

        /** @brief Holds the second substring the message names; empty when unused. */
        std::string second;
    };

    /** @brief Checks that the header of a row's bytes is refused as an invalid file. */
    void expect_header_refused(const HeaderRefusalRow& row)
    {
        std::istringstream in(row.bytes);

        SA_EXPECT_FAILURE(read_raw_image_header(in), StatusCode::InvalidFile, row.first, row.second);
    }

    /** @brief Returns the name of a header refusal row. */
    std::string header_refusal_name(const ::testing::TestParamInfo<HeaderRefusalRow>& info)
    {
        return info.param.name;
    }

    /** @brief Runs each format-version refusal. */
    class Small_ImagingRawCacheVersion : public ::testing::TestWithParam<HeaderRefusalRow>
    {
    };

    /** @brief Runs each short-header refusal. */
    class Small_ImagingRawCacheShortHeader : public ::testing::TestWithParam<HeaderRefusalRow>
    {
    };

    /** @brief Runs each extent refusal. */
    class Small_ImagingRawCacheExtent : public ::testing::TestWithParam<HeaderRefusalRow>
    {
    };

    /** @brief Runs each channel-count refusal. */
    class Small_ImagingRawCacheChannels : public ::testing::TestWithParam<HeaderRefusalRow>
    {
    };

    /** @brief Returns the version rows: a newer version and version zero. */
    std::vector<HeaderRefusalRow> version_rows()
    {
        return {
            {"ANewerVersion", header_bytes(K_RAW_IMAGE_MAGIC, 2, 1, 1, 3, K_RGB),
             "format version 2", "reads version 1"},
            {"VersionZero", header_bytes(K_RAW_IMAGE_MAGIC, 0, 1, 1, 3, K_RGB),
             "format version 0", "reads version 1"}};
    }

    /** @brief Returns the short-header rows: 23 bytes of 24 and an empty stream. */
    std::vector<HeaderRefusalRow> short_header_rows()
    {
        return {
            {"ATruncatedHeader", layout_bytes(1, 1, 3, K_RGB).substr(0, 23), "header ended",
             "23 of 24"},
            {"AnEmptyStream", std::string(), "header ended", "0 of 24"}};
    }

    /** @brief Returns the extent rows: zero, above the limit, an overflowing product, a sign bit. */
    std::vector<HeaderRefusalRow> extent_rows()
    {
        return {
            {"AZeroHeight", layout_bytes(0, 4, 3, K_RGB), "height 0", "16384"},
            {"AZeroWidth", layout_bytes(4, 0, 3, K_RGB), "width 0", "16384"},
            {"AHeightAboveTheLimit", layout_bytes(16385, 4, 3, K_RGB), "height 16385", "16384"},
            {"AWidthAboveTheLimit", layout_bytes(4, 16385, 3, K_RGB), "width 16385", "16384"},
            {"ExtentsWhoseProductWouldOverflow",
             layout_bytes(0xFFFFFFFFu, 0xFFFFFFFFu, 3, K_RGB), "height 4294967295", ""},
            {"AnExtentWithTheSignBitSet", layout_bytes(4, 0x80000001u, 3, K_RGB),
             "width 2147483649", ""}};
    }

    /** @brief Returns the channel rows: an unknown order and counts the order does not allow. */
    std::vector<HeaderRefusalRow> channel_rows()
    {
        return {
            {"AnUnknownChannelOrder", layout_bytes(4, 4, 3, 3), "channel order 3", "unknown"},
            {"ThreeChannelsInAGrayFile", layout_bytes(4, 4, 3, 0), "channel count 3",
             "channel order 0"},
            {"OneChannelInAnRGBFile", layout_bytes(4, 4, 1, K_RGB), "channel count 1",
             "channel order 1"},
            {"FourChannels", layout_bytes(4, 4, 4, K_RGB), "channel count 4", ""},
            {"ZeroChannels", layout_bytes(4, 4, 0, 2), "channel count 0", "channel order 2"}};
    }
} // namespace

TEST(Small_ImagingRawCache, WriterEmitsTheDocumentedBytes)
{
    Image image(ImageSize{1, 2}, 3);
    const std::uint8_t pixels[6] = {10, 20, 30, 40, 50, 60};
    for (std::size_t i = 0; i < 6; ++i)
        image.data()[i] = pixels[i];

    const std::string bytes = written_bytes(image, ChannelOrder::RGB);

    const std::string expected = {
        'S', 'A', 'I', 'M', 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0,
        10,  20,  30,  40,  50, 60,
    };
    EXPECT_EQ(bytes, expected);
}

/** @brief Tests that an image written to a stream is read back with its header and its bytes. */
TEST_P(Small_ImagingRawCacheRoundTrip, ImageRoundTripsThroughAStream)
{
    const RoundTripRow& row = GetParam();
    const Image image = make_image(row.size, row.channels);
    std::istringstream in(written_bytes(image, row.order), std::ios::binary);

    const RawImageHeader header = require_value(read_raw_image_header(in));
    Image read;
    SA_EXPECT_OK(read_raw_image_pixels(in, header, read));

    EXPECT_EQ(header.size, row.size);
    EXPECT_EQ(header.channels, row.channels);
    EXPECT_EQ(header.order, row.order);
    EXPECT_EQ(header.pixel_bytes(), image.byte_count());
    EXPECT_EQ(read.size(), row.size);
    EXPECT_EQ(read.channels(), row.channels);
    EXPECT_EQ(bytes_of(read), bytes_of(image));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ImagingRawCacheRoundTrip,
    ::testing::Values(RoundTripRow{"RGBImage", ImageSize{5, 7}, 3, ChannelOrder::RGB},
                      RoundTripRow{"GrayImage", ImageSize{4, 3}, 1, ChannelOrder::Gray}),
    [](const ::testing::TestParamInfo<RoundTripRow>& info)
    {
        return std::string(info.param.name);
    });

TEST(Small_ImagingRawCache, BGRFileIsWrittenAsGivenAndReadAsRGB)
{
    Image image(ImageSize{1, 2}, 3);
    const std::uint8_t pixels[6] = {1, 2, 3, 4, 5, 6};
    for (std::size_t i = 0; i < 6; ++i)
        image.data()[i] = pixels[i];

    const std::string bytes = written_bytes(image, ChannelOrder::BGR);
    std::istringstream in(bytes, std::ios::binary);
    const RawImageHeader header = require_value(read_raw_image_header(in));
    Image read;
    SA_EXPECT_OK(read_raw_image_pixels(in, header, read));

    EXPECT_EQ(bytes.substr(K_RAW_IMAGE_HEADER_BYTES), std::string({1, 2, 3, 4, 5, 6}));
    EXPECT_EQ(header.order, ChannelOrder::BGR);
    EXPECT_EQ(bytes_of(read), (std::vector<std::uint8_t>{3, 2, 1, 6, 5, 4}));
}

TEST(Small_ImagingRawCache, ReadReusesADestinationOfAnotherSize)
{
    const Image image = make_image(ImageSize{2, 2}, 3);
    std::istringstream in(written_bytes(image, ChannelOrder::RGB), std::ios::binary);
    Image read = make_image(ImageSize{9, 9}, 1);

    const RawImageHeader header = require_value(read_raw_image_header(in));
    SA_EXPECT_OK(read_raw_image_pixels(in, header, read));

    EXPECT_EQ(read.height(), 2);
    EXPECT_EQ(read.width(), 2);
    EXPECT_EQ(read.channels(), 3);
    EXPECT_EQ(bytes_of(read), bytes_of(image));
}

TEST(Small_ImagingRawCache, HeaderRejectsAWrongMagic)
{
    std::istringstream in(header_bytes(0x474E5089u, K_RAW_IMAGE_VERSION, 1, 1, 3, K_RGB));

    SA_EXPECT_FAILURE(read_raw_image_header(in), StatusCode::InvalidFile, "magic",
                      "not a raw image file");
}

/** @brief Tests that a format version other than 1 is refused, naming both versions. */
TEST_P(Small_ImagingRawCacheVersion, HeaderRejectsAVersionOtherThanOne)
{
    expect_header_refused(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_ImagingRawCacheVersion, ::testing::ValuesIn(version_rows()),
                         header_refusal_name);

/** @brief Tests that a stream shorter than the 24 header bytes is refused, naming the count. */
TEST_P(Small_ImagingRawCacheShortHeader, HeaderRejectsAStreamShorterThanTheHeader)
{
    expect_header_refused(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_ImagingRawCacheShortHeader,
                         ::testing::ValuesIn(short_header_rows()), header_refusal_name);

/** @brief Tests that an extent of zero, above the limit or not representable is refused. */
TEST_P(Small_ImagingRawCacheExtent, HeaderRejectsAnExtentOutsideTheRange)
{
    expect_header_refused(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_ImagingRawCacheExtent, ::testing::ValuesIn(extent_rows()),
                         header_refusal_name);

/** @brief Tests that a channel order or count the format does not allow is refused. */
TEST_P(Small_ImagingRawCacheChannels, HeaderRejectsAChannelCountOrOrderTheFormatDoesNotAllow)
{
    expect_header_refused(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_ImagingRawCacheChannels,
                         ::testing::ValuesIn(channel_rows()), header_refusal_name);

TEST(Small_ImagingRawCache, HeaderAcceptsBothSidesAtTheLimit)
{
    std::istringstream in(layout_bytes(16384, 16384, 3, K_RGB));

    const RawImageHeader header = require_value(read_raw_image_header(in));

    EXPECT_EQ(header.size.height, 16384);
    EXPECT_EQ(header.size.width, 16384);
    EXPECT_EQ(header.pixel_bytes(), std::size_t{805306368});
}

TEST(Small_ImagingRawCache, PixelsRejectAStreamThatEndsEarly)
{
    const std::string bytes = written_bytes(make_image(ImageSize{2, 2}, 3), ChannelOrder::RGB);
    std::istringstream in(bytes.substr(0, bytes.size() - 1), std::ios::binary);
    const RawImageHeader header = require_value(read_raw_image_header(in));
    Image read;

    SA_EXPECT_FAILURE(read_raw_image_pixels(in, header, read), StatusCode::InvalidFile,
                      "pixels ended", "11 of 12");
}

TEST(Small_ImagingRawCache, PixelsRejectAHeaderTheReaderDidNotValidate)
{
    std::istringstream in(std::string(64, '\0'), std::ios::binary);
    RawImageHeader header;
    header.size = ImageSize{0, 4};
    header.channels = 3;
    Image destination(ImageSize{1, 1}, 1);

    SA_EXPECT_FAILURE(read_raw_image_pixels(in, header, destination), StatusCode::InvalidFile,
                      "height 0");
    expect_untouched(destination);
}

TEST(Medium_ImagingRawCache, WriterStopsTheProcessOnAnEmptyView)
{
    std::ostringstream out(std::ios::binary);

    SA_EXPECT_FATAL((void)write_raw_image(out, ImageView(), ChannelOrder::RGB),
                    "write_raw_image: the image to write has no pixels");
}

TEST(Small_ImagingRawCache, WriterRejectsAChannelCountThatDoesNotFitTheOrder)
{
    const Image gray = make_image(ImageSize{2, 2}, 1);
    const Image color = make_image(ImageSize{2, 2}, 3);
    std::ostringstream out(std::ios::binary);

    SA_EXPECT_FAILURE(write_raw_image(out, gray.view(), ChannelOrder::RGB),
                      StatusCode::InvalidDescription, "channel count 1");
    SA_EXPECT_FAILURE(write_raw_image(out, gray.view(), ChannelOrder::BGR),
                      StatusCode::InvalidDescription, "channel count 1");
    SA_EXPECT_FAILURE(write_raw_image(out, color.view(), ChannelOrder::Gray),
                      StatusCode::InvalidDescription, "channel count 3");
    EXPECT_TRUE(out.str().empty());
}

TEST(Small_ImagingRawCache, WriterRejectsASideAboveTheLimit)
{
    const std::uint8_t pixel[3] = {0, 0, 0};
    const ImageView view(pixel, ImageSize{1, K_MAX_IMAGE_SIDE + 1}, 3);
    std::ostringstream out(std::ios::binary);

    SA_EXPECT_FAILURE(write_raw_image(out, view, ChannelOrder::RGB), StatusCode::InvalidDescription,
                      "width 16385");
    EXPECT_TRUE(out.str().empty());
}

TEST(Small_ImagingRawCache, WriterReportsAFailedStream)
{
    const Image image = make_image(ImageSize{2, 2}, 3);
    std::ostringstream out(std::ios::binary);
    out.setstate(std::ios::badbit);

    SA_EXPECT_FAILURE(write_raw_image(out, image.view(), ChannelOrder::RGB),
                      StatusCode::InvalidFile, "stream failed");
}

TEST(Medium_ImagingRawCache, FileWriterReplacesALongerFile)
{
    const ScratchDirectory directory("sushiai_unit_raw_cache_replace");
    const std::filesystem::path path = directory.file("image.saim");
    const Image small = make_image(ImageSize{1, 1}, 3);

    SA_EXPECT_OK(write_raw_image_file(path, make_image(ImageSize{8, 8}, 3).view(),
                                      ChannelOrder::RGB));
    SA_EXPECT_OK(write_raw_image_file(path, small.view(), ChannelOrder::RGB));
    Image read;
    SA_EXPECT_OK(read_raw_image_file(path, read));

    EXPECT_EQ(std::filesystem::file_size(path), K_RAW_IMAGE_HEADER_BYTES + 3u);
    EXPECT_EQ(bytes_of(read), bytes_of(small));
}

TEST(Medium_ImagingRawCache, FileWriterLeavesTheFileAloneWhenItRejectsTheImage)
{
    const ScratchDirectory directory("sushiai_unit_raw_cache_reject");
    const std::filesystem::path path = directory.write("image.saim", "kept");
    const Image gray = make_image(ImageSize{2, 2}, 1);

    SA_EXPECT_FAILURE(write_raw_image_file(path, gray.view(), ChannelOrder::RGB),
                      StatusCode::InvalidDescription, "channel count");
    EXPECT_EQ(std::filesystem::file_size(path), 4u);
}

TEST(Medium_ImagingRawCache, FileWriterRejectsAPathItCannotOpen)
{
    const ScratchDirectory directory("sushiai_unit_raw_cache_no_parent");
    const std::filesystem::path path = directory.file("absent") / "image.saim";
    const Image image = make_image(ImageSize{2, 2}, 3);

    SA_EXPECT_FAILURE(write_raw_image_file(path, image.view(), ChannelOrder::RGB),
                      StatusCode::InvalidFile, "cannot open");
}

TEST(Medium_ImagingRawCache, FileReaderRejectsAMissingFile)
{
    const ScratchDirectory directory("sushiai_unit_raw_cache_missing");
    Image destination(ImageSize{1, 1}, 1);

    SA_EXPECT_FAILURE(read_raw_image_file(directory.file("absent.saim"), destination),
                      StatusCode::FileNotFound, "cannot open", "absent.saim");
    expect_untouched(destination);
}

TEST(Medium_ImagingRawCache, FileReaderRejectsTruncatedPixelsBeforeResettingTheDestination)
{
    const ScratchDirectory directory("sushiai_unit_raw_cache_short_pixels");
    const std::string bytes = written_bytes(make_image(ImageSize{2, 2}, 3), ChannelOrder::RGB);
    const std::filesystem::path path =
        directory.write("image.saim", bytes.substr(0, bytes.size() - 1));
    Image destination(ImageSize{1, 1}, 1);

    SA_EXPECT_FAILURE(read_raw_image_file(path, destination), StatusCode::InvalidFile,
                      "holds 35 bytes", "declares 36");
    expect_untouched(destination);
}

TEST(Medium_ImagingRawCache, FileReaderRejectsTrailingBytesBeforeResettingTheDestination)
{
    const ScratchDirectory directory("sushiai_unit_raw_cache_trailing");
    const std::string bytes = written_bytes(make_image(ImageSize{2, 2}, 3), ChannelOrder::RGB);
    const std::filesystem::path path = directory.write("image.saim", bytes + std::string(1, '\0'));
    Image destination(ImageSize{1, 1}, 1);

    SA_EXPECT_FAILURE(read_raw_image_file(path, destination), StatusCode::InvalidFile,
                      "holds 37 bytes", "declares 36");
    expect_untouched(destination);
}

TEST(Medium_ImagingRawCache, FileReaderAllocatesNothingForALargeImageTheFileDoesNotHold)
{
    const ScratchDirectory directory("sushiai_unit_raw_cache_declared_large");
    const std::filesystem::path path =
        directory.write("image.saim", layout_bytes(16384, 16384, 3, K_RGB));
    Image destination(ImageSize{1, 1}, 1);

    SA_EXPECT_FAILURE(read_raw_image_file(path, destination), StatusCode::InvalidFile,
                      "holds 24 bytes", "declares 805306392");
    expect_untouched(destination);
}

