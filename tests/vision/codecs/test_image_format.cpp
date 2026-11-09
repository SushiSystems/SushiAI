/**************************************************************************/
/* test_image_format.cpp                                                  */
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
 * @file test_image_format.cpp
 * @brief Tests the detection of an image file's format from its leading bytes.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/vision/codecs/image_format.hpp>

using namespace SushiAI::Vision::Codecs;

namespace
{
    /** @brief Returns the eight bytes that open every PNG file. */
    std::vector<std::uint8_t> png_signature()
    {
        return {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    }

    /** @brief Holds bytes that name no format and the row's name. */
    struct BytesRow
    {
        std::string name;                 /**< The row's name. */
        std::vector<std::uint8_t> bytes;  /**< The bytes handed to detect_image_format. */
    };

    /** @brief Runs the detection of unrecognised bytes over each row. */
    class Small_CodecsImageFormatGarbage : public ::testing::TestWithParam<BytesRow>
    {
    };

    /** @brief Runs the detection of a signature cut short over each row. */
    class Small_CodecsImageFormatCutShort : public ::testing::TestWithParam<BytesRow>
    {
    };

    /** @brief Runs the detection of a signature with one byte changed over each row. */
    class Small_CodecsImageFormatChanged : public ::testing::TestWithParam<BytesRow>
    {
    };

    /** @brief Names a test after its row. */
    std::string row_name(const ::testing::TestParamInfo<BytesRow>& info)
    {
        return info.param.name;
    }
} // namespace

TEST(Small_CodecsImageFormat, JPEGIsDetectedFromItsThreeLeadingBytes)
{
    const std::vector<std::uint8_t> bytes = {0xFF, 0xD8, 0xFF};

    EXPECT_EQ(detect_image_format(bytes), ImageFormat::JPEG);
}

TEST(Small_CodecsImageFormat, JPEGIsDetectedWhateverFollowsItsSignature)
{
    const std::vector<std::uint8_t> jfif = {0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10};
    const std::vector<std::uint8_t> exif = {0xFF, 0xD8, 0xFF, 0xE1, 0x12, 0x34};

    EXPECT_EQ(detect_image_format(jfif), ImageFormat::JPEG);
    EXPECT_EQ(detect_image_format(exif), ImageFormat::JPEG);
}

TEST(Small_CodecsImageFormat, PNGIsDetectedFromItsEightByteSignature)
{
    std::vector<std::uint8_t> bytes = png_signature();
    EXPECT_EQ(detect_image_format(bytes), ImageFormat::PNG);

    bytes.push_back(0x00);
    EXPECT_EQ(detect_image_format(bytes), ImageFormat::PNG);
}

TEST(Small_CodecsImageFormat, RawImageIsDetectedFromTheLettersSAIM)
{
    const std::vector<std::uint8_t> bytes = {'S', 'A', 'I', 'M', 1, 0, 0, 0};

    EXPECT_EQ(detect_image_format(bytes), ImageFormat::Raw);
}

TEST(Small_CodecsImageFormat, NoBytesAreUnknown)
{
    EXPECT_EQ(detect_image_format(SushiRuntime::span<const std::uint8_t>()), ImageFormat::Unknown);
}

/** @brief Tests that bytes of no known format are Unknown. */
TEST_P(Small_CodecsImageFormatGarbage, GarbageIsUnknown)
{
    EXPECT_EQ(detect_image_format(GetParam().bytes), ImageFormat::Unknown);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_CodecsImageFormatGarbage,
    ::testing::Values(
        BytesRow{"Text", {'n', 'o', 't', ' ', 'a', 'n', ' ', 'i', 'm', 'g'}},
        BytesRow{"Zeros", std::vector<std::uint8_t>(16, 0x00)},
        BytesRow{"Gif", {'G', 'I', 'F', '8', '9', 'a'}}),
    row_name);

/** @brief Tests that a signature cut short is Unknown. */
TEST_P(Small_CodecsImageFormatCutShort, SignatureCutShortIsUnknown)
{
    EXPECT_EQ(detect_image_format(GetParam().bytes), ImageFormat::Unknown);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_CodecsImageFormatCutShort,
    ::testing::Values(
        BytesRow{"Jpeg", {0xFF, 0xD8}},
        BytesRow{"Png", {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A}},
        BytesRow{"Raw", {'S', 'A', 'I'}}),
    row_name);

/** @brief Tests that a signature with one byte changed is Unknown. */
TEST_P(Small_CodecsImageFormatChanged, SignatureWithOneByteChangedIsUnknown)
{
    EXPECT_EQ(detect_image_format(GetParam().bytes), ImageFormat::Unknown);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_CodecsImageFormatChanged,
    ::testing::Values(
        BytesRow{"Jpeg", {0xFF, 0xD8, 0x00}},
        BytesRow{"Png", {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0B}},
        BytesRow{"Raw", {'S', 'A', 'I', 'N'}}),
    row_name);
