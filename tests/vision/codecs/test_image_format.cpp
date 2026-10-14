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
 * @file test_image_format.cpp
 * @brief Tests the detection of an image file's format from its leading bytes.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/vision/codecs/image_format.hpp>
#include <SushiAI/vision/imaging/raw_cache.hpp>

using namespace SushiAI::Vision::Codecs;

namespace
{
    /** @brief Returns the eight bytes that open every PNG file. */
    std::vector<std::uint8_t> png_signature()
    {
        return {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    }

    /** @brief Returns the format detected in the first count bytes of a sequence. */
    ImageFormat detect_prefix(const std::vector<std::uint8_t>& bytes, std::size_t count)
    {
        return detect_image_format(SushiRuntime::span<const std::uint8_t>(bytes.data(), count));
    }
} // namespace

TEST(Unit_CodecsImageFormat, JPEGIsDetectedFromItsThreeLeadingBytes)
{
    const std::vector<std::uint8_t> bytes = {0xFF, 0xD8, 0xFF};

    EXPECT_EQ(detect_image_format(bytes), ImageFormat::JPEG);
}

TEST(Unit_CodecsImageFormat, JPEGIsDetectedWhateverFollowsItsSignature)
{
    const std::vector<std::uint8_t> jfif = {0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10};
    const std::vector<std::uint8_t> exif = {0xFF, 0xD8, 0xFF, 0xE1, 0x12, 0x34};

    EXPECT_EQ(detect_image_format(jfif), ImageFormat::JPEG);
    EXPECT_EQ(detect_image_format(exif), ImageFormat::JPEG);
}

TEST(Unit_CodecsImageFormat, PNGIsDetectedFromItsEightByteSignature)
{
    std::vector<std::uint8_t> bytes = png_signature();
    EXPECT_EQ(detect_image_format(bytes), ImageFormat::PNG);

    bytes.push_back(0x00);
    EXPECT_EQ(detect_image_format(bytes), ImageFormat::PNG);
}

TEST(Unit_CodecsImageFormat, RawImageIsDetectedFromTheCacheMagic)
{
    std::vector<std::uint8_t> bytes;
    for (int i = 0; i < 4; ++i)
        bytes.push_back(static_cast<std::uint8_t>(
            (SushiAI::Vision::Imaging::K_RAW_IMAGE_MAGIC >> (8 * i)) & 0xFFu));

    EXPECT_EQ(detect_image_format(bytes), ImageFormat::Raw);
}

TEST(Unit_CodecsImageFormat, RawImageIsDetectedFromTheLettersSAIM)
{
    const std::vector<std::uint8_t> bytes = {'S', 'A', 'I', 'M', 1, 0, 0, 0};

    EXPECT_EQ(detect_image_format(bytes), ImageFormat::Raw);
}

TEST(Unit_CodecsImageFormat, NoBytesAreUnknown)
{
    EXPECT_EQ(detect_image_format(SushiRuntime::span<const std::uint8_t>()), ImageFormat::Unknown);
}

TEST(Unit_CodecsImageFormat, GarbageIsUnknown)
{
    const std::vector<std::uint8_t> text = {'n', 'o', 't', ' ', 'a', 'n', ' ', 'i', 'm', 'g'};
    const std::vector<std::uint8_t> zeros(16, 0x00);
    const std::vector<std::uint8_t> gif = {'G', 'I', 'F', '8', '9', 'a'};

    EXPECT_EQ(detect_image_format(text), ImageFormat::Unknown);
    EXPECT_EQ(detect_image_format(zeros), ImageFormat::Unknown);
    EXPECT_EQ(detect_image_format(gif), ImageFormat::Unknown);
}

TEST(Unit_CodecsImageFormat, SignatureCutShortIsUnknown)
{
    const std::vector<std::uint8_t> jpeg = {0xFF, 0xD8, 0xFF};
    const std::vector<std::uint8_t> png = png_signature();
    const std::vector<std::uint8_t> raw = {'S', 'A', 'I', 'M'};

    EXPECT_EQ(detect_prefix(jpeg, 2), ImageFormat::Unknown);
    EXPECT_EQ(detect_prefix(png, 7), ImageFormat::Unknown);
    EXPECT_EQ(detect_prefix(raw, 3), ImageFormat::Unknown);
}

TEST(Unit_CodecsImageFormat, SignatureWithOneByteChangedIsUnknown)
{
    std::vector<std::uint8_t> png = png_signature();
    png[7] = 0x0B;
    const std::vector<std::uint8_t> jpeg = {0xFF, 0xD8, 0x00};
    const std::vector<std::uint8_t> raw = {'S', 'A', 'I', 'N'};

    EXPECT_EQ(detect_image_format(png), ImageFormat::Unknown);
    EXPECT_EQ(detect_image_format(jpeg), ImageFormat::Unknown);
    EXPECT_EQ(detect_image_format(raw), ImageFormat::Unknown);
}
