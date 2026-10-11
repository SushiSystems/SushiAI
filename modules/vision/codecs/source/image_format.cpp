/**************************************************************************/
/* image_format.cpp                                                       */
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
 * @file image_format.cpp
 * @brief Implements the detection of an image file's format from its leading bytes.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/codecs/image_format.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include <SushiRuntime/core/span.hpp>
#include <SushiAI/vision/imaging/raw_cache.hpp>

namespace SushiAI::Vision::Codecs
{
    namespace
    {
        /** @brief Holds the byte count of the longest signature. */
        constexpr std::size_t K_MAX_SIGNATURE_BYTES = 8;

        /** @brief Holds the bytes that open every file of one format. */
        struct Signature
        {
            /** @brief Names the format the bytes announce. */
            ImageFormat format;

            /** @brief Counts the leading entries of bytes that belong to the signature. */
            std::size_t length;

            /** @brief Holds the signature, padded with zeros to the longest one. */
            std::array<std::uint8_t, K_MAX_SIGNATURE_BYTES> bytes;
        };

        /** @brief Returns one byte of the raw image magic, index 0 being the first in a file. */
        [[nodiscard]] constexpr std::uint8_t raw_magic_byte(std::size_t index) noexcept
        {
            return static_cast<std::uint8_t>((Imaging::K_RAW_IMAGE_MAGIC >> (8 * index)) & 0xFFu);
        }

        /** @brief Holds one signature per format; a new format adds a row. */
        constexpr std::array<Signature, 3> K_SIGNATURES = {{
            {ImageFormat::JPEG, 3, {{0xFF, 0xD8, 0xFF}}},
            {ImageFormat::PNG, 8, {{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}}},
            {ImageFormat::Raw,
             4,
             {{raw_magic_byte(0), raw_magic_byte(1), raw_magic_byte(2), raw_magic_byte(3)}}},
        }};
    } // namespace

    ImageFormat detect_image_format(SushiRuntime::span<const std::uint8_t> bytes) noexcept
    {
        for (const Signature& signature : K_SIGNATURES)
        {
            if (bytes.size() < signature.length)
                continue;

            if (std::equal(signature.bytes.begin(), signature.bytes.begin() + signature.length,
                           bytes.begin()))
                return signature.format;
        }

        return ImageFormat::Unknown;
    }
} // namespace SushiAI::Vision::Codecs
