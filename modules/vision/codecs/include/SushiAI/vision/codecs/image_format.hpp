/**************************************************************************/
/* image_format.hpp                                                       */
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
 * @file image_format.hpp
 * @brief Declares the image file formats the module tells apart and their detection.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

#include <SushiRuntime/core/span.hpp>

/** @brief Holds the decoders that turn the bytes of an image file into an 8-bit image. */
namespace SushiAI::Vision::Codecs
{
    /** @brief Names the format of an image file. */
    enum class ImageFormat : std::uint8_t
    {
        Unknown, /**< The leading bytes match no signature. */
        JPEG,    /**< The bytes FF D8 FF open the file. */
        PNG,     /**< The eight-byte PNG signature opens the file. */
        Raw,     /**< Imaging::K_RAW_IMAGE_MAGIC opens the file, least significant byte first. */
    };

    /**
     * @brief Returns the format whose signature opens the bytes, reading nothing past it.
     * @return Unknown when no signature matches or the bytes end inside every one.
     */
    [[nodiscard]] ImageFormat detect_image_format(
        SushiRuntime::span<const std::uint8_t> bytes) noexcept;
} // namespace SushiAI::Vision::Codecs
