/**************************************************************************/
/* jpeg_decoder.hpp                                                       */
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
 * @file jpeg_decoder.hpp
 * @brief Declares the decoder of a JPEG file held in memory.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/codecs/decode_options.hpp>

/** @brief Holds the decoders that turn the bytes of an image file into an 8-bit image. */
namespace SushiAI::Vision::Codecs
{
    /**
     * @brief Decodes a JPEG file into one gray channel or three RGB channels.
     *
     * Grayscale, YCbCr and RGB files decode to either layout, CMYK and YCCK files to RGB.
     * @return The image; InvalidFile with the library's message for bytes it rejects or
     *         reports as corrupt; Unsupported for another colour space or conversion and
     *         for a side above Imaging::K_MAX_IMAGE_SIDE; OutOfMemory.
     */
    [[nodiscard]] Result<Imaging::Image> decode_jpeg(SushiRuntime::span<const std::uint8_t> bytes,
                                                     const DecodeOptions& options);
} // namespace SushiAI::Vision::Codecs
