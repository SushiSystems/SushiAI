/**************************************************************************/
/* decode_image.hpp                                                       */
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
 * @file decode_image.hpp
 * @brief Declares the decoder that picks a format from an image file's leading bytes.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <filesystem>

#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/codecs/decode_options.hpp>

/** @brief Holds the decoders that turn the bytes of an image file into an 8-bit image. */
namespace SushiAI::Vision::Codecs
{
    /**
     * @brief Decodes an image file held in memory with the decoder its leading bytes select.
     *
     * A raw image file must hold exactly the bytes its header declares.
     * @return The image; the decoder's failure; InvalidFile for a raw image file its reader
     *         rejects; Unsupported for an unknown format and for a three-channel raw image
     *         asked for as gray.
     */
    [[nodiscard]] Result<Imaging::Image> decode_image(SushiRuntime::span<const std::uint8_t> bytes,
                                                      const DecodeOptions& options) noexcept;

    /**
     * @brief Reads a file and decodes it with decode_image.
     * @return The image; FileNotFound when the path names nothing; InvalidFile when it names
     *         something other than a readable file; OutOfMemory; otherwise the failure of
     *         decode_image. Every message opens with the path.
     */
    [[nodiscard]] Result<Imaging::Image> load_image(const std::filesystem::path& path,
                                                    const DecodeOptions& options) noexcept;
} // namespace SushiAI::Vision::Codecs
