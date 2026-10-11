/**************************************************************************/
/* decode_options.hpp                                                     */
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
 * @file decode_options.hpp
 * @brief Declares the options every image decoder of the module takes.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

/** @brief Holds the decoders that turn the bytes of an image file into an 8-bit image. */
namespace SushiAI::Vision::Codecs
{
    /** @brief Names the channels a decoded image has. */
    enum class ChannelLayout : std::uint8_t
    {
        Stored, /**< One channel for a gray file, three in RGB order for any other. */
        Gray,   /**< One channel, whatever the file stores. */
        Color,  /**< Three channels in RGB order, whatever the file stores. */
    };

    /** @brief Names what a pixel of a palette image decodes to. */
    enum class PaletteRead : std::uint8_t
    {
        Colors,  /**< The palette entry the pixel points at, in the layout asked for. */
        Indices, /**< The stored index in one channel, whatever the layout asked for. */
    };

    /** @brief Holds what a caller asks of a decoder; the defaults keep what the file stores. */
    struct DecodeOptions
    {
        /** @brief Selects the channels of the decoded image. */
        ChannelLayout layout = ChannelLayout::Stored;

        /** @brief Selects colours or indices for a palette image; other images ignore it. */
        PaletteRead palette = PaletteRead::Colors;
    };
} // namespace SushiAI::Vision::Codecs
