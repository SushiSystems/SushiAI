/**************************************************************************/
/* gray.hpp                                                               */
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
 * @file gray.hpp
 * @brief Declares the expansion of a one-channel image to the three channels detection uses.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

#include <SushiAI/imaging/image.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Imaging
{
    /** @brief Holds the channel count of a grey image. */
    inline constexpr std::int32_t K_GRAY_CHANNELS = 1;

    /**
     * @brief Copies a one-channel image into three equal channels.
     *
     * Red, green and blue of a pixel each take its grey level, which is what the reference's
     * colour load gives for a grey file (data/base.py:153). A source with another channel
     * count throws Error.
     * @param destination Reset to the source's size with K_RGB_CHANNELS; must pass check_source.
     */
    void expand_gray(ImageView source, Image& destination);
} // namespace SushiAI::Imaging
