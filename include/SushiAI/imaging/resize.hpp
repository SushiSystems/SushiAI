/**************************************************************************/
/* resize.hpp                                                             */
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
 * @file resize.hpp
 * @brief Declares the bilinear resize of an 8-bit image and the training-load size rule.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

#include <SushiAI/imaging/image.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Imaging
{
    /** @brief Holds the number of fraction bits of a bilinear weight; the weights sum to 2048. */
    inline constexpr std::int32_t K_BILINEAR_WEIGHT_BITS = 11;

    /**
     * @brief Returns the size an image is loaded at so that its long side is image_size.
     *
     * With r = image_size / max(height, width) in double: when r == 1 the size is returned
     * as it is; otherwise each side becomes min(ceil(side * r), image_size). Small images
     * are enlarged, as in the reference (data/base.py:297-299).
     * @pre Both sides of original and image_size are positive.
     */
    [[nodiscard]] ImageSize fit_long_side(ImageSize original, std::int32_t image_size) noexcept;

    /**
     * @brief Resizes an image by bilinear interpolation between pixel centres.
     *
     * Destination column x reads source position (x + 0.5) * (source width / width) - 0.5,
     * clamped to the edge columns; rows likewise. The kernel does not widen on a reduction.
     * @param size Must pass is_valid_size; otherwise throws Error.
     * @param destination Reset to size with the source's channels; must pass check_source.
     */
    void resize_bilinear(ImageView source, ImageSize size, Image& destination);

    /**
     * @brief Resizes an image to fit_long_side(source.size(), image_size).
     *
     * When that size equals the source's, the pixels are copied unchanged.
     * @param image_size Must pass is_valid_side; otherwise throws Error.
     * @param destination Reset to the fitted size; must pass check_source.
     */
    void resize_long_side(ImageView source, std::int32_t image_size, Image& destination);
} // namespace SushiAI::Imaging
