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
 * @file resize.hpp
 * @brief Declares the resizes of an 8-bit image and the training-load size rule.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/imaging/image.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Vision::Imaging
{
    /** @brief Holds the number of fraction bits of a bilinear weight; the weights sum to 2048. */
    inline constexpr std::int32_t K_BILINEAR_WEIGHT_BITS = 11;

    /**
     * @brief Returns the size an image is loaded at so that it just fits inside @p target.
     *
     * With r = min(target.height / height, target.width / width) in double: when r == 1 the
     * size is returned as it is; otherwise each side becomes min(ceil(side * r), its target
     * side). A square target is the reference's long-side rule (data/base.py:297-299).
     * @pre Both sides of original and of target are positive.
     */
    [[nodiscard]] ImageSize fit_within(ImageSize original, ImageSize target) noexcept;

    /**
     * @brief Resizes an image by bilinear interpolation between pixel centres.
     *
     * Destination column x reads source position (x + 0.5) * (source width / width) - 0.5,
     * clamped to the edge columns; rows likewise. The kernel does not widen on a reduction.
     * @param destination Reset to size with the source's channels; must pass check_source.
     * @return InvalidDescription unless @p size passes is_valid_size.
     */
    [[nodiscard]] Status resize_bilinear(ImageView source, ImageSize size, Image& destination) noexcept;

    /**
     * @brief Resizes an image by giving each pixel the source pixel its centre falls in.
     *
     * Destination column x reads source column floor((x + 0.5) * source width / width),
     * and rows likewise. No two values are blended, so an image of class indices stays one.
     * @param size Must pass is_valid_size; otherwise Image::reset is fatal.
     * @param destination Reset to size with the source's channels; must pass check_source.
     */
    void resize_nearest(ImageView source, ImageSize size, Image& destination) noexcept;

    /**
     * @brief Resizes an image to fit_within(source.size(), target).
     *
     * When that size equals the source's, the pixels are copied unchanged.
     * @param destination Reset to the fitted size; must pass check_source.
     * @return InvalidDescription unless @p target passes is_valid_size.
     */
    [[nodiscard]] Status resize_within(ImageView source, ImageSize target,
                                       Image& destination) noexcept;
} // namespace SushiAI::Vision::Imaging
