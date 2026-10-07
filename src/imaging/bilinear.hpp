/**************************************************************************/
/* bilinear.hpp                                                           */
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
 * @file bilinear.hpp
 * @brief Declares the fixed-point bilinear arithmetic the resize and the affine warp share.
 * @author Mustafa Garip
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <SushiAI/imaging/image.hpp>
#include <SushiAI/imaging/resize.hpp>

/** @brief Holds the units of the imaging module that no caller names. */
namespace SushiAI::Imaging::Detail
{
    /** @brief Holds the sum of the two weights of one axis, 2^K_BILINEAR_WEIGHT_BITS. */
    inline constexpr std::uint32_t K_WEIGHT_ONE = std::uint32_t{1} << K_BILINEAR_WEIGHT_BITS;

    /** @brief Holds the shift that removes the weights of both axes from a blended sum. */
    inline constexpr std::int32_t K_PRODUCT_SHIFT = 2 * K_BILINEAR_WEIGHT_BITS;

    /** @brief Holds one half in the fixed point of a weight product, for rounding. */
    inline constexpr std::uint32_t K_PRODUCT_HALF = std::uint32_t{1} << (K_PRODUCT_SHIFT - 1);

    /**
     * @brief Returns the weight of the upper neighbour, floor(fraction * 2048 + 0.5).
     * @param fraction The distance from the lower neighbour, in [0, 1).
     */
    [[nodiscard]] inline std::uint32_t to_weight(double fraction) noexcept
    {
        const double scaled = fraction * static_cast<double>(K_WEIGHT_ONE);
        return static_cast<std::uint32_t>(std::floor(scaled + 0.5));
    }

    /**
     * @brief Blends two neighbours of one axis: (2048 - w) * first + w * second.
     * @param second_weight The weight w of second, in [0, K_WEIGHT_ONE].
     * @return The weighted sum, at most 255 * K_WEIGHT_ONE.
     */
    [[nodiscard]] constexpr std::uint32_t blend_pair(std::uint32_t first, std::uint32_t second,
                                                     std::uint32_t second_weight) noexcept
    {
        return (K_WEIGHT_ONE - second_weight) * first + second_weight * second;
    }

    /**
     * @brief Blends two blend_pair sums along the other axis and rounds the result to a level.
     *
     * The largest sum is 2048 * 2048 * 255 + 2^21, below 2^32.
     * @param lower_weight The weight of lower, in [0, K_WEIGHT_ONE].
     */
    [[nodiscard]] constexpr std::uint8_t blend_rows(std::uint32_t upper, std::uint32_t lower,
                                                    std::uint32_t lower_weight) noexcept
    {
        const std::uint32_t sum = blend_pair(upper, lower, lower_weight) + K_PRODUCT_HALF;
        return static_cast<std::uint8_t>(sum >> K_PRODUCT_SHIFT);
    }

    /**
     * @brief Resizes an image into rows the caller owns, by the kernel of resize_bilinear.
     *
     * Row y of the result starts at destination + y * row_stride and takes size.width
     * pixels of the source's channel count. No byte between two rows is written.
     * @param source Valid by check_source; size is valid by is_valid_size.
     * @param row_stride At least size.width times the source's channel count.
     */
    void resize_rows(ImageView source, ImageSize size, std::uint8_t* destination,
                     std::size_t row_stride);
} // namespace SushiAI::Imaging::Detail
