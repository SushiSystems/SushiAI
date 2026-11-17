/**************************************************************************/
/* darknet_letterbox.hpp                                                  */
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
 * @file darknet_letterbox.hpp
 * @brief Declares the letterbox of Darknet's letterbox_image: truncated sides, grey padding.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/imaging/fit_transform.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/label_set.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Vision::Imaging
{
    /** @brief Holds the value of every padding byte, 0.5 of Darknet's range rounded up. */
    inline constexpr std::uint8_t K_DARKNET_PADDING = 128;

    /** @brief Holds everything apply_darknet_letterbox needs; nothing in it is random. */
    struct DarknetLetterboxPlan
    {
        /** @brief Holds the size of the image the plan was made for. */
        ImageSize source{};

        /** @brief Holds the size of the result. */
        ImageSize target{};

        /** @brief Holds the size after the resize and before the padding. */
        ImageSize resized{};

        /** @brief Counts the padding rows above the image. */
        std::int32_t top = 0;

        /** @brief Counts the padding columns left of the image. */
        std::int32_t left = 0;
    };

    /**
     * @brief Computes the resized size and the offsets of Darknet's letterbox_image: the
     *        tighter axis takes the target side, the other scales by it and truncates, and the
     *        offsets are (target - resized) / 2, truncated. The image is scaled up too.
     * @return The plan, or InvalidDescription unless both sizes are valid and the resized
     *         sides are positive.
     */
    [[nodiscard]] Result<DarknetLetterboxPlan> plan_darknet_letterbox(ImageSize source,
                                                                      ImageSize target) noexcept;

    /**
     * @brief Resizes an image to plan.resized with resize_darknet and embeds it in a grey target.
     *
     * The resized image starts at row plan.top and column plan.left; every other byte is
     * K_DARKNET_PADDING.
     * @pre source.size() is plan.source and check_source passes.
     */
    void apply_darknet_letterbox(ImageView source, const DarknetLetterboxPlan& plan,
                                 Image& destination) noexcept;

    /**
     * @brief Moves boxes and polygons into the letterboxed image.
     *
     * With sx = float(resized.width) / source.width and sy likewise: x becomes x * sx + left
     * and y becomes y * sy + top, in float, multiply first. Classes and order are kept;
     * nothing is clipped or removed.
     */
    [[nodiscard]] Labels apply_darknet_letterbox(const Labels& labels,
                                                 const DarknetLetterboxPlan& plan) noexcept;

    /** @brief Returns the placement apply_darknet_letterbox gave the source: two scales and the offsets. */
    [[nodiscard]] FitTransform darknet_letterbox_transform(
        const DarknetLetterboxPlan& plan) noexcept;
} // namespace SushiAI::Vision::Imaging
