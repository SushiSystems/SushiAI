/**************************************************************************/
/* darknet_augment.hpp                                                    */
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
 * @file darknet_augment.hpp
 * @brief Declares the training augmentation of Darknet: a jittered crop and a colour distortion.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/imaging/fit_transform.hpp>
#include <SushiAI/vision/imaging/hsv.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/label_set.hpp>
#include <SushiAI/vision/imaging/random.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Vision::Imaging
{
    /** @brief Holds what a .cfg's jitter, hue, saturation and exposure choose. */
    struct DarknetAugmentSettings
    {
        /** @brief Holds the fraction of a side each crop edge moves by, at most, in [0, 0.5). */
        float jitter = 0.0f;

        /** @brief Holds the half-range of the hue shift as a fraction of the hue circle. */
        float hue = 0.0f;

        /** @brief Holds the bound of the saturation factor, which is drawn in [1 / s, s]. */
        float saturation = 1.0f;

        /** @brief Holds the bound of the exposure factor, which is drawn in [1 / e, e]. */
        float exposure = 1.0f;
    };

    /** @brief Holds everything apply_darknet_augment needs: the crop and the colour. */
    struct DarknetAugmentPlan
    {
        /** @brief Holds the size of the image the plan was made for. */
        ImageSize source{};

        /** @brief Holds the drawn left edge of the crop, in pixels of the source, not truncated. */
        float shift_x = 0.0f;

        /** @brief Holds the drawn top edge of the crop, in pixels of the source, not truncated. */
        float shift_y = 0.0f;

        /** @brief Holds the first column the pixels are cropped from; shift_x truncated. */
        std::int32_t left = 0;

        /** @brief Holds the first row the pixels are cropped from; shift_y truncated. */
        std::int32_t top = 0;

        /** @brief Holds the width of the crop in pixels, at least 1. */
        std::int32_t crop_width = 1;

        /** @brief Holds the height of the crop in pixels, at least 1. */
        std::int32_t crop_height = 1;

        /** @brief Holds the colour distortion: the hue shift and the two factors less one. */
        HSVPlan color{};
    };

    /**
     * @brief Returns InvalidDescription unless the settings state a crop and factors Darknet
     *        can draw.
     *
     * The jitter is finite and in [0, 0.5), so a crop keeps a positive width; the hue is finite
     * and not negative; the saturation and the exposure are finite and at least 1.
     */
    [[nodiscard]] Status validate_darknet_augment(const DarknetAugmentSettings& settings) noexcept;

    /**
     * @brief Draws one sample's crop and colour as Darknet's load_data_detection does: four
     *        shifts, the hue shift, then the saturation and exposure factors as rand_scale, in
     *        that order, and the draws are made when the settings are the identity.
     * @return The plan, or InvalidDescription, before any draw, for what
     *         validate_darknet_augment rejects, an invalid source size or a crop beyond an
     *         image's bounds.
     */
    [[nodiscard]] Result<DarknetAugmentPlan> plan_darknet_augment(
        const DarknetAugmentSettings& settings, ImageSize source, Random& random) noexcept;

    /**
     * @brief Crops as Darknet's crop_image does, repeating the edge pixel outside the image,
     *        resizes to @p target with resize_darknet and recolours with apply_hsv.
     * @param destination Reset to @p target with the source's channels; must pass check_source.
     * @return InvalidDescription unless @p target passes is_valid_size and the plan was made
     *         for the size of @p source.
     */
    [[nodiscard]] Status apply_darknet_augment(ImageView source, const DarknetAugmentPlan& plan,
                                               ImageSize target, Image& destination) noexcept;

    /**
     * @brief Moves boxes and polygons into the cropped and resized image.
     *
     * With sx = float(target.width) / plan.crop_width and sy likewise: x becomes
     * (x - shift_x) * sx and y becomes (y - shift_y) * sy, in float. Classes and order are
     * kept; nothing is clipped or removed.
     */
    [[nodiscard]] Labels apply_darknet_augment(const Labels& labels,
                                               const DarknetAugmentPlan& plan,
                                               ImageSize target) noexcept;

    /** @brief Returns how the source sits in the augmented image: two scales and negative offsets. */
    [[nodiscard]] FitTransform darknet_augment_transform(const DarknetAugmentPlan& plan,
                                                         ImageSize target) noexcept;
} // namespace SushiAI::Vision::Imaging
