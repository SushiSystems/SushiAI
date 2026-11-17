/**************************************************************************/
/* darknet_augment.cpp                                                    */
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
 * @file darknet_augment.cpp
 * @brief Implements the jittered crop and the colour distortion of Darknet's training.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/darknet_augment.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/darknet_resize.hpp>
#include <SushiAI/vision/imaging/polygon.hpp>

namespace SushiAI::Vision::Imaging
{
    namespace
    {
        /** @brief Holds the exclusive upper bound of the jitter, past which a crop may vanish. */
        constexpr float K_JITTER_LIMIT = 0.5f;

        /** @brief Holds the probability with which rand_scale keeps its scale and not its inverse. */
        constexpr float K_KEEP_SCALE = 0.5f;

        /** @brief Reports whether a value is a finite number that is not below @p lowest. */
        [[nodiscard]] bool is_at_least(float value, float lowest) noexcept
        {
            return std::isfinite(value) && value >= lowest;
        }

        /**
         * @brief Draws a factor in [1 / bound, bound] as Darknet's rand_scale does.
         *
         * A scale in [1, bound] is one draw; a second draw keeps it or takes its reciprocal.
         */
        [[nodiscard]] float draw_scale(float bound, Random& random) noexcept
        {
            const float scale = random.uniform(1.0f, bound);
            return random.bernoulli(K_KEEP_SCALE) ? scale : 1.0f / scale;
        }

        /** @brief Returns the source index of a crop position, repeating the edge outside. */
        [[nodiscard]] std::int32_t clamp_index(std::int64_t index, std::int32_t extent) noexcept
        {
            return static_cast<std::int32_t>(
                std::clamp<std::int64_t>(index, 0, static_cast<std::int64_t>(extent) - 1));
        }

        /** @brief Copies the crop of a plan out of the source, repeating the edge pixels. */
        void crop_edge_repeated(ImageView source, const DarknetAugmentPlan& plan, Image& cropped)
        {
            cropped.reset(ImageSize{plan.crop_height, plan.crop_width}, source.channels());
            const auto channels = static_cast<std::size_t>(source.channels());

            for (std::int32_t y = 0; y < plan.crop_height; ++y)
            {
                const std::uint8_t* const in =
                    source.row(clamp_index(static_cast<std::int64_t>(plan.top) + y,
                                           source.height()));
                std::uint8_t* out = cropped.row(y);

                for (std::int32_t x = 0; x < plan.crop_width; ++x)
                {
                    const auto column = static_cast<std::size_t>(clamp_index(
                        static_cast<std::int64_t>(plan.left) + x, source.width()));
                    std::memcpy(out, in + column * channels, channels);
                    out += channels;
                }
            }
        }
    } // namespace

    Status validate_darknet_augment(const DarknetAugmentSettings& settings) noexcept
    {
        SA_FAIL_IF(!is_at_least(settings.jitter, 0.0f) || !(settings.jitter < K_JITTER_LIMIT),
                   StatusCode::InvalidDescription,
                   "validate_darknet_augment: the jitter is {}; it is a number in [0, {})",
                   settings.jitter, K_JITTER_LIMIT);
        SA_FAIL_IF(!is_at_least(settings.hue, 0.0f), StatusCode::InvalidDescription,
                   "validate_darknet_augment: the hue is {}; it is a number from 0", settings.hue);
        SA_FAIL_IF(!is_at_least(settings.saturation, 1.0f) ||
                       !is_at_least(settings.exposure, 1.0f),
                   StatusCode::InvalidDescription,
                   "validate_darknet_augment: the saturation is {} and the exposure {}; each is "
                   "a number from 1",
                   settings.saturation, settings.exposure);
        return Status();
    }

    Result<DarknetAugmentPlan> plan_darknet_augment(const DarknetAugmentSettings& settings,
                                                    ImageSize source, Random& random) noexcept
    {
        SA_TRY(validate_darknet_augment(settings));
        SA_FAIL_IF(!is_valid_size(source), StatusCode::InvalidDescription,
                   "plan_darknet_augment: a source size of {}x{} is outside [1, {}]",
                   source.height, source.width, K_MAX_IMAGE_SIDE);

        const float width_reach = static_cast<float>(source.width) * settings.jitter;
        const float height_reach = static_cast<float>(source.height) * settings.jitter;
        const float shift_left = random.uniform(-width_reach, width_reach);
        const float shift_right = random.uniform(-width_reach, width_reach);
        const float shift_top = random.uniform(-height_reach, height_reach);
        const float shift_bottom = random.uniform(-height_reach, height_reach);

        DarknetAugmentPlan plan;
        plan.source = source;
        plan.shift_x = shift_left;
        plan.shift_y = shift_top;
        plan.left = static_cast<std::int32_t>(shift_left);
        plan.top = static_cast<std::int32_t>(shift_top);
        plan.crop_width = std::max(
            static_cast<std::int32_t>(static_cast<float>(source.width) - shift_left - shift_right),
            1);
        plan.crop_height = std::max(
            static_cast<std::int32_t>(static_cast<float>(source.height) - shift_top - shift_bottom),
            1);

        plan.color.hue = random.uniform(-settings.hue, settings.hue);
        plan.color.saturation = draw_scale(settings.saturation, random) - 1.0f;
        plan.color.value = draw_scale(settings.exposure, random) - 1.0f;

        SA_FAIL_IF(!is_valid_size(ImageSize{plan.crop_height, plan.crop_width}),
                   StatusCode::InvalidDescription,
                   "plan_darknet_augment: a {}x{} source with a jitter of {} crops {}x{}, "
                   "beyond [1, {}]",
                   source.height, source.width, settings.jitter, plan.crop_height,
                   plan.crop_width, K_MAX_IMAGE_SIDE);
        return Result<DarknetAugmentPlan>(plan);
    }

    Status apply_darknet_augment(ImageView source, const DarknetAugmentPlan& plan,
                                 ImageSize target, Image& destination) noexcept
    {
        check_source(source, destination, "apply_darknet_augment");
        SA_FAIL_IF(source.size() != plan.source, StatusCode::InvalidDescription,
                   "apply_darknet_augment: the plan was made for a {}x{} source; the image is "
                   "{}x{}",
                   plan.source.height, plan.source.width, source.height(), source.width());
        SA_FAIL_IF(!is_valid_size(target), StatusCode::InvalidDescription,
                   "apply_darknet_augment: a target size of {}x{} is outside [1, {}]",
                   target.height, target.width, K_MAX_IMAGE_SIDE);

        Image cropped;
        crop_edge_repeated(source, plan, cropped);

        Image resized;
        SA_TRY(resize_darknet(cropped.view(), target, resized));

        apply_hsv(resized.view(), plan.color, destination);
        return Status();
    }

    Labels apply_darknet_augment(const Labels& labels, const DarknetAugmentPlan& plan,
                                 ImageSize target) noexcept
    {
        const FitTransform transform = darknet_augment_transform(plan, target);

        Labels moved;
        moved.reserve(labels.size());

        for (std::size_t i = 0; i < labels.size(); ++i)
        {
            Outline outline = labels.outlines()[i];
            map_outline(outline, transform.scale_x, transform.scale_y, transform.offset_x,
                        transform.offset_y);
            moved.add(labels.classes()[i],
                      translate_box(scale_box(labels.boxes()[i], transform.scale_x,
                                              transform.scale_y),
                                    transform.offset_x, transform.offset_y),
                      std::move(outline));
        }

        return moved;
    }

    FitTransform darknet_augment_transform(const DarknetAugmentPlan& plan,
                                           ImageSize target) noexcept
    {
        FitTransform transform;
        transform.source = plan.source;
        transform.scale_x = static_cast<float>(target.width) /
                            static_cast<float>(plan.crop_width);
        transform.scale_y = static_cast<float>(target.height) /
                            static_cast<float>(plan.crop_height);
        transform.offset_x = -plan.shift_x * transform.scale_x;
        transform.offset_y = -plan.shift_y * transform.scale_y;
        return transform;
    }
} // namespace SushiAI::Vision::Imaging
