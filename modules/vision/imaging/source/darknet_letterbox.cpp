/**************************************************************************/
/* darknet_letterbox.cpp                                                  */
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
 * @file darknet_letterbox.cpp
 * @brief Implements the letterbox of Darknet's letterbox_image.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/darknet_letterbox.hpp>

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
        /** @brief Returns source * target / reference, truncated, in 64 bits. */
        [[nodiscard]] std::int32_t scaled_side(std::int32_t source, std::int32_t target,
                                               std::int32_t reference) noexcept
        {
            return static_cast<std::int32_t>(static_cast<std::int64_t>(source) * target /
                                             reference);
        }
    } // namespace

    Result<DarknetLetterboxPlan> plan_darknet_letterbox(ImageSize source,
                                                        ImageSize target) noexcept
    {
        SA_FAIL_IF(!is_valid_size(source), StatusCode::InvalidDescription,
                   "plan_darknet_letterbox: a source size of {}x{} is outside [1, {}]",
                   source.height, source.width, K_MAX_IMAGE_SIDE);
        SA_FAIL_IF(!is_valid_size(target), StatusCode::InvalidDescription,
                   "plan_darknet_letterbox: a target size of {}x{} is outside [1, {}]",
                   target.height, target.width, K_MAX_IMAGE_SIDE);

        DarknetLetterboxPlan plan;
        plan.source = source;
        plan.target = target;

        const float width_ratio = static_cast<float>(target.width) / static_cast<float>(source.width);
        const float height_ratio =
            static_cast<float>(target.height) / static_cast<float>(source.height);

        if (width_ratio < height_ratio)
        {
            plan.resized.width = target.width;
            plan.resized.height = scaled_side(source.height, target.width, source.width);
        }
        else
        {
            plan.resized.height = target.height;
            plan.resized.width = scaled_side(source.width, target.height, source.height);
        }

        SA_FAIL_IF(plan.resized.height <= 0 || plan.resized.width <= 0,
                   StatusCode::InvalidDescription,
                   "plan_darknet_letterbox: a {}x{} source in a {}x{} target leaves a resized "
                   "side of zero pixels",
                   source.height, source.width, target.height, target.width);

        plan.top = (target.height - plan.resized.height) / 2;
        plan.left = (target.width - plan.resized.width) / 2;
        return Result<DarknetLetterboxPlan>(plan);
    }

    void apply_darknet_letterbox(ImageView source, const DarknetLetterboxPlan& plan,
                                 Image& destination) noexcept
    {
        check_source(source, destination, "apply_darknet_letterbox");
        SA_FATAL_IF(source.size() != plan.source,
                    "apply_darknet_letterbox: the plan was made for a {}x{} source; the image is "
                    "{}x{}",
                    plan.source.height, plan.source.width, source.height(), source.width());
        SA_FATAL_IF(!is_valid_size(plan.resized) || !is_valid_size(plan.target) ||
                        plan.top < 0 || plan.left < 0 ||
                        plan.top + plan.resized.height > plan.target.height ||
                        plan.left + plan.resized.width > plan.target.width,
                    "apply_darknet_letterbox: the resized size {}x{} at row {} and column {} "
                    "does not lie in the target {}x{}",
                    plan.resized.height, plan.resized.width, plan.top, plan.left,
                    plan.target.height, plan.target.width);

        Image resized;
        const Status resize = resize_darknet(source, plan.resized, resized);
        SA_FATAL_IF(!resize.is_ok(), "apply_darknet_letterbox: {}", resize.message());

        destination.reset(plan.target, source.channels());
        destination.fill(K_DARKNET_PADDING);

        const auto channels = static_cast<std::size_t>(source.channels());
        const std::size_t content_bytes = static_cast<std::size_t>(plan.resized.width) * channels;
        const std::size_t left_bytes = static_cast<std::size_t>(plan.left) * channels;

        for (std::int32_t y = 0; y < plan.resized.height; ++y)
            std::memcpy(destination.row(plan.top + y) + left_bytes, resized.row(y), content_bytes);
    }

    Labels apply_darknet_letterbox(const Labels& labels, const DarknetLetterboxPlan& plan) noexcept
    {
        const FitTransform transform = darknet_letterbox_transform(plan);

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

    FitTransform darknet_letterbox_transform(const DarknetLetterboxPlan& plan) noexcept
    {
        FitTransform transform;
        transform.source = plan.source;
        transform.scale_x = static_cast<float>(plan.resized.width) /
                            static_cast<float>(plan.source.width);
        transform.scale_y = static_cast<float>(plan.resized.height) /
                            static_cast<float>(plan.source.height);
        transform.offset_x = static_cast<float>(plan.left);
        transform.offset_y = static_cast<float>(plan.top);
        return transform;
    }
} // namespace SushiAI::Vision::Imaging
