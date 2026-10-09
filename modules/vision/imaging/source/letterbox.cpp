/**************************************************************************/
/* letterbox.cpp                                                          */
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
 * @file letterbox.cpp
 * @brief Implements the letterbox plan, its resize and padding of an image, and its box mapping.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/letterbox.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <utility>

#include <SushiAI/core/common.hpp>

#include "bilinear.hpp"

namespace SushiAI::Vision::Imaging
{
    namespace
    {
        /** @brief Rounds a non-negative value to the nearest integer, a half to the even one. */
        [[nodiscard]] std::int32_t round_half_even(double value) noexcept
        {
            const double below = std::floor(value);
            const double fraction = value - below;
            const auto whole = static_cast<std::int32_t>(below);

            if (fraction > 0.5)
                return whole + 1;

            if (fraction < 0.5)
                return whole;

            return whole % 2 == 0 ? whole : whole + 1;
        }

        /** @brief Reports whether the resized image and its four paddings tile a valid target. */
        [[nodiscard]] bool is_tiling(const LetterboxPlan& plan) noexcept
        {
            const bool is_positive = plan.resized.height > 0 && plan.resized.width > 0 &&
                                     plan.top >= 0 && plan.bottom >= 0 && plan.left >= 0 &&
                                     plan.right >= 0;

            return is_positive && is_valid_size(plan.target) &&
                   plan.top + plan.resized.height + plan.bottom == plan.target.height &&
                   plan.left + plan.resized.width + plan.right == plan.target.width;
        }
    } // namespace

    void validate_letterbox(const LetterboxSettings& settings) noexcept
    {
        static_cast<void>(settings);
    }

    LetterboxPlan plan_letterbox(const LetterboxSettings& settings, ImageSize source,
                                 ImageSize target)
    {
        validate_letterbox(settings);
        SA_THROW_IF(!is_valid_size(source),
                    "plan_letterbox: a source size of {}x{} is outside [1, {}]", source.height,
                    source.width, K_MAX_IMAGE_SIDE);
        SA_THROW_IF(!is_valid_size(target),
                    "plan_letterbox: a target size of {}x{} is outside [1, {}]", target.height,
                    target.width, K_MAX_IMAGE_SIDE);

        double ratio = std::min(static_cast<double>(target.height) / source.height,
                                static_cast<double>(target.width) / source.width);
        if (!settings.scale_up)
            ratio = std::min(ratio, 1.0);

        LetterboxPlan plan;
        plan.source = source;
        plan.target = target;
        plan.ratio = ratio;
        plan.resized.height = round_half_even(source.height * ratio);
        plan.resized.width = round_half_even(source.width * ratio);
        plan.padding_value = settings.padding_value;

        SA_THROW_IF(plan.resized.height <= 0 || plan.resized.width <= 0,
                    "plan_letterbox: a {}x{} source in a {}x{} target leaves a resized side of "
                    "zero pixels",
                    source.height, source.width, target.height, target.width);

        const std::int32_t pad_height = target.height - plan.resized.height;
        const std::int32_t pad_width = target.width - plan.resized.width;

        plan.top = pad_height / 2;
        plan.bottom = pad_height - plan.top;
        plan.left = pad_width / 2;
        plan.right = pad_width - plan.left;
        return plan;
    }

    void apply_letterbox(ImageView source, const LetterboxPlan& plan, Image& destination)
    {
        check_source(source, destination, "apply_letterbox");
        SA_THROW_IF(source.size() != plan.source,
                    "apply_letterbox: the plan was made for a {}x{} source; the image is {}x{}",
                    plan.source.height, plan.source.width, source.height(), source.width());
        SA_THROW_IF(!is_tiling(plan),
                    "apply_letterbox: the resized size {}x{} and the paddings do not tile the "
                    "target {}x{}",
                    plan.resized.height, plan.resized.width, plan.target.height,
                    plan.target.width);

        destination.reset(plan.target, source.channels());

        const auto channels = static_cast<std::size_t>(source.channels());
        const std::size_t left_bytes = static_cast<std::size_t>(plan.left) * channels;
        const std::size_t right_bytes = static_cast<std::size_t>(plan.right) * channels;
        const std::size_t content_bytes = static_cast<std::size_t>(plan.resized.width) * channels;
        const std::size_t row_bytes = destination.row_bytes();
        const bool is_copied = plan.resized == source.size();

        for (std::int32_t y = 0; y < plan.target.height; ++y)
        {
            std::uint8_t* row = destination.row(y);
            const std::int32_t content_y = y - plan.top;

            if (content_y < 0 || content_y >= plan.resized.height)
            {
                std::memset(row, plan.padding_value, row_bytes);
                continue;
            }

            std::memset(row, plan.padding_value, left_bytes);
            std::memset(row + left_bytes + content_bytes, plan.padding_value, right_bytes);

            if (is_copied)
                std::memcpy(row + left_bytes, source.row(content_y), content_bytes);
        }

        // The kernel writes content_bytes of each row and leaves the padding beside them.
        if (!is_copied)
            Detail::resize_rows(source, plan.resized, destination.row(plan.top) + left_bytes,
                                row_bytes);
    }

    Labels apply_letterbox(const Labels& labels, const LetterboxPlan& plan)
    {
        const auto ratio = static_cast<float>(plan.ratio);
        const auto offset_x = static_cast<float>(plan.left);
        const auto offset_y = static_cast<float>(plan.top);

        Labels moved;
        moved.reserve(labels.size());

        for (std::size_t i = 0; i < labels.size(); ++i)
        {
            Polygon polygon = labels.polygons()[i];
            map_polygon(polygon, ratio, ratio, offset_x, offset_y);
            moved.add(labels.classes()[i],
                      translate_box(scale_box(labels.boxes()[i], ratio, ratio), offset_x,
                                    offset_y),
                      std::move(polygon));
        }

        return moved;
    }
} // namespace SushiAI::Vision::Imaging
