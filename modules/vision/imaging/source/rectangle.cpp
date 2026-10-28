/**************************************************************************/
/* rectangle.cpp                                                          */
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
 * @file rectangle.cpp
 * @brief Implements the batch order and the batch shapes of rectangular validation.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/rectangle.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI::Vision::Imaging
{
    namespace
    {
        /** @brief Returns the height of a size over its width, in double. */
        [[nodiscard]] double aspect_ratio(ImageSize size) noexcept
        {
            return static_cast<double>(size.height) / static_cast<double>(size.width);
        }

        /**
         * @brief Returns one side of a batch shape, a whole number of strides.
         *
         * The side is ceil(unit * image_size / stride + pad) * stride in double.
         * @param unit The side as a fraction of the long side, in (0, 1].
         * @return The side, or InvalidDescription when it is outside [1, K_MAX_IMAGE_SIDE].
         */
        [[nodiscard]] Result<std::int32_t> batch_side(double unit,
                                                      const RectangleSettings& settings)
        {
            const double strides =
                std::ceil(unit * settings.image_size / settings.stride + settings.pad);
            const double side = strides * settings.stride;

            SA_FAIL_IF(!(side >= 1.0 && side <= K_MAX_IMAGE_SIDE), StatusCode::InvalidDescription,
                       "plan_rectangular_batches: a batch side of {} is outside [1, {}]", side,
                       K_MAX_IMAGE_SIDE);

            return static_cast<std::int32_t>(side);
        }
    } // namespace

    Status validate_rectangle(const RectangleSettings& settings) noexcept
    {
        SA_FAIL_IF(!is_valid_side(settings.image_size), StatusCode::InvalidDescription,
                   "validate_rectangle: an image_size of {} is outside [1, {}]",
                   settings.image_size, K_MAX_IMAGE_SIDE);
        SA_FAIL_IF(settings.batch_size <= 0, StatusCode::InvalidDescription,
                   "validate_rectangle: a batch_size of {} is not above zero",
                   settings.batch_size);
        SA_FAIL_IF(settings.stride <= 0, StatusCode::InvalidDescription,
                   "validate_rectangle: a stride of {} is not above zero", settings.stride);
        SA_FAIL_IF(!(std::isfinite(settings.pad) && settings.pad >= 0.0),
                   StatusCode::InvalidDescription,
                   "validate_rectangle: a pad of {} is not a finite number at or above zero",
                   settings.pad);
        return Status();
    }

    Result<RectangularBatches> plan_rectangular_batches(const std::vector<ImageSize>& sizes,
                                                        const RectangleSettings& settings) noexcept
    {
        SA_TRY(validate_rectangle(settings));
        SA_FAIL_IF(sizes.empty(), StatusCode::InvalidDescription,
                   "plan_rectangular_batches: the dataset holds no image");

        for (std::size_t i = 0; i < sizes.size(); ++i)
        {
            SA_FAIL_IF(!is_valid_size(sizes[i]), StatusCode::InvalidDescription,
                       "plan_rectangular_batches: image {} has a size of {}x{}, outside [1, {}]",
                       i, sizes[i].height, sizes[i].width, K_MAX_IMAGE_SIDE);
        }

        RectangularBatches batches;
        batches.batch_size = static_cast<std::size_t>(settings.batch_size);
        batches.order.resize(sizes.size());
        std::iota(batches.order.begin(), batches.order.end(), std::size_t{0});
        std::stable_sort(batches.order.begin(), batches.order.end(),
                         [&sizes](std::size_t a, std::size_t b)
        {
            return aspect_ratio(sizes[a]) < aspect_ratio(sizes[b]);
        });

        batches.shapes.reserve((sizes.size() + batches.batch_size - 1) / batches.batch_size);

        for (std::size_t first = 0; first < sizes.size(); first += batches.batch_size)
        {
            // order is sorted, so a batch's first and last images hold its two extremes.
            const std::size_t last = std::min(first + batches.batch_size, sizes.size()) - 1;
            const double smallest = aspect_ratio(sizes[batches.order[first]]);
            const double largest = aspect_ratio(sizes[batches.order[last]]);

            double unit_height = 1.0;
            double unit_width = 1.0;
            if (largest < 1.0)
                unit_height = largest;
            else if (smallest > 1.0)
                unit_width = 1.0 / smallest;

            SA_TRY_ASSIGN(const std::int32_t height, batch_side(unit_height, settings));
            SA_TRY_ASSIGN(const std::int32_t width, batch_side(unit_width, settings));
            batches.shapes.push_back(ImageSize{height, width});
        }

        return Result<RectangularBatches>(std::move(batches));
    }
} // namespace SushiAI::Vision::Imaging
