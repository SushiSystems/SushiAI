/**************************************************************************/
/* flip.cpp                                                               */
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
 * @file flip.cpp
 * @brief Implements the draw of the horizontal flip and its mirroring of pixels and labels.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/flip.hpp>

#include <cstddef>
#include <cstring>
#include <utility>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI::Vision::Imaging
{
    Status validate_flip(const FlipSettings& settings)
    {
        SA_FAIL_IF(!(settings.probability >= 0.0f && settings.probability <= 1.0f),
                   StatusCode::InvalidDescription,
                   "validate_flip: a probability of {} is outside [0, 1]", settings.probability);
        return Status();
    }

    Result<FlipPlan> plan_flip(const FlipSettings& settings, ImageSize source, Random& random)
    {
        SA_TRY(validate_flip(settings));
        SA_FAIL_IF(!is_valid_size(source), StatusCode::InvalidDescription,
                   "plan_flip: a source size of {}x{} is outside [1, {}]", source.height,
                   source.width, K_MAX_IMAGE_SIDE);

        FlipPlan plan;
        plan.source = source;
        plan.is_flipped = random.bernoulli(settings.probability);
        return Result<FlipPlan>(plan);
    }

    void apply_flip(ImageView source, const FlipPlan& plan, Image& destination)
    {
        check_source(source, destination, "apply_flip");
        SA_FATAL_IF(source.size() != plan.source,
                    "apply_flip: the plan was made for a {}x{} source; the image is {}x{}",
                    plan.source.height, plan.source.width, source.height(), source.width());

        destination.reset(source.size(), source.channels());

        if (!plan.is_flipped)
        {
            std::memcpy(destination.data(), source.data(), source.byte_count());
            return;
        }

        const auto channels = static_cast<std::size_t>(source.channels());
        const std::size_t row_bytes = source.row_bytes();

        for (std::int32_t y = 0; y < source.height(); ++y)
        {
            const std::uint8_t* in = source.row(y) + row_bytes;
            std::uint8_t* out = destination.row(y);

            for (std::int32_t x = 0; x < source.width(); ++x)
            {
                in -= channels;
                for (std::size_t channel = 0; channel < channels; ++channel)
                    out[channel] = in[channel];

                out += channels;
            }
        }
    }

    Labels apply_flip(const Labels& labels, const FlipPlan& plan)
    {
        if (!plan.is_flipped)
            return labels;

        const auto width = static_cast<float>(plan.source.width);

        Labels mirrored;
        mirrored.reserve(labels.size());

        for (std::size_t i = 0; i < labels.size(); ++i)
        {
            const PixelBox& box = labels.boxes()[i];
            Polygon polygon = labels.polygons()[i];

            // x * -1 + width is width - x exactly, since a negation does not round.
            map_polygon(polygon, -1.0f, 1.0f, width, 0.0f);
            mirrored.add(labels.classes()[i],
                         PixelBox{width - box.x2, box.y1, width - box.x1, box.y2},
                         std::move(polygon));
        }

        return mirrored;
    }
} // namespace SushiAI::Vision::Imaging
