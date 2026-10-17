/**************************************************************************/
/* affine.cpp                                                             */
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
 * @file affine.cpp
 * @brief Implements the random scale and translation that produce the training-size image.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/affine.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string_view>
#include <utility>
#include <vector>

#include <SushiAI/failure/error.hpp>

#include "bilinear.hpp"

namespace SushiAI::Vision::Imaging
{
    namespace
    {
        /** @brief Holds the two source neighbours one output coordinate reads on one axis. */
        struct Tap
        {
            /** @brief Holds the lower neighbour's index, in [-1, extent - 1] when covered. */
            std::int32_t first = 0;

            /** @brief Holds the weight of neighbour first + 1, in [0, Detail::K_WEIGHT_ONE]. */
            std::uint32_t second_weight = 0;

            /** @brief Reports whether at least one of the two neighbours may lie in the source. */
            bool is_covered = false;
        };

        /**
         * @brief Throws Error unless the plan's matrix is the plan's scale and a translation.
         * @param operation Names the calling function at the start of the message.
         */
        void check_matrix(const AffinePlan& plan, std::string_view operation)
        {
            const std::array<float, 6>& m = plan.matrix;

            SA_THROW_IF(!(m[0] > 0.0f) || !(m[4] > 0.0f) || !std::isfinite(m[0]) ||
                            !std::isfinite(m[4]),
                        "{}: the matrix scales must be finite and above zero; got {} and {}",
                        operation, m[0], m[4]);
            SA_THROW_IF(m[0] != plan.scale || m[4] != plan.scale,
                        "{}: the matrix scales {} and {} must equal the plan's scale {}",
                        operation, m[0], m[4], plan.scale);
            SA_THROW_IF(m[1] != 0.0f || m[3] != 0.0f,
                        "{}: the warp is a scale and a translation only; the matrix has "
                        "off-diagonal terms {} and {}",
                        operation, m[1], m[3]);
            SA_THROW_IF(!std::isfinite(m[2]) || !std::isfinite(m[5]),
                        "{}: the matrix needs a finite translation; got {} and {}", operation,
                        m[2], m[5]);
        }

        /**
         * @brief Maps one output coordinate back to the source and returns its two neighbours.
         * @param offset The matrix translation of the axis.
         * @param scale The matrix scale of the axis, above zero.
         * @param extent The source length along the axis.
         */
        [[nodiscard]] Tap make_tap(std::int32_t output_index, double offset, double scale,
                                   std::int32_t extent) noexcept
        {
            const double position = (static_cast<double>(output_index) - offset) / scale;
            const double lower = std::floor(position);

            if (!(lower >= -1.0 && lower <= static_cast<double>(extent - 1)))
                return Tap{};

            return Tap{static_cast<std::int32_t>(lower), Detail::to_weight(position - lower), true};
        }

        /**
         * @brief Blends the two column neighbours of one source row for one channel.
         * @param row The source row, or null when the row lies outside the source.
         * @return The weighted sum, at most 255 * Detail::K_WEIGHT_ONE.
         */
        [[nodiscard]] std::uint32_t blend_columns(const std::uint8_t* row, const Tap& column,
                                                  std::int32_t width, std::int32_t channels,
                                                  std::int32_t channel,
                                                  std::uint32_t border) noexcept
        {
            std::uint32_t left = border;
            std::uint32_t right = border;

            if (row != nullptr && column.first >= 0)
                left = row[static_cast<std::size_t>(column.first) * channels + channel];

            if (row != nullptr && column.first + 1 < width)
                right = row[static_cast<std::size_t>(column.first + 1) * channels + channel];

            return Detail::blend_pair(left, right, column.second_weight);
        }

        /** @brief Holds what the warp makes of one object: its box and its polygon. */
        struct WarpedObject
        {
            /** @brief Holds the box in the output, clipped to it. */
            PixelBox box{};

            /** @brief Holds the polygon in the output, clamped to the box; empty without one. */
            Polygon polygon;
        };

        /**
         * @brief Warps one object that has no polygon: its two corners, clipped to the output.
         * @pre Both matrix scales are above zero, so the two warped corners bound all four.
         */
        [[nodiscard]] WarpedObject warp_box(const PixelBox& box, const AffinePlan& plan) noexcept
        {
            const std::array<float, 6>& m = plan.matrix;
            const PixelBox warped{m[0] * box.x1 + m[2], m[4] * box.y1 + m[5],
                                  m[0] * box.x2 + m[2], m[4] * box.y2 + m[5]};

            return WarpedObject{clip_box(warped, plan.output), Polygon{}};
        }

        /**
         * @brief Warps one object that has a polygon and takes its box from the warped polygon.
         *
         * The box is visible_polygon_box clipped to the output, and the polygon is clamped to
         * that box: the reference's apply_segments (data/augment.py:1313-1322) followed by
         * Instances.clip (utils/instance.py:426-429).
         */
        [[nodiscard]] WarpedObject warp_polygon(Polygon polygon, const AffinePlan& plan) noexcept
        {
            const std::array<float, 6>& m = plan.matrix;

            map_polygon(polygon, m[0], m[4], m[2], m[5]);
            const PixelBox box = clip_box(visible_polygon_box(polygon, plan.output), plan.output);
            clamp_polygon(polygon, box);

            return WarpedObject{box, std::move(polygon)};
        }
    } // namespace

    void validate_affine(const AffineSettings& settings)
    {
        SA_THROW_IF(!(settings.scale >= 0.0f && settings.scale < 1.0f),
                    "validate_affine: a scale half-range of {} is outside [0, 1)", settings.scale);
        SA_THROW_IF(!(settings.translate >= 0.0f && settings.translate <= 0.5f),
                    "validate_affine: a translate half-range of {} is outside [0, 0.5]",
                    settings.translate);
    }

    AffinePlan plan_affine(const AffineSettings& settings, ImageSize source, ImageSize output,
                           Random& random)
    {
        validate_affine(settings);
        SA_THROW_IF(!is_valid_size(source),
                    "plan_affine: a source size of {}x{} is outside [1, {}]", source.height,
                    source.width, K_MAX_IMAGE_SIDE);
        SA_THROW_IF(!is_valid_size(output),
                    "plan_affine: an output size of {}x{} is outside [1, {}]", output.height,
                    output.width, K_MAX_IMAGE_SIDE);

        AffinePlan plan;
        plan.source = source;
        plan.output = output;
        plan.border_value = settings.border_value;
        plan.filter = settings.filter;
        plan.polygon_filter = settings.polygon_filter;

        // The three draws are made in this order: scale, translate x, translate y.
        plan.scale = random.uniform(1.0f - settings.scale, 1.0f + settings.scale);
        plan.translate_x = random.uniform(0.5f - settings.translate, 0.5f + settings.translate) *
                           static_cast<float>(output.width);
        plan.translate_y = random.uniform(0.5f - settings.translate, 0.5f + settings.translate) *
                           static_cast<float>(output.height);

        const float center_x = -static_cast<float>(source.width) / 2.0f;
        const float center_y = -static_cast<float>(source.height) / 2.0f;

        plan.matrix = {plan.scale, 0.0f, plan.scale * center_x + plan.translate_x,
                       0.0f, plan.scale, plan.scale * center_y + plan.translate_y};

        return plan;
    }

    void apply_affine(ImageView source, const AffinePlan& plan, Image& destination)
    {
        check_source(source, destination, "apply_affine");
        SA_THROW_IF(source.size() != plan.source,
                    "apply_affine: the plan was made for a {}x{} source; the image is {}x{}",
                    plan.source.height, plan.source.width, source.height(), source.width());
        SA_THROW_IF(!is_valid_size(plan.output),
                    "apply_affine: an output size of {}x{} is outside [1, {}]", plan.output.height,
                    plan.output.width, K_MAX_IMAGE_SIDE);
        check_matrix(plan, "apply_affine");

        const std::int32_t channels = source.channels();
        const std::int32_t width = source.width();
        const std::uint8_t border = plan.border_value;

        destination.reset(plan.output, channels);

        std::vector<Tap> columns(static_cast<std::size_t>(plan.output.width));

        for (std::int32_t x = 0; x < plan.output.width; ++x)
            columns[static_cast<std::size_t>(x)] =
                make_tap(x, plan.matrix[2], plan.matrix[0], width);

        for (std::int32_t y = 0; y < plan.output.height; ++y)
        {
            const Tap row = make_tap(y, plan.matrix[5], plan.matrix[4], source.height());
            std::uint8_t* out = destination.row(y);

            if (!row.is_covered)
            {
                std::fill_n(out, destination.row_bytes(), border);
                continue;
            }

            const std::uint8_t* upper = row.first >= 0 ? source.row(row.first) : nullptr;
            const std::uint8_t* lower =
                row.first + 1 < source.height() ? source.row(row.first + 1) : nullptr;

            for (const Tap& column : columns)
            {
                for (std::int32_t channel = 0; channel < channels; ++channel, ++out)
                {
                    if (!column.is_covered)
                    {
                        *out = border;
                        continue;
                    }

                    *out = Detail::blend_rows(
                        blend_columns(upper, column, width, channels, channel, border),
                        blend_columns(lower, column, width, channels, channel, border),
                        row.second_weight);
                }
            }
        }
    }

    Labels apply_affine(const Labels& labels, const AffinePlan& plan)
    {
        check_matrix(plan, "apply_affine");

        Labels kept;
        kept.reserve(labels.size());

        for (std::size_t i = 0; i < labels.size(); ++i)
        {
            const PixelBox& box = labels.boxes()[i];
            const Polygon& polygon = labels.polygons()[i];
            const bool has_polygon = !polygon.empty();

            WarpedObject warped = has_polygon ? warp_polygon(polygon, plan) : warp_box(box, plan);
            const CandidateFilter& filter = has_polygon ? plan.polygon_filter : plan.filter;

            if (!is_box_candidate(scale_box(box, plan.scale, plan.scale), warped.box, filter))
                continue;

            kept.add(labels.classes()[i], warped.box, std::move(warped.polygon));
        }

        return kept;
    }
} // namespace SushiAI::Vision::Imaging
