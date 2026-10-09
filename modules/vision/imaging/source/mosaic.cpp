/**************************************************************************/
/* mosaic.cpp                                                             */
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
 * @file mosaic.cpp
 * @brief Implements the four-image mosaic: its draws, its placement and its composition.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/mosaic.hpp>

#include <algorithm>
#include <cstring>
#include <string_view>
#include <utility>

#include <SushiAI/core/common.hpp>

namespace SushiAI::Vision::Imaging
{
    namespace
    {
        /** @brief Holds the position of a patch relative to the centre the four patches share. */
        struct Quadrant
        {
            /** @brief Tells whether the patch lies left of the centre column. */
            bool is_left = false;

            /** @brief Tells whether the patch lies above the centre row. */
            bool is_top = false;
        };

        /** @brief Holds each patch's quadrant: top left, top right, bottom left, bottom right. */
        constexpr std::array<Quadrant, K_MOSAIC_PATCHES> K_QUADRANTS = {
            Quadrant{true, true}, Quadrant{false, true}, Quadrant{true, false},
            Quadrant{false, false}};

        /** @brief Holds the name each patch's source check opens its message with. */
        constexpr std::array<std::string_view, K_MOSAIC_PATCHES> K_PATCH_OPERATIONS = {
            "apply_mosaic: patch 0", "apply_mosaic: patch 1", "apply_mosaic: patch 2",
            "apply_mosaic: patch 3"};

        /** @brief Holds one axis of a placement: the canvas interval and the patch interval. */
        struct AxisSpan
        {
            /** @brief Holds the first canvas coordinate written. */
            std::int32_t canvas_begin = 0;

            /** @brief Holds the canvas coordinate after the last one written. */
            std::int32_t canvas_end = 0;

            /** @brief Holds the first patch coordinate read. */
            std::int32_t patch_begin = 0;

            /** @brief Holds the patch coordinate after the last one read. */
            std::int32_t patch_end = 0;
        };

        /**
         * @brief Places one axis of a patch against the centre (data/augment.py:525-538).
         *
         * A patch before the centre ends at it and keeps its last pixels; a patch after the
         * centre starts at it and keeps its first pixels. The canvas edge cuts both.
         * @param extent The patch's length on this axis, above zero.
         * @param canvas_side The canvas length on this axis, with center in [0, canvas_side].
         */
        [[nodiscard]] AxisSpan place_axis(bool is_before_center, std::int32_t center,
                                          std::int32_t extent, std::int32_t canvas_side) noexcept
        {
            AxisSpan span;
            if (is_before_center)
            {
                span.canvas_begin = std::max(center - extent, 0);
                span.canvas_end = center;
                span.patch_begin = extent - (span.canvas_end - span.canvas_begin);
                span.patch_end = extent;
            }
            else
            {
                span.canvas_begin = center;
                span.canvas_end = std::min(center + extent, canvas_side);
                span.patch_begin = 0;
                span.patch_end = std::min(extent, span.canvas_end - span.canvas_begin);
            }

            return span;
        }

        /** @brief Draws one centre coordinate: a uniform in [low, high) truncated toward zero. */
        [[nodiscard]] std::int32_t draw_center(std::int32_t low, std::int32_t high,
                                               Random& random) noexcept
        {
            const float drawn = random.uniform(static_cast<float>(low), static_cast<float>(high));
            return static_cast<std::int32_t>(drawn);
        }

        /** @brief Throws Error unless a placement stays inside its patch and the canvas. */
        void check_placement(const MosaicPlacement& placement, ImageSize patch,
                             std::int32_t canvas_side, std::size_t patch_index)
        {
            const std::int32_t canvas_width = placement.canvas_x2 - placement.canvas_x1;
            const std::int32_t canvas_height = placement.canvas_y2 - placement.canvas_y1;
            const bool is_canvas_inside =
                placement.canvas_x1 >= 0 && placement.canvas_y1 >= 0 && canvas_width >= 0 &&
                canvas_height >= 0 && placement.canvas_x2 <= canvas_side &&
                placement.canvas_y2 <= canvas_side;
            SA_THROW_IF(!is_canvas_inside,
                        "apply_mosaic: patch {} writes canvas columns [{}, {}) and rows [{}, {}), "
                        "outside a canvas of side {}",
                        patch_index, placement.canvas_x1, placement.canvas_x2,
                        placement.canvas_y1, placement.canvas_y2, canvas_side);

            const bool is_patch_inside =
                placement.patch_x1 >= 0 && placement.patch_y1 >= 0 &&
                placement.patch_x2 <= patch.width && placement.patch_y2 <= patch.height;
            SA_THROW_IF(!is_patch_inside,
                        "apply_mosaic: patch {} is {}x{} but its placement reads columns [{}, {}) "
                        "and rows [{}, {}); the plan was made for another size",
                        patch_index, patch.height, patch.width, placement.patch_x1,
                        placement.patch_x2, placement.patch_y1, placement.patch_y2);

            const bool is_same_extent = placement.patch_x2 - placement.patch_x1 == canvas_width &&
                                        placement.patch_y2 - placement.patch_y1 == canvas_height;
            SA_THROW_IF(!is_same_extent,
                        "apply_mosaic: patch {} reads {}x{} pixels but writes {}x{}; a mosaic "
                        "resamples nothing",
                        patch_index, placement.patch_y2 - placement.patch_y1,
                        placement.patch_x2 - placement.patch_x1, canvas_height, canvas_width);
        }

        /** @brief Copies one patch rectangle to its canvas rectangle, row by row. */
        void copy_patch(ImageView patch, const MosaicPlacement& placement, Image& canvas) noexcept
        {
            const std::size_t channels = static_cast<std::size_t>(patch.channels());
            const std::size_t bytes =
                static_cast<std::size_t>(placement.canvas_x2 - placement.canvas_x1) * channels;
            if (bytes == 0)
                return;

            const std::size_t canvas_skip =
                static_cast<std::size_t>(placement.canvas_x1) * channels;
            const std::size_t patch_skip = static_cast<std::size_t>(placement.patch_x1) * channels;
            const std::int32_t rows = placement.canvas_y2 - placement.canvas_y1;
            for (std::int32_t row = 0; row < rows; ++row)
            {
                std::memcpy(canvas.row(placement.canvas_y1 + row) + canvas_skip,
                            patch.row(placement.patch_y1 + row) + patch_skip, bytes);
            }
        }
    } // namespace

    void validate_mosaic(const MosaicSettings& settings)
    {
        SA_THROW_IF(!(settings.probability >= 0.0f && settings.probability <= 1.0f),
                    "validate_mosaic: a probability of {} is outside [0, 1]",
                    settings.probability);
    }

    void validate_mosaic_size(std::int32_t image_size)
    {
        SA_THROW_IF(!is_valid_side(2 * static_cast<std::int64_t>(image_size)),
                    "validate_mosaic_size: an image_size of {} is outside [1, {}]; the canvas "
                    "has twice that side",
                    image_size, K_MAX_MOSAIC_IMAGE_SIZE);
    }

    MosaicDraw draw_mosaic(const MosaicSettings& settings, std::int32_t image_size,
                           std::size_t index, PartnerSource& partners, Random& random)
    {
        validate_mosaic(settings);
        validate_mosaic_size(image_size);

        // The partners are chosen before the gate, so the source sees every sample.
        const MosaicPartners chosen = partners.choose(index, random);

        MosaicDraw draw;
        draw.is_enabled = random.bernoulli(settings.probability);
        if (!draw.is_enabled)
            return draw;

        draw.partners = chosen;

        const std::int32_t border = (image_size + 1) / 2;
        draw.center_y = draw_center(border, 2 * image_size - border, random);
        draw.center_x = draw_center(border, 2 * image_size - border, random);
        return draw;
    }

    MosaicPlan plan_mosaic(const MosaicSettings& settings, const MosaicDraw& draw,
                           std::int32_t image_size,
                           const std::array<ImageSize, K_MOSAIC_PATCHES>& patch_sizes)
    {
        SA_THROW_IF(!draw.is_enabled,
                    "plan_mosaic: the draw is not enabled, so the sample is not a mosaic and has "
                    "no placement");
        validate_mosaic_size(image_size);

        const std::int32_t canvas_side = 2 * image_size;
        const bool is_center_inside = draw.center_x >= 0 && draw.center_x <= canvas_side &&
                                      draw.center_y >= 0 && draw.center_y <= canvas_side;
        SA_THROW_IF(!is_center_inside,
                    "plan_mosaic: the centre ({}, {}) (x, y) is outside the canvas of side {}",
                    draw.center_x, draw.center_y, canvas_side);

        MosaicPlan plan;
        plan.image_size = image_size;
        plan.fill_value = settings.fill_value;
        for (std::size_t i = 0; i < K_MOSAIC_PATCHES; ++i)
        {
            const ImageSize patch = patch_sizes[i];
            SA_THROW_IF(!is_valid_size(patch),
                        "plan_mosaic: patch {} has a size of {}x{}, outside [1, {}]", i,
                        patch.height, patch.width, K_MAX_IMAGE_SIDE);

            const AxisSpan columns =
                place_axis(K_QUADRANTS[i].is_left, draw.center_x, patch.width, canvas_side);
            const AxisSpan rows =
                place_axis(K_QUADRANTS[i].is_top, draw.center_y, patch.height, canvas_side);

            MosaicPlacement& placement = plan.placements[i];
            placement.canvas_x1 = columns.canvas_begin;
            placement.canvas_x2 = columns.canvas_end;
            placement.patch_x1 = columns.patch_begin;
            placement.patch_x2 = columns.patch_end;
            placement.canvas_y1 = rows.canvas_begin;
            placement.canvas_y2 = rows.canvas_end;
            placement.patch_y1 = rows.patch_begin;
            placement.patch_y2 = rows.patch_end;
            placement.offset_x = placement.canvas_x1 - placement.patch_x1;
            placement.offset_y = placement.canvas_y1 - placement.patch_y1;
        }

        return plan;
    }

    void apply_mosaic(const std::array<ImageView, K_MOSAIC_PATCHES>& patches,
                      const MosaicPlan& plan, Image& destination)
    {
        validate_mosaic_size(plan.image_size);

        const std::int32_t canvas_side = 2 * plan.image_size;
        const std::int32_t channels = patches[0].channels();
        for (std::size_t i = 0; i < K_MOSAIC_PATCHES; ++i)
        {
            check_source(patches[i], destination, K_PATCH_OPERATIONS[i]);
            SA_THROW_IF(patches[i].channels() != channels,
                        "apply_mosaic: patch {} has {} channels but patch 0 has {}; a mosaic "
                        "needs one channel count",
                        i, patches[i].channels(), channels);
            check_placement(plan.placements[i], patches[i].size(), canvas_side, i);
        }

        destination.reset(ImageSize{canvas_side, canvas_side}, channels);
        destination.fill(plan.fill_value);
        for (std::size_t i = 0; i < K_MOSAIC_PATCHES; ++i)
            copy_patch(patches[i], plan.placements[i], destination);
    }

    Labels apply_mosaic(const std::array<Labels, K_MOSAIC_PATCHES>& labels, const MosaicPlan& plan)
    {
        validate_mosaic_size(plan.image_size);

        std::size_t total = 0;
        for (const Labels& patch : labels)
            total += patch.size();

        Labels joined;
        joined.reserve(total);
        for (std::size_t i = 0; i < K_MOSAIC_PATCHES; ++i)
        {
            const float offset_x = static_cast<float>(plan.placements[i].offset_x);
            const float offset_y = static_cast<float>(plan.placements[i].offset_y);
            for (std::size_t j = 0; j < labels[i].size(); ++j)
            {
                Polygon polygon = labels[i].polygons()[j];
                map_polygon(polygon, 1.0f, 1.0f, offset_x, offset_y);
                joined.add(labels[i].classes()[j],
                           translate_box(labels[i].boxes()[j], offset_x, offset_y),
                           std::move(polygon));
            }
        }

        const std::int32_t canvas_side = 2 * plan.image_size;
        clip_labels(joined, ImageSize{canvas_side, canvas_side});
        remove_empty_boxes(joined);
        return joined;
    }
} // namespace SushiAI::Vision::Imaging
