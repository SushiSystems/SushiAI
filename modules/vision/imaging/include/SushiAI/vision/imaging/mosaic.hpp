/**************************************************************************/
/* mosaic.hpp                                                             */
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
 * @file mosaic.hpp
 * @brief Declares the four-image mosaic: its draws, its placement and its composition.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/label_set.hpp>
#include <SushiAI/vision/imaging/partner_source.hpp>
#include <SushiAI/vision/imaging/random.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Vision::Imaging
{
    /** @brief Holds the number of images one mosaic combines. */
    inline constexpr std::size_t K_MOSAIC_PATCHES = K_MOSAIC_PARTNERS + 1;

    /** @brief Holds what the caller chooses about the mosaic. */
    struct MosaicSettings
    {
        /** @brief Holds the probability that a sample is a mosaic (cfg/default.yaml:126). */
        float probability = 1.0f;

        /** @brief Holds the value of every canvas byte no patch covers. */
        std::uint8_t fill_value = 114;
    };

    /** @brief Holds what one mosaic draws before its partner images are loaded. */
    struct MosaicDraw
    {
        /** @brief Tells whether the sample is a mosaic; the other fields are zero when not. */
        bool is_enabled = false;

        /** @brief Holds the dataset indices of patches 1, 2 and 3, in draw order. */
        MosaicPartners partners{};

        /** @brief Holds the canvas column where the four patches meet. */
        std::int32_t center_x = 0;

        /** @brief Holds the canvas row where the four patches meet. */
        std::int32_t center_y = 0;
    };

    /** @brief Holds where one patch lands: half-open pixel rectangles, x2 and y2 excluded. */
    struct MosaicPlacement
    {
        /** @brief Holds the left column of the rectangle written on the canvas. */
        std::int32_t canvas_x1 = 0;

        /** @brief Holds the top row of the rectangle written on the canvas. */
        std::int32_t canvas_y1 = 0;

        /** @brief Holds the column after the last one written on the canvas. */
        std::int32_t canvas_x2 = 0;

        /** @brief Holds the row after the last one written on the canvas. */
        std::int32_t canvas_y2 = 0;

        /** @brief Holds the left column of the rectangle read from the patch. */
        std::int32_t patch_x1 = 0;

        /** @brief Holds the top row of the rectangle read from the patch. */
        std::int32_t patch_y1 = 0;

        /** @brief Holds the column after the last one read from the patch. */
        std::int32_t patch_x2 = 0;

        /** @brief Holds the row after the last one read from the patch. */
        std::int32_t patch_y2 = 0;

        /** @brief Holds canvas_x1 - patch_x1, the amount added to the patch's box x. */
        std::int32_t offset_x = 0;

        /** @brief Holds canvas_y1 - patch_y1, the amount added to the patch's box y. */
        std::int32_t offset_y = 0;
    };

    /** @brief Holds everything apply_mosaic needs. */
    struct MosaicPlan
    {
        /** @brief Holds the training size s; the canvas is 2s by 2s. */
        std::int32_t image_size = 0;

        /** @brief Holds one placement per patch: top left, top right, bottom left, bottom right. */
        std::array<MosaicPlacement, K_MOSAIC_PATCHES> placements{};

        /** @brief Holds the value of every canvas byte no patch covers. */
        std::uint8_t fill_value = 114;
    };

    /** @brief Holds the largest training size whose canvas of twice that side is an image. */
    inline constexpr std::int32_t K_MAX_MOSAIC_IMAGE_SIZE = K_MAX_IMAGE_SIDE / 2;

    /** @brief Returns InvalidDescription unless the probability is in [0, 1]. */
    [[nodiscard]] Status validate_mosaic(const MosaicSettings& settings);

    /** @brief Returns InvalidDescription unless image_size is in [1, K_MAX_MOSAIC_IMAGE_SIZE]. */
    [[nodiscard]] Status validate_mosaic_size(std::int32_t image_size);

    /**
     * @brief Draws the three partners, the gate and the centre of one mosaic.
     *
     * Three draws in partners.choose(index, random), made whatever the gate says; then
     * is_enabled = random.bernoulli(settings.probability). When enabled, center_y and center_x,
     * each random.uniform(b, 2s - b) truncated toward zero, s = image_size, b = (s + 1) / 2.
     * @return The draw, or what validate_mosaic and validate_mosaic_size return.
     */
    [[nodiscard]] Result<MosaicDraw> draw_mosaic(const MosaicSettings& settings,
                                                 std::int32_t image_size, std::size_t index,
                                                 PartnerSource& partners, Random& random);

    /**
     * @brief Places four patches, the sample then three partners, around the drawn centre.
     *
     * Each patch has the corner nearest the centre pinned to it and is cut by the canvas
     * edge, by the reference's rectangles (data/augment.py:525-538). README.md lists them.
     * @return The plan, or InvalidDescription for an image_size or a patch size out of range.
     * @pre @p draw is enabled and its centre lies inside the canvas, as draw_mosaic leaves it.
     */
    [[nodiscard]] Result<MosaicPlan> plan_mosaic(
        const MosaicSettings& settings, const MosaicDraw& draw, std::int32_t image_size,
        const std::array<ImageSize, K_MOSAIC_PATCHES>& patch_sizes);

    /**
     * @brief Composes four patches on a canvas of 2s by 2s.
     *
     * The canvas is filled with plan.fill_value, then each patch's rectangle is copied to
     * its canvas rectangle, patch 0 first. Nothing is resampled; each patch passes check_source.
     * @return InvalidDescription when the patches do not share one channel count.
     * @pre The plan holds a valid image_size; each placement fits its patch and the canvas.
     */
    [[nodiscard]] Status apply_mosaic(const std::array<ImageView, K_MOSAIC_PATCHES>& patches,
                                      const MosaicPlan& plan, Image& destination);

    /**
     * @brief Joins the labels of four patches in canvas coordinates.
     *
     * Each patch's boxes and polygons are moved by its offset_x and offset_y, then joined in
     * patch order. The result is clipped with clip_labels to the canvas and loses its empty
     * boxes through remove_empty_boxes.
     * @param labels Pixel labels of each patch in the coordinates of its own image.
     */
    [[nodiscard]] Labels apply_mosaic(const std::array<Labels, K_MOSAIC_PATCHES>& labels,
                                      const MosaicPlan& plan);
} // namespace SushiAI::Vision::Imaging
