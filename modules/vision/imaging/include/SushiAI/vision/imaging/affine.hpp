/**************************************************************************/
/* affine.hpp                                                             */
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
 * @file affine.hpp
 * @brief Declares the random scale and translation that produce the training-size image.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstdint>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/label_set.hpp>
#include <SushiAI/vision/imaging/random.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Vision::Imaging
{
    /** @brief Holds what the caller chooses about the random affine warp. */
    struct AffineSettings
    {
        /** @brief Holds the scale half-range; the scale is drawn from [1 - scale, 1 + scale). */
        float scale = 0.5f;

        /** @brief Holds the translation half-range as a fraction of the output side. */
        float translate = 0.1f;

        /** @brief Holds the value of every output byte the source does not cover. */
        std::uint8_t border_value = 114;

        /** @brief Holds the thresholds a warped box must pass when its object has no polygon. */
        CandidateFilter filter{};

        /** @brief Holds the thresholds a box rebuilt from a warped polygon must pass. */
        CandidateFilter polygon_filter = make_polygon_filter();
    };

    /** @brief Holds everything apply_affine needs: the sizes, the draws and the matrix. */
    struct AffinePlan
    {
        /** @brief Holds the size of the image the plan was made for. */
        ImageSize source{};

        /** @brief Holds the size of the result. */
        ImageSize output{};

        /** @brief Holds the drawn scale, above zero. */
        float scale = 1.0f;

        /** @brief Holds the drawn output x, in pixels, that the source centre maps to. */
        float translate_x = 0.0f;

        /** @brief Holds the drawn output y, in pixels, that the source centre maps to. */
        float translate_y = 0.0f;

        /** @brief Holds the forward matrix, row-major 2 by 3: x' = m[0] x + m[1] y + m[2]. */
        std::array<float, 6> matrix{};

        /** @brief Holds the value of every output byte the source does not cover. */
        std::uint8_t border_value = 114;

        /** @brief Holds the thresholds a warped box must pass when its object has no polygon. */
        CandidateFilter filter{};

        /** @brief Holds the thresholds a box rebuilt from a warped polygon must pass. */
        CandidateFilter polygon_filter = make_polygon_filter();
    };

    /** @brief Returns InvalidDescription unless scale is in [0, 1) and translate is in [0, 0.5]. */
    [[nodiscard]] Status validate_affine(const AffineSettings& settings) noexcept;

    /**
     * @brief Draws one scale and one translation and builds the forward matrix.
     *
     * Three draws in order: scale from [1 - s, 1 + s), then translate_x and translate_y, each
     * from [0.5 - t, 0.5 + t) times the output width or height. In float, m[0] = m[4] = scale,
     * m[2] = scale * (-source.width / 2.0f) + translate_x, and m[5] likewise with height and y.
     * @return The plan, or InvalidDescription for what validate_affine or is_valid_size rejects.
     */
    [[nodiscard]] Result<AffinePlan> plan_affine(const AffineSettings& settings,
                                                 ImageSize source, ImageSize output,
                                                 Random& random) noexcept;

    /**
     * @brief Warps an image into plan.output by bilinear interpolation.
     *
     * Output pixel (x, y) reads source position ((x - m[2]) / m[0], (y - m[5]) / m[4]) in
     * double; a neighbour outside the source counts as plan.border_value.
     * @pre source.size() is plan.source, plan.output is a valid size, the matrix is the plan's.
     * @param destination Reset to plan.output; must pass check_source.
     */
    void apply_affine(ImageView source, const AffinePlan& plan, Image& destination) noexcept;

    /**
     * @brief Warps boxes and polygons into the output and drops the objects a filter rejects.
     *
     * In float, x' = m[0] * x + m[2] and y' = m[4] * y + m[5]. Without a polygon, an object
     * has its two corners warped and clipped and is tested with plan.filter; with one, every
     * vertex is warped, visible_polygon_box is its box and plan.polygon_filter its test. Both
     * compare the source box scaled by plan.scale with the new box. Survivors keep their order.
     */
    [[nodiscard]] Labels apply_affine(const Labels& labels, const AffinePlan& plan) noexcept;
} // namespace SushiAI::Vision::Imaging
