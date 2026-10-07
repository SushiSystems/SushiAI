/**************************************************************************/
/* flip.hpp                                                               */
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
 * @file flip.hpp
 * @brief Declares the random horizontal flip of an image and its labels.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/imaging/image.hpp>
#include <SushiAI/imaging/label_set.hpp>
#include <SushiAI/imaging/random.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Imaging
{
    /** @brief Holds what the caller chooses about the horizontal flip. */
    struct FlipSettings
    {
        /** @brief Holds the probability of a flip, in [0, 1] (cfg/default.yaml:124). */
        float probability = 0.5f;
    };

    /** @brief Holds everything apply_flip needs: the drawn decision and the image size. */
    struct FlipPlan
    {
        /** @brief Holds the size of the image the plan was made for. */
        ImageSize source{};

        /** @brief Tells whether the image is mirrored. */
        bool is_flipped = false;
    };

    /** @brief Throws Error unless the probability is in [0, 1]. */
    void validate_flip(const FlipSettings& settings);

    /**
     * @brief Draws whether to flip an image of the given size.
     *
     * One draw: is_flipped = random.bernoulli(settings.probability). Throws Error, before
     * any draw, on what validate_flip rejects and on a size that fails is_valid_size.
     */
    [[nodiscard]] FlipPlan plan_flip(const FlipSettings& settings, ImageSize source,
                                     Random& random);

    /**
     * @brief Copies an image, mirrored left to right when the plan says so.
     *
     * A flipped destination column x holds source column width - 1 - x, channels in the
     * same order. Without a flip the pixels are copied unchanged. Throws Error unless
     * source.size() is plan.source.
     * @param destination Reset to the source's size and channels; must pass check_source.
     */
    void apply_flip(ImageView source, const FlipPlan& plan, Image& destination);

    /**
     * @brief Mirrors boxes and polygons left to right when the plan says so.
     *
     * With w = plan.source.width, a flipped box has x1 = w - (its x2) and x2 = w - (its x1)
     * and a flipped vertex has x = w - x, in float; every y is unchanged. Classes and order
     * are kept. Without a flip the labels are returned unchanged.
     */
    [[nodiscard]] Labels apply_flip(const Labels& labels, const FlipPlan& plan);
} // namespace SushiAI::Imaging
