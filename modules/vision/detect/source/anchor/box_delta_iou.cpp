/**************************************************************************/
/* box_delta_iou.cpp                                                      */
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
 * @file box_delta_iou.cpp
 * @brief Implements the complete IoU box delta of an anchor head under both codings.
 * @author Mustafa Garip
 */

#include "box_delta.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/iou_gradient.hpp>

namespace SushiAI::Vision::Detect::Anchor::Detail
{
    namespace
    {
        /** @brief Holds the number of values of a box and of its gradient. */
        constexpr std::size_t K_BOX_ENTRIES = 4;

        /** @brief Holds a box or a gradient as x, y, width and height. */
        using BoxValues = std::array<float, K_BOX_ENTRIES>;

        /** @brief Returns the values of @p box in the order the gradient reads them. */
        BoxValues list_values(const CentreBox& box) noexcept
        {
            return {box.x, box.y, box.width, box.height};
        }

        /** @brief Returns @p value within a positive @p max_delta; 0 when it is not finite. */
        float bound_delta(float value, float max_delta) noexcept
        {
            if (!std::isfinite(value))
                return 0.0f;
            if (max_delta > 0.0f)
                return std::min(std::max(value, -max_delta), max_delta);
            return value;
        }

        /** @brief Returns the gradient of the complete IoU at the predicted box of @p terms. */
        BoxValues measure_gradient(const BoxDeltaTerms& terms) noexcept
        {
            const BoxValues predicted = list_values(terms.predicted);
            const BoxValues truth = list_values(terms.truth);
            BoxValues gradient{};
            complete_iou_gradient(predicted, truth, terms.gradient, gradient);
            return gradient;
        }

        /** @brief Adds @p gradient to @p delta, weighted and bounded as @p terms state. */
        void add_weighted(const BoxValues& gradient, const BoxDeltaTerms& terms,
                          float* delta) noexcept
        {
            for (std::size_t entry = 0; entry < K_BOX_ENTRIES; ++entry)
                delta[entry] += bound_delta(gradient[entry] * terms.box_weight, terms.max_delta);
        }
    } // namespace

    void add_complete_iou_box_delta(const BoxDeltaTerms& terms, float* delta) noexcept
    {
        BoxValues gradient = measure_gradient(terms);

        // A Logit size is anchor * exp(entry), so its slope by the entry is exp(entry).
        gradient[K_WIDTH_ENTRY] *= std::exp(terms.raw_width);
        gradient[K_HEIGHT_ENTRY] *= std::exp(terms.raw_height);

        add_weighted(gradient, terms, delta);
    }

    void add_activated_complete_iou_box_delta(const BoxDeltaTerms& terms, float* delta) noexcept
    {
        // An Activated size entry takes the gradient of its side with no further factor.
        add_weighted(measure_gradient(terms), terms, delta);
    }

} // namespace SushiAI::Vision::Detect::Anchor::Detail
