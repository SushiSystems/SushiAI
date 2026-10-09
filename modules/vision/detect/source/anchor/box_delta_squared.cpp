/**************************************************************************/
/* box_delta_squared.cpp                                                  */
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
 * @file box_delta_squared.cpp
 * @brief Implements the squared-error box delta of an anchor head.
 * @author Mustafa Garip
 */

#include "box_delta.hpp"

#include <cmath>

#include <SushiAI/vision/detect/anchor/head_options.hpp>

namespace SushiAI::Vision::Detect::Anchor::Detail
{
    namespace
    {
        /** @brief Holds the weight of a truth whose area is zero; the area is taken off it. */
        constexpr float K_SMALL_BOX_WEIGHT = 2.0f;
    } // namespace

    void add_squared_error_box_delta(const BoxDeltaTerms& terms, float* delta) noexcept
    {
        const float weight =
            (K_SMALL_BOX_WEIGHT - terms.truth.width * terms.truth.height) * terms.box_weight;
        const float target_width = std::log(terms.truth.width / terms.anchor.width);
        const float target_height = std::log(terms.truth.height / terms.anchor.height);

        delta[K_X_ENTRY] += weight * (terms.target_x - terms.offset_x);
        delta[K_Y_ENTRY] += weight * (terms.target_y - terms.offset_y);
        delta[K_WIDTH_ENTRY] += weight * (target_width - terms.raw_width);
        delta[K_HEIGHT_ENTRY] += weight * (target_height - terms.raw_height);
    }

} // namespace SushiAI::Vision::Detect::Anchor::Detail
