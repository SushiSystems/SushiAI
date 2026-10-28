/**************************************************************************/
/* head_decode.hpp                                                        */
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
 * @file head_decode.hpp
 * @brief Declares the decode of an anchor head's output into boxes and class scores.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        /**
         * @brief Decodes a head's `[B, H, W, A * (5 + C)]` output into boxes and class scores.
         * @param boxes Receives `[B, H * W * A, 4]`: centre x, centre y, width, height, as
         *              fractions of the network input. Row `(row * W + col) * A + a`.
         * @param scores Receives `[B, H * W * A, C]`: objectness times class, same rows.
         * @pre validate_head succeeds, every extent of @p shape is positive and the three
         *      spans hold exactly those values; anything else is fatal.
         */
        void decode_head(SushiRuntime::span<const float> head, const HeadShape& shape,
                         const HeadOptions& options, const AnchorLayout& layout,
                         std::int64_t head_index, SushiRuntime::span<float> boxes,
                         SushiRuntime::span<float> scores) noexcept;

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
