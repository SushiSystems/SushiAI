/**************************************************************************/
/* block_attributes.hpp                                                   */
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
 * @file block_attributes.hpp
 * @brief Defines the defaults the block rows share and names the attribute readers they use.
 * @author Mustafa Garip
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <SushiAI/nn/node_attributes.hpp>

namespace SushiAI
{
    namespace Vision::Blocks
    {
        namespace Detail
        {
            /** @brief Holds the kernel and the stride of a 1x1 convolution. */
            inline constexpr int64_t K_POINTWISE = 1;

            /** @brief Holds the inner-block count of a row that states none. */
            inline constexpr int64_t K_DEFAULT_REPEATS = 1;

            /** @brief Holds whether a bottleneck adds its input in a row that does not say. */
            inline constexpr bool K_DEFAULT_SHORTCUT = true;

            /** @brief Holds the channels per attention head of a row that states no head count. */
            inline constexpr int64_t K_CHANNELS_PER_DEFAULT_HEAD = 64;

            /** @brief Holds the smallest channel count of an attention head; its key is half. */
            inline constexpr int64_t K_SMALLEST_HEAD_WIDTH = 2;

            /** @brief Returns max(hidden / K_CHANNELS_PER_DEFAULT_HEAD, 1). */
            [[nodiscard]] constexpr int64_t default_head_count(int64_t hidden) noexcept
            {
                return std::max<int64_t>(hidden / K_CHANNELS_PER_DEFAULT_HEAD, 1);
            }

            using NN::Detail::K_NORM_EPSILON_KEY;
            using NN::Detail::K_NORM_MOMENTUM_KEY;
            using NN::Detail::KeySpan;
            using NN::Detail::list_names;
            using NN::Detail::read_feature_map;
            using NN::Detail::reject_unknown_keys;
            using NN::Detail::read_positive_int;
            using NN::Detail::read_positive_int_or;
            using NN::Detail::read_bool_or;
            using NN::Detail::read_norm_options;
            using NN::Detail::wrap_block;
        } // namespace Detail
    } // namespace Vision::Blocks
} // namespace SushiAI
