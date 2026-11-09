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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file block_attributes.hpp
 * @brief Defines the defaults the block rows share and names the attribute readers they use.
 * @author Mustafa Garip
 */

#pragma once

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

            /** @brief Holds the smallest channel count of an attention head; its key is half. */
            inline constexpr int64_t K_SMALLEST_HEAD_WIDTH = 2;

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
