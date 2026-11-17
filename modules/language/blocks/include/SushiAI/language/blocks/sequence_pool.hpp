/**************************************************************************/
/* sequence_pool.hpp                                                      */
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
 * @file sequence_pool.hpp
 * @brief Declares SequencePool, the mean of a sequence over the positions its length holds.
 * @author Mustafa Garip
 */

#pragma once

#include <tuple>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        /** @brief Averages each [T, dim] sequence of a [N, T, dim] value over its first positions. */
        struct SequencePool
        {
            /** @brief Lists no field: the block holds no state. */
            static constexpr auto fields() noexcept
            {
                return std::tuple<>{};
            }

            /**
             * @brief Traces row n of the result as the mean of the first lengths[n] positions.
             * @param x A [N, T, dim] value of a real floating type.
             * @param lengths N real floating counts, each at most T, as a [N] value or any shape.
             * @return The [N, dim] id, or @p x after trace.fail() when an argument misfits.
             */
            Graph::ValueID forward_with_lengths(const NN::Trace& trace, Graph::ValueID x,
                                                Graph::ValueID lengths) noexcept;

            /** @brief Returns the per-sample [dim] shape forward_with_lengths() produces. */
            Result<Shape> output_shape(const Shape& input) const noexcept;
        };

    } // namespace Language::Blocks
} // namespace SushiAI
