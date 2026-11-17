/**************************************************************************/
/* key_padding.hpp                                                        */
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
 * @file key_padding.hpp
 * @brief Defines the emissions that turn a [N] lengths value into the positions it keeps.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/traced.hpp>
#include <SushiAI/language/blocks/incremental.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        namespace Detail
        {
            /** @brief Holds the score a padded key takes in half precision: the lowest finite one. */
            inline constexpr double K_HALF_LOWEST_SCORE = -65504.0;

            /** @brief Holds the largest position count a half value counts without rounding. */
            inline constexpr int64_t K_HALF_EXACT_POSITIONS = 2048;

            /**
             * @brief Returns the score a key past its row's length takes before the softmax.
             * @return The lowest finite half for half precision, K_MASKED_ATTENTION_SCORE
             *         otherwise, so a row of padded keys alone still softmaxes to a number.
             */
            [[nodiscard]] inline double padded_score(DType dtype) noexcept
            {
                return dtype == DType::Half ? K_HALF_LOWEST_SCORE : K_MASKED_ATTENTION_SCORE;
            }

            /**
             * @brief Returns the lengths of @p lengths repeated along @p length positions.
             * @param lengths A value of builder.graph() holding @p batch real floating counts.
             * @param dtype The element type of the sequence the lengths mask.
             * @return The [batch, length] value whose row n holds lengths[n] length times, or
             *         InvalidDescription when the lengths do not fit.
             */
            [[nodiscard]] inline Result<Graph::ValueID> expand_lengths(
                const Graph::GraphBuilder& builder, Graph::ValueID lengths, int64_t batch,
                int64_t length, DType dtype, std::string_view block)
            {
                SA_FATAL_IF(lengths >= builder.graph().value_count(),
                            "{} was given lengths value {}; the graph holds {} values", block,
                            lengths, builder.graph().value_count());

                const Shape given = builder.graph().value(lengths).shape;
                const DType given_dtype = builder.graph().value(lengths).dtype;
                SA_FAIL_IF(given.num_elements() != batch || !is_real_float(given_dtype),
                           StatusCode::InvalidDescription,
                           "{} needs one real floating length for each of its {} sequences; got "
                           "{} of {}",
                           block, batch, given.to_string(), dtype_name(given_dtype));
                SA_FAIL_IF(dtype == DType::Half && length > K_HALF_EXACT_POSITIONS,
                           StatusCode::InvalidDescription,
                           "{} masks {} positions in half precision, which counts exactly only "
                           "up to {}",
                           block, length, K_HALF_EXACT_POSITIONS);

                SA_TRY_ASSIGN(const Graph::ValueID column,
                              builder.reshape(lengths, Shape{batch, 1}));
                Graph::ValueID typed = column;
                if (given_dtype != dtype)
                {
                    SA_TRY_ASSIGN(typed, builder.cast(column, dtype));
                }

                const Graph::Traced ones = builder.constant(Shape{1, length}, 1.0, dtype);

                // An outer product is the broadcast the graph has no elementwise form of.
                SA_TRY_ASSIGN(const Graph::ValueID expanded, builder.matmul(typed, ones));

                return expanded;
            }

            /**
             * @brief Returns 1 at each position a sequence of its row's length holds.
             * @param expanded What expand_lengths() returned, [batch, length].
             * @return The [batch, length] value of 0 and 1: position t of row n is 1 when
             *         t < lengths[n].
             */
            [[nodiscard]] inline Result<Graph::ValueID> valid_positions(
                const Graph::GraphBuilder& builder, Graph::ValueID expanded, int64_t batch,
                int64_t length, DType dtype)
            {
                // The lower triangle of ones has i + 1 ones in row i, which counts the positions.
                const Graph::Traced triangle =
                    builder.causal_mask(builder.constant(Shape{length, length}, 1.0, dtype), 0.0);
                const Graph::Traced counted =
                    builder.matmul(triangle, builder.constant(Shape{length, 1}, 1.0, dtype));
                const Graph::Traced counts = builder.reshape(counted, Shape{1, length});
                const Graph::Traced numbered =
                    builder.matmul(builder.constant(Shape{batch, 1}, 1.0, dtype), counts);

                // A position t is held when t + 1 <= length.
                SA_TRY_ASSIGN(const Graph::ValueID held, builder.less_equal(numbered, expanded));

                return held;
            }

            /**
             * @brief Returns the [batch, length, length] condition that keeps each query's view
             *        of the keys inside the length of its row.
             * @param valid What valid_positions() returned.
             * @return Entry (n, i, j) is 1 when key j of sequence n is held, whatever i is.
             */
            [[nodiscard]] inline Result<Graph::ValueID> key_condition(
                const Graph::GraphBuilder& builder, Graph::ValueID valid, int64_t batch,
                int64_t length, DType dtype)
            {
                const Graph::Traced keys = builder.reshape(valid, Shape{batch, 1, length});
                const Graph::Traced ones = builder.constant(Shape{batch, length, 1}, 1.0, dtype);
                SA_TRY_ASSIGN(const Graph::ValueID condition, builder.matmul(ones, keys));

                return condition;
            }

        } // namespace Detail
    } // namespace Language::Blocks
} // namespace SushiAI
