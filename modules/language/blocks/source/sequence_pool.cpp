/**************************************************************************/
/* sequence_pool.cpp                                                      */
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
 * @file sequence_pool.cpp
 * @brief Implements SequencePool's trace and its shape rule.
 * @author Mustafa Garip
 */

#include <SushiAI/language/blocks/sequence_pool.hpp>

#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/traced.hpp>
#include "block_attributes.hpp"
#include "key_padding.hpp"

namespace SushiAI
{
    namespace Language::Blocks
    {
        namespace
        {
            /** @brief Names the rank of a traced [N, T, dim] sequence value. */
            constexpr int32_t K_POOLED_RANK = 3;

            /** @brief Holds the smallest count a mean divides by, so a length of 0 divides by 1. */
            constexpr double K_SMALLEST_COUNT = 1.0;

            /** @brief Traces the pool, or returns why @p x or @p lengths does not fit. */
            [[nodiscard]] Result<Graph::ValueID> trace_pool(const NN::Trace& trace,
                                                            Graph::ValueID x,
                                                            Graph::ValueID lengths)
            {
                const Graph::GraphBuilder& builder = trace.builder();
                SA_FATAL_IF(x >= builder.graph().value_count(),
                            "SequencePool was traced over value {}; the graph holds {} values", x,
                            builder.graph().value_count());

                // By value: the emissions below grow the value table.
                const Shape sequence = builder.graph().value(x).shape;
                const DType dtype = builder.graph().value(x).dtype;
                SA_FAIL_IF(sequence.rank() != K_POOLED_RANK || !is_real_float(dtype),
                           StatusCode::InvalidDescription,
                           "SequencePool needs a rank-3 [N, T, dim] input of a half, float32 or "
                           "float64 type; got {} of {}",
                           sequence.to_string(), dtype_name(dtype));

                const int64_t batch = sequence[0];
                const int64_t length = sequence[1];
                SA_TRY_ASSIGN(const Graph::ValueID expanded,
                              Detail::expand_lengths(builder, lengths, batch, length, dtype,
                                                     "SequencePool"));
                SA_TRY_ASSIGN(const Graph::ValueID valid,
                              Detail::valid_positions(builder, expanded, batch, length, dtype));

                const Graph::Traced counts = builder.maximum(
                    expanded, builder.constant(Shape{batch, length}, K_SMALLEST_COUNT, dtype));
                const Graph::Traced weights =
                    builder.reshape(builder.div(valid, counts), Shape{batch, 1, length});
                SA_TRY_ASSIGN(const Graph::ValueID mean, builder.matmul(weights, x));
                SA_TRY_ASSIGN(const Graph::ValueID rows,
                              builder.reshape(mean, Shape{batch, sequence[2]}));

                return rows;
            }
        } // namespace

        Graph::ValueID SequencePool::forward_with_lengths(const NN::Trace& trace, Graph::ValueID x,
                                                          Graph::ValueID lengths) noexcept
        {
            return trace.value_or_fail(trace_pool(trace, x, lengths), x);
        }

        Result<Shape> SequencePool::output_shape(const Shape& input) const noexcept
        {
            SA_FAIL_IF(input.rank() != Detail::K_SAMPLE_RANK, StatusCode::InvalidDescription,
                       "SequencePool reads a per-sample [T, dim] input; got {}",
                       input.to_string());

            return Shape{input[1]};
        }

    } // namespace Language::Blocks
} // namespace SushiAI
