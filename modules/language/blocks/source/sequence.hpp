/**************************************************************************/
/* sequence.hpp                                                           */
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
 * @file sequence.hpp
 * @brief Defines the checks and the row view the blocks share over a [N, T, C] sequence value.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <utility>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/throwing_bridge.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        namespace Detail
        {
            /** @brief Names the rank of a traced [N, T, C] sequence value. */
            inline constexpr int32_t K_SEQUENCE_RANK = 3;

            /**
             * @brief Returns the shape of @p x once it is known to be a [N, T, dim] sequence.
             * @param x A value of builder.graph(); anything else is fatal.
             * @param block What to call the block in a diagnostic.
             * @return A copy of the shape, or InvalidDescription when the rank, the last
             *         extent or the element type does not fit.
             */
            [[nodiscard]] inline Result<Shape> checked_sequence(
                const Graph::GraphBuilder& builder, Graph::ValueID x, int64_t dim,
                std::string_view block)
            {
                SA_FATAL_IF(x >= builder.graph().value_count(),
                            "{} was traced over value {}; the graph holds {} values", block, x,
                            builder.graph().value_count());

                const Shape shape = builder.graph().value(x).shape;
                const DType dtype = builder.graph().value(x).dtype;
                SA_FAIL_IF(shape.rank() != K_SEQUENCE_RANK || shape[2] != dim,
                           StatusCode::InvalidDescription,
                           "{} needs a rank-3 [N, T, {}] input; got {}", block, dim,
                           shape.to_string());
                SA_FAIL_IF(!is_real_float(dtype), StatusCode::InvalidDescription,
                           "{} needs a half, float32 or float64 input; got {}", block,
                           dtype_name(dtype));

                return shape;
            }

            /**
             * @brief Traces @p trace_rows over the [N * T, C] rows of @p x and returns its
             *        result seen as [N, T, out_features].
             * @param sequence The [N, T, C] shape of @p x.
             * @param trace_rows Maps the id of the rows to the id of [N * T, out_features].
             * @return The result's id, or InvalidDescription carrying what an emitter threw.
             */
            template<typename TraceRows>
            [[nodiscard]] Result<Graph::ValueID> over_rows(const Graph::GraphBuilder& builder,
                                                           Graph::ValueID x,
                                                           const Shape& sequence,
                                                           int64_t out_features,
                                                           TraceRows&& trace_rows)
            {
                return NN::Detail::bridge_throwing([&]()
                {
                    const Graph::ValueID rows = builder.reshape(
                        x, Shape{sequence[0] * sequence[1], sequence[2]});
                    const Graph::ValueID mapped = std::forward<TraceRows>(trace_rows)(rows);

                    return builder.reshape(mapped,
                                           Shape{sequence[0], sequence[1], out_features});
                });
            }

        } // namespace Detail
    } // namespace Language::Blocks
} // namespace SushiAI
