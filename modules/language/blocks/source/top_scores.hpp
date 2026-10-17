/**************************************************************************/
/* top_scores.hpp                                                         */
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
 * @file top_scores.hpp
 * @brief Defines the emission that keeps the largest scores of each row and drops the rest.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        namespace Detail
        {
            /** @brief Holds the score a dropped expert takes, the one attention masks with. */
            inline constexpr double K_DROPPED_EXPERT_SCORE = -1.0e9;

            /**
             * @brief Emits @p scores with every entry below its row's @p top_k th largest
             *        distinct value replaced by K_DROPPED_EXPERT_SCORE.
             * @param scores A [R, E] value of a real floating type.
             * @param top_k In [1, E]; equal scores are kept together, so a tie keeps more.
             * @return The [R, E] result's id; the kept entries pass their gradient.
             * @throws Error What an emitter throws for an operand that does not fit.
             */
            [[nodiscard]] inline Graph::ValueID keep_top_scores(
                const Graph::GraphBuilder& builder, Graph::ValueID scores, int64_t top_k)
            {
                // By value: every emission below can reallocate the value vector.
                const Shape shape = builder.graph().value(scores).shape;
                const int64_t experts = shape[1];

                // A row is one image row of E pixels, so one window covers a whole row.
                Graph::ConvGeometry row_window;
                row_window.kernel_w = experts;
                row_window.stride_w = experts;

                const Graph::ValueID map =
                    builder.reshape(scores, Shape{shape[0], 1, experts, 1});
                const Graph::ValueID dropped =
                    builder.constant_like(map, K_DROPPED_EXPERT_SCORE);

                Graph::ValueID remaining = map;
                Graph::ValueID kept = Graph::K_NO_VALUE;
                for (int64_t level = 0; level < top_k; ++level)
                {
                    if (level != 0)
                        remaining = builder.select(kept, dropped, map);

                    const Graph::ValueID largest =
                        builder.maxpool2d(remaining, row_window).output;
                    const Graph::ValueID threshold = builder.upsample2d(largest, 1, experts);
                    kept = builder.less_equal(threshold, map);
                }

                return builder.reshape(builder.select(kept, map, dropped), shape);
            }

        } // namespace Detail
    } // namespace Language::Blocks
} // namespace SushiAI
