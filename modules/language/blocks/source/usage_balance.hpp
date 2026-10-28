/**************************************************************************/
/* usage_balance.hpp                                                      */
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
 * @file usage_balance.hpp
 * @brief Defines the emission of each expert's share of a batch and of the spread of the shares.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        namespace Detail
        {
            /** @brief Holds the share of each expert and the spread of the shares. */
            struct UsageBalance
            {
                Graph::ValueID usage = Graph::K_NO_VALUE;   /**< The [1, E] column means. */
                Graph::ValueID balance = Graph::K_NO_VALUE; /**< The [1, 1] spread. */
            };

            /**
             * @brief Emits the mean of each column of @p probabilities, and the squared
             *        coefficient of variation of those means, E * sum(mean^2) - 1.
             * @param probabilities A [R, E] value whose rows each sum to one.
             * @return The two ids, the spread 0 for equal shares and E - 1 for one expert, or
             *         the refusal of an emitter.
             */
            [[nodiscard]] inline Result<UsageBalance> trace_usage_balance(
                const Graph::GraphBuilder& builder, Graph::ValueID probabilities)
            {
                // By value: every emission below can reallocate the value vector.
                const Shape shape = builder.graph().value(probabilities).shape;
                const int64_t experts = shape[1];

                const Graph::Traced means = builder.scale(
                    builder.sum_rows(probabilities), 1.0 / static_cast<double>(shape[0]));
                SA_TRY_ASSIGN(const Graph::ValueID usage,
                              builder.reshape(means, Shape{1, experts}));

                const Graph::Traced squares = builder.scale(
                    builder.matmul(usage, usage, false, true), static_cast<double>(experts));
                SA_TRY_ASSIGN(const Graph::ValueID balance,
                              builder.sub(squares, builder.constant_like(squares, 1.0)));

                return UsageBalance{usage, balance};
            }

        } // namespace Detail
    } // namespace Language::Blocks
} // namespace SushiAI
