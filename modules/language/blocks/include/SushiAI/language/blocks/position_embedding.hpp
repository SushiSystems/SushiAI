/**************************************************************************/
/* position_embedding.hpp                                                 */
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
 * @file position_embedding.hpp
 * @brief Declares PositionEmbedding, the learned table added to each position of a sequence.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <tuple>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/embedding.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        /** @brief Adds row t of a learned [length, dim] table to position t of [N, T, dim]. */
        struct PositionEmbedding
        {
            /** @brief Holds the rows of the table, which is the longest sequence it serves. */
            int64_t length = 0;

            /** @brief Holds the width of a row, which is the width of the input. */
            int64_t dim = 0;

            /** @brief Holds the [length, dim] table, of role Embedding. */
            NN::Parameter weight{};

            /** @brief Holds the fill of the table: a normal of NN::K_EMBEDDING_DEVIATION. */
            NN::ParameterInit weight_init = NN::ParameterInit::normal(NN::K_EMBEDDING_DEVIATION);

            /** @brief Builds a table of no row, which no input fits. */
            PositionEmbedding() = default;

            /** @brief Describes the table; forward() checks the extents. */
            PositionEmbedding(int64_t length, int64_t dim) noexcept;

            /** @brief Lists the one parameter, under "weight". */
            static constexpr auto fields()
            {
                return std::tuple{
                    NN::field("weight", &PositionEmbedding::weight),
                };
            }

            /**
             * @brief Traces x plus the first T rows of the table, over [N, T * dim] rows.
             * @param x A [N, T, dim] value of a real floating type with T <= length.
             * @return The [N, T, dim] sum's id, or @p x after trace.fail() when @p x is longer
             *         than the table or does not fit this block's description.
             */
            Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID x);

            /**
             * @brief Traces x plus the row of the table the leaf K_POSITION_LEAF names.
             * @param x The [1, 1, dim] slice of one sequence, of a real floating type.
             * @param sample The per-sample [T, dim] shape of the ordinary form; T <= length.
             * @return The [1, 1, dim] sum's id, or @p x after trace.fail() when an argument or
             *         the trace does not fit.
             */
            Graph::ValueID forward_incremental(const NN::Trace& trace, Graph::ValueID x,
                                               const Shape& sample);

            /** @brief Returns the per-sample shape forward() produces, which is @p input. */
            Result<Shape> output_shape(const Shape& input) const;
        };

        static_assert(NN::TraceModule<PositionEmbedding>,
                      "PositionEmbedding must satisfy the TraceModule contract");
        static_assert(NN::IncrementalModule<PositionEmbedding>,
                      "PositionEmbedding must have an incremental form");

        /**
         * @brief Builds a PositionEmbedding node from an optional "length", T when absent.
         * @param inputs One per-sample [T, C] shape; C becomes dim.
         * @return The node, or InvalidDescription when @p inputs or a key does not fit, or
         *         "length" is below T.
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_position_embedding_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context);

    } // namespace Language::Blocks
} // namespace SushiAI
