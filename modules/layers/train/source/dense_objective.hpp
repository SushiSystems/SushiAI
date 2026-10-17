/**************************************************************************/
/* dense_objective.hpp                                                    */
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
 * @file dense_objective.hpp
 * @brief Declares what the objectives over one [batch, columns] output share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/train/objective_context.hpp>

namespace SushiAI
{
    namespace Train
    {
        namespace Detail
        {
            /** @brief Names the leaf a dense objective declares and fills. */
            inline constexpr std::string_view K_TARGETS_LEAF = "targets";

            /** @brief Names the rank of the [batch, columns] output a dense objective reads. */
            inline constexpr int32_t K_MATRIX_RANK = 2;

            /**
             * @brief Returns InvalidDescription unless a [batch, columns] target leaf of
             *        @p dtype can be declared.
             * @param objective Names the objective in the message.
             */
            [[nodiscard]] Status validate_dense_leaf(std::int64_t batch, std::int64_t columns,
                                                     DType dtype, std::string_view objective);

            /**
             * @brief Returns the only output of @p outputs.
             * @pre @p outputs holds one value of the shape @p expected; anything else stops
             *      the process, naming @p objective.
             */
            [[nodiscard]] Graph::ValueID require_matrix_output(const Graph::Graph& graph,
                                                               const NN::ModelOutputs& outputs,
                                                               const Shape& expected,
                                                               std::string_view objective);

            /**
             * @brief Returns the extents of the one [batch, columns] output of @p given.
             * @param context What to call the objective in a diagnostic.
             * @return InvalidDescription for a model that publishes another count of
             *         outputs, or one of another rank or another batch.
             */
            [[nodiscard]] Result<Shape> read_matrix_output(const ObjectiveContext& given,
                                                           std::string_view context);

        } // namespace Detail
    } // namespace Train
} // namespace SushiAI
