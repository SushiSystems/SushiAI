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
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/objective_output.hpp>
#include <SushiAI/train/row_mask.hpp>

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
             * @brief Returns the output @p selector names, or the only output of @p outputs.
             * @pre @p outputs holds that output, of the shape @p expected; anything else stops
             *      the process, naming @p objective.
             */
            [[nodiscard]] Graph::ValueID require_matrix_output(
                const Graph::Graph& graph, const NN::ModelOutputs& outputs, const Shape& expected,
                std::string_view objective, const ObjectiveOutput& selector = ObjectiveOutput());

            /**
             * @brief Returns the [batch, columns] output of @p given that @p selector names,
             *        or its only output.
             * @param context What to call the objective in a diagnostic.
             * @return A pointer into @p given, never null, or InvalidDescription for a model
             *         that publishes no such output, or one of another rank or another batch.
             */
            [[nodiscard]] Result<const OutputShape*> read_matrix_output(
                const ObjectiveContext& given, std::string_view context,
                const ObjectiveOutput& selector = ObjectiveOutput());

            /**
             * @brief Emits the loss of a dense objective from its per-element @p terms.
             * @param terms The [N, width] terms, e.g. emit_logistic_terms().
             * @param mask The rows the loss keeps; the leaves of an off mask are not read.
             * @return The mean of every term for an off mask; the mean, over the kept rows, of
             *         the mean of each row for an on one.
             */
            [[nodiscard]] Graph::Traced emit_dense_loss(const Graph::GraphBuilder& builder,
                                                        const Graph::Graph& graph,
                                                        const Graph::Traced& terms,
                                                        const RowMask& mask,
                                                        const ObjectiveLeaves& leaves) noexcept;

        } // namespace Detail
    } // namespace Train
} // namespace SushiAI
