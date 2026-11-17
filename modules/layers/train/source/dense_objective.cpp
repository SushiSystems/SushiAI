/**************************************************************************/
/* dense_objective.cpp                                                    */
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
 * @file dense_objective.cpp
 * @brief Implements what the objectives over one [batch, columns] output share.
 * @author Mustafa Garip
 */

#include "dense_objective.hpp"

#include <SushiAI/failure/check.hpp>
#include <SushiAI/train/dense_losses.hpp>

namespace SushiAI
{
    namespace Train
    {
        namespace Detail
        {
            Status validate_dense_leaf(std::int64_t batch, std::int64_t columns, DType dtype,
                                       std::string_view objective)
            {
                SA_FAIL_IF(batch < 1 || columns < 1, StatusCode::InvalidDescription,
                           "A {} objective needs a batch and a column count of at least 1; "
                           "got {} and {}",
                           objective, batch, columns);
                SA_FAIL_IF(dtype != DType::Float32 && dtype != DType::Float64,
                           StatusCode::InvalidDescription,
                           "A {} objective writes its targets as {} or {}; got {}", objective,
                           dtype_name(DType::Float32), dtype_name(DType::Float64),
                           dtype_name(dtype));

                return Status();
            }

            Graph::ValueID require_matrix_output(const Graph::Graph& graph,
                                                 const NN::ModelOutputs& outputs,
                                                 const Shape& expected,
                                                 std::string_view objective,
                                                 const ObjectiveOutput& selector)
            {
                const Result<Graph::ValueID> chosen = selector.resolve(outputs);
                SA_FATAL_IF(!chosen.has_value(), "The {} objective reads one output: {}",
                            objective, chosen.status().message());

                const Shape& produced = graph.value(chosen.value()).shape;
                SA_FATAL_IF(produced != expected,
                            "The model produced {} but the {} objective was made for {}",
                            produced.to_string(), objective, expected.to_string());

                return chosen.value();
            }

            Result<const OutputShape*> read_matrix_output(const ObjectiveContext& given,
                                                          std::string_view context,
                                                          const ObjectiveOutput& selector)
            {
                SA_TRY_ASSIGN(const OutputShape* const output, selector.resolve(given, context));
                SA_FAIL_IF(output->shape.rank() != K_MATRIX_RANK
                               || output->shape[0] != given.batch,
                           StatusCode::InvalidDescription,
                           "{} reads [{}, columns] values, but the model's output '{}' is {}",
                           context, given.batch, output->name, output->shape.to_string());

                return output;
            }

            Graph::Traced emit_dense_loss(const Graph::GraphBuilder& builder,
                                          const Graph::Graph& graph, const Graph::Traced& terms,
                                          const RowMask& mask,
                                          const ObjectiveLeaves& leaves) noexcept
            {
                if (!mask.is_active())
                    return emit_mean_of_elements(builder, graph, terms);

                SA_TRY_ASSIGN(const Graph::ValueID held, terms);

                // By value: the emissions below can reallocate the value vector.
                const std::int64_t width = graph.value(held).shape[1];

                return mask.emit_weighted(builder, graph, held, leaves,
                                          1.0 / static_cast<double>(width));
            }

        } // namespace Detail
    } // namespace Train
} // namespace SushiAI
