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
 * @file dense_objective.cpp
 * @brief Implements what the objectives over one [batch, columns] output share.
 * @author Mustafa Garip
 */

#include "dense_objective.hpp"

#include <SushiAI/core/check.hpp>

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
                                                 std::string_view objective)
            {
                const Result<Graph::ValueID> single = outputs.only();
                SA_FATAL_IF(!single.has_value(), "The {} objective reads one output: {}",
                            objective, single.status().message());

                const Shape& produced = graph.value(single.value()).shape;
                SA_FATAL_IF(produced != expected,
                            "The model produced {} but the {} objective was made for {}",
                            produced.to_string(), objective, expected.to_string());

                return single.value();
            }

            Result<Shape> read_matrix_output(const ObjectiveContext& given,
                                             std::string_view context)
            {
                SA_FAIL_IF(given.outputs.size() != 1, StatusCode::InvalidDescription,
                           "{} reads one [batch, columns] output, but the model publishes {}",
                           context, given.outputs.size());

                const OutputShape& output = given.outputs.front();
                SA_FAIL_IF(output.shape.rank() != K_MATRIX_RANK || output.shape[0] != given.batch,
                           StatusCode::InvalidDescription,
                           "{} reads [{}, columns] values, but the model's output '{}' is {}",
                           context, given.batch, output.name, output.shape.to_string());

                return output.shape;
            }

        } // namespace Detail
    } // namespace Train
} // namespace SushiAI
