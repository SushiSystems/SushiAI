/**************************************************************************/
/* mean_squared_error_objective.cpp                                       */
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
 * @file mean_squared_error_objective.cpp
 * @brief Implements the regression objective and the factory a training file reaches it by.
 * @author Mustafa Garip
 */

#include <SushiAI/train/mean_squared_error_objective.hpp>

#include <cmath>
#include <string>
#include <utility>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/train/batch_role.hpp>
#include <SushiAI/train/dense_losses.hpp>
#include <SushiAI/train/dense_targets.hpp>
#include <SushiAI/train/host_values.hpp>

#include "dense_objective.hpp"

namespace SushiAI
{
    namespace Train
    {
        Result<std::unique_ptr<MeanSquaredErrorObjective>> MeanSquaredErrorObjective::make(
            std::int64_t batch, std::int64_t width, DType dtype)
        {
            SA_TRY(Detail::validate_dense_leaf(batch, width, dtype, K_MEAN_SQUARED_ERROR_TYPE));

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                return std::unique_ptr<MeanSquaredErrorObjective>(
                    new MeanSquaredErrorObjective(batch, width, dtype));
            });
        }

        MeanSquaredErrorObjective::MeanSquaredErrorObjective(std::int64_t batch,
                                                             std::int64_t width,
                                                             DType dtype) noexcept
            : batch_(batch), width_(width), dtype_(dtype)
        {

        }

        void MeanSquaredErrorObjective::declare(const Graph::GraphBuilder& builder,
                                                ObjectiveLeaves& leaves) const
        {
            leaves.add(Detail::K_TARGETS_LEAF,
                       builder.input(Shape{batch_, width_}, Detail::K_TARGETS_LEAF, dtype_));
            leaves.require(Data::TargetDescription{std::string(Data::K_ROLE_VALUES),
                                                   Shape{width_}, DType::Float32, 0});
        }

        Result<Graph::ValueID> MeanSquaredErrorObjective::loss(
            const Graph::GraphBuilder& builder, const Graph::Graph& graph,
            const NN::ModelOutputs& outputs, const ObjectiveLeaves& leaves) const
        {
            const Graph::ValueID predicted = Detail::require_matrix_output(
                graph, outputs, Shape{batch_, width_}, K_MEAN_SQUARED_ERROR_TYPE);

            return emit_squared_error(builder, graph, predicted,
                                      leaves.at(Detail::K_TARGETS_LEAF));
        }

        Status MeanSquaredErrorObjective::feed(const Data::Batch& batch,
                                               const ObjectiveLeaves& leaves,
                                               Graph::TensorPool& pool) const
        {
            SA_TRY_ASSIGN(const Data::BatchTarget* const values,
                          require_batch_role(batch, Data::K_ROLE_VALUES));
            SA_TRY_ASSIGN(SushiBLAS::Tensor* const targets,
                          find_leaf_tensor(leaves, Detail::K_TARGETS_LEAF, pool));

            return fill_dense_targets(values->values, *targets);
        }

        Status MeanSquaredErrorObjective::measure(const StepView& step, Metrics& metrics) const
        {
            const std::int64_t elements = batch_ * width_;

            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const predicted, only_output(step));
            SA_TRY(require_host_reals(*predicted, elements, "the outputs"));

            SA_TRY_ASSIGN(const Data::BatchTarget* const values,
                          require_batch_role(step.batch, Data::K_ROLE_VALUES));
            SA_TRY(require_host_reals(values->values, elements, "the values"));

            double total = 0.0;
            for (std::int64_t index = 0; index < elements; ++index)
                total += std::abs(host_real(*predicted, index) - host_real(values->values, index));

            return metrics.add(
                Metric{"mean_absolute_error", total / static_cast<double>(elements), false});
        }

        Result<std::unique_ptr<Objective>> make_mean_squared_error(
            const ObjectiveContext& given, const Config::Value& attributes,
            std::string_view context)
        {
            SA_FAIL_IF(!attributes.keys().empty(), StatusCode::InvalidDescription,
                       "{} states the key '{}', but a mean_squared_error objective has no keys",
                       context, attributes.keys().front());
            SA_TRY_ASSIGN(const Shape outputs, Detail::read_matrix_output(given, context));

            SA_TRY_ASSIGN(std::unique_ptr<MeanSquaredErrorObjective> made,
                          MeanSquaredErrorObjective::make(outputs[0], outputs[1]));
            std::unique_ptr<Objective> base = std::move(made);
            return std::move(base);
        }

    } // namespace Train
} // namespace SushiAI
