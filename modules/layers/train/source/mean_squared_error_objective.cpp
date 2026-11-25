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
 * @file mean_squared_error_objective.cpp
 * @brief Implements the regression objective and the factory a training file reaches it by.
 * @author Mustafa Garip
 */

#include <SushiAI/train/mean_squared_error_objective.hpp>

#include <cmath>
#include <string>
#include <utility>
#include <vector>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/train/batch_role.hpp>
#include <SushiAI/train/dense_losses.hpp>
#include <SushiAI/train/dense_targets.hpp>
#include <SushiAI/train/host_values.hpp>
#include <SushiAI/train/kept_rows.hpp>

#include "dense_objective.hpp"

namespace SushiAI
{
    namespace Train
    {
        Result<std::unique_ptr<MeanSquaredErrorObjective>> MeanSquaredErrorObjective::make(
            std::int64_t batch, std::int64_t width, DType dtype,
            ObjectiveSelection selection) noexcept
        {
            SA_TRY(Detail::validate_dense_leaf(batch, width, dtype, K_MEAN_SQUARED_ERROR_TYPE));

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                return std::unique_ptr<MeanSquaredErrorObjective>(
                    new MeanSquaredErrorObjective(batch, width, dtype, std::move(selection)));
            });
        }

        MeanSquaredErrorObjective::MeanSquaredErrorObjective(std::int64_t batch,
                                                             std::int64_t width,
                                                             DType dtype,
                                                             ObjectiveSelection selection) noexcept
            : batch_(batch), width_(width), dtype_(dtype), selection_(std::move(selection))
        {

        }

        void MeanSquaredErrorObjective::declare(const Graph::GraphBuilder& builder,
                                                ObjectiveLeaves& leaves) const
        {
            leaves.add(Detail::K_TARGETS_LEAF,
                       builder.input(Shape{batch_, width_}, Detail::K_TARGETS_LEAF, dtype_));
            leaves.require(Data::TargetDescription{
                std::string(selection_.target.role_or(Data::K_ROLE_VALUES)), Shape{width_},
                DType::Float32, 0});
            selection_.mask.declare(builder, leaves, Shape{batch_, width_}, dtype_);
        }

        Result<LossTrace> MeanSquaredErrorObjective::loss(
            const Graph::GraphBuilder& builder, const Graph::Graph& graph,
            const NN::ModelOutputs& outputs, const ObjectiveLeaves& leaves) const
        {
            const Graph::ValueID predicted = Detail::require_matrix_output(
                graph, outputs, Shape{batch_, width_}, K_MEAN_SQUARED_ERROR_TYPE,
                selection_.output);

            SA_TRY_ASSIGN(const Graph::ValueID loss,
                          Detail::emit_dense_loss(
                              builder, graph,
                              emit_squared_terms(builder, graph, predicted,
                                                 leaves.at(Detail::K_TARGETS_LEAF)),
                              selection_.mask, leaves));
            return LossTrace(loss);
        }

        Status MeanSquaredErrorObjective::feed(const Data::Batch& batch,
                                               const ObjectiveLeaves& leaves,
                                               Graph::TensorPool& pool) const
        {
            SA_TRY_ASSIGN(const Data::BatchTarget* const values,
                          require_batch_role(batch,
                                             selection_.target.role_or(Data::K_ROLE_VALUES)));
            SA_TRY_ASSIGN(SushiBLAS::Tensor* const targets,
                          find_leaf_tensor(leaves, Detail::K_TARGETS_LEAF, pool));

            SA_TRY(fill_dense_targets(values->values, *targets));

            return selection_.mask.feed(batch, leaves, pool);
        }

        Status MeanSquaredErrorObjective::measure(const StepView& step, Metrics& metrics) const
        {
            const std::int64_t elements = batch_ * width_;

            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const predicted, selection_.output.resolve(step));
            SA_TRY_ASSIGN(const std::vector<double> outputs,
                          read_host_reals(*predicted, elements, "the outputs"));

            SA_TRY_ASSIGN(const Data::BatchTarget* const values,
                          require_batch_role(step.batch,
                                             selection_.target.role_or(Data::K_ROLE_VALUES)));
            SA_TRY_ASSIGN(const std::vector<double> truth,
                          read_host_reals(values->values, elements, "the values"));

            SA_TRY_ASSIGN(const KeptRows kept, selection_.mask.kept_rows_of(step.batch, batch_));

            // A dropped row's values are placeholders, so neither the sum nor the count has them.
            double total = 0.0;
            std::int64_t counted = 0;
            for (std::int64_t index = 0; index < elements; ++index)
            {
                if (!kept.keeps(index / width_))
                    continue;

                ++counted;
                const auto at = static_cast<std::size_t>(index);
                total += std::abs(outputs[at] - truth[at]);
            }

            const double mean = counted == 0 ? 0.0 : total / static_cast<double>(counted);

            return metrics.add(Metric{K_MEAN_ABSOLUTE_ERROR_METRIC, mean, false});
        }

        SushiRuntime::span<const std::string_view> MeanSquaredErrorObjective::metric_names()
            const noexcept
        {
            static constexpr std::string_view K_NAMES[] = {K_MEAN_ABSOLUTE_ERROR_METRIC};

            return SushiRuntime::span<const std::string_view>(K_NAMES, 1);
        }

        Result<std::unique_ptr<Objective>> make_mean_squared_error(
            const ObjectiveContext& given, const Config::Value& attributes,
            std::string_view context) noexcept
        {
            SA_TRY_ASSIGN(ObjectiveSelection selection, read_selection(attributes, context));
            SA_TRY_ASSIGN(const OutputShape* const port,
                          Detail::read_matrix_output(given, context, selection.output));

            SA_TRY_ASSIGN(std::unique_ptr<MeanSquaredErrorObjective> made,
                          MeanSquaredErrorObjective::make(port->shape[0], port->shape[1],
                                                          DType::Float32,
                                                          std::move(selection)));
            std::unique_ptr<Objective> base = std::move(made);
            return std::move(base);
        }

    } // namespace Train
} // namespace SushiAI
