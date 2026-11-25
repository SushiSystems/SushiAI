/**************************************************************************/
/* binary_cross_entropy_objective.cpp                                     */
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
 * @file binary_cross_entropy_objective.cpp
 * @brief Implements the binary objective and the factory a training file reaches it by.
 * @author Mustafa Garip
 */

#include <SushiAI/train/binary_cross_entropy_objective.hpp>

#include <string>
#include <utility>
#include <vector>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/train/batch_role.hpp>
#include <SushiAI/train/binary_confusion.hpp>
#include <SushiAI/train/dense_losses.hpp>
#include <SushiAI/train/dense_targets.hpp>
#include <SushiAI/train/host_values.hpp>
#include <SushiAI/train/kept_rows.hpp>

#include "dense_objective.hpp"

namespace SushiAI
{
    namespace Train
    {
        Result<std::unique_ptr<BinaryCrossEntropyObjective>> BinaryCrossEntropyObjective::make(
            std::int64_t batch, double threshold, DType dtype,
            ObjectiveSelection selection) noexcept
        {
            SA_TRY(Detail::validate_dense_leaf(batch, 1, dtype, K_BINARY_CROSS_ENTROPY_TYPE));
            SA_TRY(validate_threshold(threshold));

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                return std::unique_ptr<BinaryCrossEntropyObjective>(
                    new BinaryCrossEntropyObjective(batch, threshold, dtype, std::move(selection)));
            });
        }

        BinaryCrossEntropyObjective::BinaryCrossEntropyObjective(std::int64_t batch,
                                                                 double threshold, DType dtype,
                                                                 ObjectiveSelection selection) noexcept
            : batch_(batch), threshold_(threshold), dtype_(dtype), selection_(std::move(selection))
        {

        }

        void BinaryCrossEntropyObjective::declare(const Graph::GraphBuilder& builder,
                                                  ObjectiveLeaves& leaves) const
        {
            leaves.add(Detail::K_TARGETS_LEAF,
                       builder.input(Shape{batch_, 1}, Detail::K_TARGETS_LEAF, dtype_));
            leaves.require(Data::TargetDescription{
                std::string(selection_.target.role_or(Data::K_ROLE_LABELS)), Shape{1},
                DType::Float32, 0});
            selection_.mask.declare(builder, leaves, Shape{batch_, 1}, dtype_);
        }

        Result<LossTrace> BinaryCrossEntropyObjective::loss(
            const Graph::GraphBuilder& builder, const Graph::Graph& graph,
            const NN::ModelOutputs& outputs, const ObjectiveLeaves& leaves) const
        {
            const Graph::ValueID logits = Detail::require_matrix_output(
                graph, outputs, Shape{batch_, 1}, K_BINARY_CROSS_ENTROPY_TYPE, selection_.output);

            SA_TRY_ASSIGN(const Graph::ValueID loss,
                          Detail::emit_dense_loss(
                              builder, graph,
                              emit_logistic_terms(builder, graph, logits,
                                                  leaves.at(Detail::K_TARGETS_LEAF)),
                              selection_.mask, leaves));
            return LossTrace(loss);
        }

        Status BinaryCrossEntropyObjective::feed(const Data::Batch& batch,
                                                 const ObjectiveLeaves& leaves,
                                                 Graph::TensorPool& pool) const
        {
            SA_TRY_ASSIGN(const Data::BatchTarget* const labels,
                          require_batch_role(batch,
                                             selection_.target.role_or(Data::K_ROLE_LABELS)));
            SA_TRY_ASSIGN(SushiBLAS::Tensor* const targets,
                          find_leaf_tensor(leaves, Detail::K_TARGETS_LEAF, pool));

            SA_TRY(fill_binary_targets(labels->values, *targets));

            return selection_.mask.feed(batch, leaves, pool);
        }

        Status BinaryCrossEntropyObjective::measure(const StepView& step, Metrics& metrics) const
        {
            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const logits, selection_.output.resolve(step));
            SA_TRY_ASSIGN(const std::vector<double> logit_values,
                          read_host_reals(*logits, batch_, "the logits"));

            SA_TRY_ASSIGN(const Data::BatchTarget* const labels,
                          require_batch_role(step.batch,
                                             selection_.target.role_or(Data::K_ROLE_LABELS)));
            SA_TRY_ASSIGN(const std::vector<double> label_values,
                          read_host_reals(labels->values, batch_, "the labels"));

            SA_TRY_ASSIGN(const KeptRows kept, selection_.mask.kept_rows_of(step.batch, batch_));

            const BinaryConfusion counts = count_confusion(
                SushiRuntime::span<const double>(logit_values.data(), logit_values.size()),
                SushiRuntime::span<const double>(label_values.data(), label_values.size()),
                ColumnOfLogits{batch_, 1, 0}, logit_of(threshold_), kept);

            SA_TRY(metrics.add(Metric{K_ACCURACY_METRIC, accuracy_of(counts), true}));
            SA_TRY(metrics.add(Metric{K_PRECISION_METRIC, precision_of(counts), true}));

            return metrics.add(Metric{K_RECALL_METRIC, recall_of(counts), true});
        }

        SushiRuntime::span<const std::string_view> BinaryCrossEntropyObjective::metric_names()
            const noexcept
        {
            static constexpr std::string_view K_NAMES[] = {K_ACCURACY_METRIC, K_PRECISION_METRIC,
                                                           K_RECALL_METRIC};

            return SushiRuntime::span<const std::string_view>(K_NAMES, 3);
        }

        Result<std::unique_ptr<Objective>> make_binary_cross_entropy(
            const ObjectiveContext& given, const Config::Value& attributes,
            std::string_view context) noexcept
        {
            SA_TRY_ASSIGN(ObjectiveSelection selection,
                          read_selection(attributes, context, K_THRESHOLD_KEY));
            SA_TRY_ASSIGN(const double threshold, read_threshold(attributes, context));
            SA_TRY_ASSIGN(const OutputShape* const port,
                          Detail::read_matrix_output(given, context, selection.output));
            SA_FAIL_IF(port->shape[1] != 1, StatusCode::InvalidDescription,
                       "{} reads one logit per sample, but the model's output '{}' is {}",
                       context, port->name, port->shape.to_string());

            SA_TRY_ASSIGN(std::unique_ptr<BinaryCrossEntropyObjective> made,
                          BinaryCrossEntropyObjective::make(port->shape[0], threshold,
                                                            DType::Float32,
                                                            std::move(selection)));
            std::unique_ptr<Objective> base = std::move(made);
            return std::move(base);
        }

    } // namespace Train
} // namespace SushiAI
