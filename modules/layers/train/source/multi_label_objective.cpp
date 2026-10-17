/**************************************************************************/
/* multi_label_objective.cpp                                              */
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
 * @file multi_label_objective.cpp
 * @brief Implements the multi-label objective and the factory a training file reaches it by.
 * @author Mustafa Garip
 */

#include <SushiAI/train/multi_label_objective.hpp>

#include <string>
#include <utility>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/train/batch_role.hpp>
#include <SushiAI/train/binary_confusion.hpp>
#include <SushiAI/train/dense_losses.hpp>
#include <SushiAI/train/dense_targets.hpp>
#include <SushiAI/train/host_values.hpp>

#include "dense_objective.hpp"

namespace SushiAI
{
    namespace Train
    {
        Result<std::unique_ptr<MultiLabelObjective>> MultiLabelObjective::make(
            std::int64_t batch, std::int64_t classes, double threshold, DType dtype)
        {
            SA_TRY(Detail::validate_dense_leaf(batch, classes, dtype, K_MULTI_LABEL_TYPE));
            SA_TRY(validate_threshold(threshold));

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                return std::unique_ptr<MultiLabelObjective>(
                    new MultiLabelObjective(batch, classes, threshold, dtype));
            });
        }

        MultiLabelObjective::MultiLabelObjective(std::int64_t batch, std::int64_t classes,
                                                 double threshold, DType dtype) noexcept
            : batch_(batch), classes_(classes), threshold_(threshold), dtype_(dtype)
        {

        }

        void MultiLabelObjective::declare(const Graph::GraphBuilder& builder,
                                          ObjectiveLeaves& leaves) const
        {
            leaves.add(Detail::K_TARGETS_LEAF,
                       builder.input(Shape{batch_, classes_}, Detail::K_TARGETS_LEAF, dtype_));
            leaves.require(Data::TargetDescription{std::string(Data::K_ROLE_LABELS),
                                                   Shape{classes_}, DType::Float32, 0});
        }

        Graph::ValueID MultiLabelObjective::loss(const Graph::GraphBuilder& builder,
                                                 const Graph::Graph& graph,
                                                 const NN::ModelOutputs& outputs,
                                                 const ObjectiveLeaves& leaves) const
        {
            const Graph::ValueID logits = Detail::require_matrix_output(
                graph, outputs, Shape{batch_, classes_}, K_MULTI_LABEL_TYPE);

            return emit_logistic_loss(builder, graph, logits, leaves.at(Detail::K_TARGETS_LEAF));
        }

        Status MultiLabelObjective::feed(const Data::Batch& batch, const ObjectiveLeaves& leaves,
                                         Graph::TensorPool& pool) const
        {
            SA_TRY_ASSIGN(const Data::BatchTarget* const labels,
                          require_batch_role(batch, Data::K_ROLE_LABELS));
            SA_TRY_ASSIGN(SushiBLAS::Tensor* const targets,
                          find_leaf_tensor(leaves, Detail::K_TARGETS_LEAF, pool));

            return fill_binary_targets(labels->values, *targets);
        }

        Status MultiLabelObjective::measure(const StepView& step, Metrics& metrics) const
        {
            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const logits, only_output(step));
            SA_TRY(require_host_reals(*logits, batch_ * classes_, "the logits"));

            SA_TRY_ASSIGN(const Data::BatchTarget* const labels,
                          require_batch_role(step.batch, Data::K_ROLE_LABELS));
            SA_TRY(require_host_reals(labels->values, batch_ * classes_, "the labels"));

            const double cut = logit_of(threshold_);
            double total = 0.0;
            for (std::int64_t column = 0; column < classes_; ++column)
            {
                total += f1_of(count_confusion(*logits, labels->values,
                                               ColumnOfLogits{batch_, classes_, column}, cut));
            }

            return metrics.add(Metric{"f1", total / static_cast<double>(classes_), true});
        }

        Result<std::unique_ptr<Objective>> make_multi_label(const ObjectiveContext& given,
                                                            const Config::Value& attributes,
                                                            std::string_view context)
        {
            SA_TRY_ASSIGN(const double threshold, read_threshold(attributes, context));
            SA_TRY_ASSIGN(const Shape logits, Detail::read_matrix_output(given, context));

            SA_TRY_ASSIGN(std::unique_ptr<MultiLabelObjective> made,
                          MultiLabelObjective::make(logits[0], logits[1], threshold));
            std::unique_ptr<Objective> base = std::move(made);
            return std::move(base);
        }

    } // namespace Train
} // namespace SushiAI
