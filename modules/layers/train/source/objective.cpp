/**************************************************************************/
/* objective.cpp                                                          */
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
 * @file objective.cpp
 * @brief Implements training objective functions.
 * @author Mustafa Garip
 */

#include <SushiAI/train/objective.hpp>

#include <cstddef>
#include <cstdint>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/nn/loss.hpp>
#include <SushiAI/train/batch_role.hpp>
#include <SushiAI/train/host_values.hpp>
#include <SushiAI/train/kept_rows.hpp>
#include <SushiAI/train/one_hot.hpp>

namespace SushiAI
{
    namespace Train
    {
        void ObjectiveLeaves::add(std::string_view role, Graph::ValueID id) noexcept
        {
            SA_FATAL_IF(has(role),
                        "the objective declared the leaf '{}' twice; each role names "
                        "one value", role);
            leaves_.emplace_back(std::string(role), id);
        }

        bool ObjectiveLeaves::has(std::string_view role) const noexcept
        {
            for (const std::pair<std::string, Graph::ValueID>& leaf : leaves_)
                if (leaf.first == role) return true;
            return false;
        }

        Graph::ValueID ObjectiveLeaves::at(std::string_view role) const noexcept
        {
            for (const std::pair<std::string, Graph::ValueID>& leaf : leaves_)
                if (leaf.first == role) return leaf.second;

            std::string declared;
            for (const std::pair<std::string, Graph::ValueID>& leaf : leaves_)
            {
                if (!declared.empty()) declared += ", ";
                declared += "'" + leaf.first + "'";
            }
            if (declared.empty()) declared = "(none)";

            SA_FATAL("the objective declared no leaf '{}'; it declared {}", role, declared);
        }

        void ObjectiveLeaves::require(Data::TargetDescription role) noexcept
        {
            required_.push_back(std::move(role));
        }

        Result<SushiBLAS::Tensor*> find_leaf_tensor(const ObjectiveLeaves& leaves,
                                                    std::string_view role,
                                                    Graph::TensorPool& pool) noexcept
        {
            Graph::ValueID id = Graph::K_NO_VALUE;
            for (const std::pair<std::string, Graph::ValueID>& leaf : leaves.all())
            {
                if (leaf.first == role)
                    id = leaf.second;
            }

            SA_FAIL_IF(id == Graph::K_NO_VALUE, StatusCode::InvalidDescription,
                       "The objective declared no leaf '{}' in the leaves it was given", role);
            SA_FAIL_IF(!pool.is_bound(id), StatusCode::InvalidDescription,
                       "The pool holds no tensor for the objective's leaf '{}'", role);

            return &pool.tensor(id);
        }

        Result<NN::ModelOutputs> Objective::declare_metric_values(
            const Graph::GraphBuilder&, const NN::ModelOutputs&, const LossTrace&) const
        {
            return Result<NN::ModelOutputs>(NN::ModelOutputs());
        }

        Status Objective::feed(const Data::Batch&, const ObjectiveLeaves&,
                               Graph::TensorPool&) const
        {
            // The base declares no leaf, so it refuses every batch.
            SA_FAIL_IF(true, StatusCode::Unsupported,
                       "The objective '{}' is fed by its own step, which computes its "
                       "targets between two runs; no batch alone fills them",
                       name());

            return Status();
        }

        bool Objective::is_fed_by_batch() const noexcept
        {
            return false;
        }

        Status Objective::measure(const StepView&, Metrics&) const
        {
            return Status();
        }

        SushiRuntime::span<const std::string_view> Objective::metric_names() const noexcept
        {
            return SushiRuntime::span<const std::string_view>();
        }

        void CrossEntropyObjective::declare(const Graph::GraphBuilder& builder,
                                            ObjectiveLeaves& leaves) const
        {
            // Leaf matches the expected targets shape and element type.
            leaves.add("targets", builder.input(Shape{batch_, classes_}, "targets", dtype_));
            leaves.require(Data::TargetDescription{
                std::string(selection_.target.role_or(Data::K_ROLE_LABELS)), Shape{},
                DType::Int32, classes_});
            selection_.mask.declare(builder, leaves, Shape{batch_, classes_}, dtype_);
        }

        Result<LossTrace> CrossEntropyObjective::loss(
            const Graph::GraphBuilder& builder, const Graph::Graph& graph,
            const NN::ModelOutputs& outputs, const ObjectiveLeaves& leaves) const
        {
            // Cross-entropy reads one output: the named one, or the only one.
            SA_TRY_ASSIGN(const Graph::ValueID output, selection_.output.resolve(outputs));

            // By value: every emission below can reallocate the value vector.
            const Shape output_shape = graph.value(output).shape;
            const Shape expected_shape{batch_, classes_};
            SA_FAIL_IF(output_shape != expected_shape, StatusCode::InvalidDescription,
                       "The model produced {} but the loss needs {}", output_shape.to_string(),
                       expected_shape.to_string());

            if (!selection_.mask.is_active())
            {
                SA_TRY_ASSIGN(const Graph::ValueID loss,
                              NN::CrossEntropyLoss{}.forward(builder, output,
                                                             leaves.at("targets")));
                return LossTrace(loss);
            }

            // The terms are t * log_softmax(x); a row's cross entropy is minus their row sum.
            const Graph::Traced terms =
                builder.mul(leaves.at("targets"), builder.log_softmax(output));
            SA_TRY_ASSIGN(const Graph::ValueID masked,
                          selection_.mask.emit_weighted(builder, graph, terms, leaves, -1.0));
            return LossTrace(masked);
        }

        Status CrossEntropyObjective::feed(const Data::Batch& batch,
                                           const ObjectiveLeaves& leaves,
                                           Graph::TensorPool& pool) const
        {
            SA_TRY_ASSIGN(const Data::BatchTarget* const labels,
                          require_batch_role(batch, selection_.target.role_or(Data::K_ROLE_LABELS)));
            SA_TRY_ASSIGN(SushiBLAS::Tensor* const targets,
                          find_leaf_tensor(leaves, "targets", pool));

            SA_TRY(fill_one_hot(labels->values, *targets));

            return selection_.mask.feed(batch, leaves, pool);
        }

        Status CrossEntropyObjective::measure(const StepView& step, Metrics& metrics) const
        {
            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const logits, selection_.output.resolve(step));
            SA_TRY(require_host_reals(*logits, batch_ * classes_, "the logits"));

            const std::string_view role = selection_.target.role_or(Data::K_ROLE_LABELS);
            SA_TRY_ASSIGN(const Data::BatchTarget* const labels,
                          require_batch_role(step.batch, role));
            SA_FAIL_IF(labels->values.dtype != DType::Int32
                       || labels->values.num_elements != batch_,
                       StatusCode::Incompatible,
                       "Accuracy reads the role '{}' as {} {} values; the batch holds {} {}",
                       role, batch_, dtype_name(DType::Int32),
                       labels->values.num_elements, dtype_name(labels->values.dtype));
            const int32_t* const label = labels->values.data_as<int32_t>();

            SA_TRY_ASSIGN(const KeptRows kept, selection_.mask.kept_rows_of(step.batch, batch_));

            // A dropped row's label is a placeholder, so neither side of the fraction counts it.
            std::size_t correct = 0;
            std::size_t counted = 0;
            for (int64_t row = 0; row < batch_; ++row)
            {
                if (!kept.keeps(row))
                    continue;

                ++counted;
                int64_t best = 0;
                for (int64_t column = 1; column < classes_; ++column)
                {
                    if (host_real(*logits, row * classes_ + column)
                        > host_real(*logits, row * classes_ + best))
                        best = column;
                }

                if (best == label[row])
                    ++correct;
            }

            const double accuracy = counted == 0
                ? 0.0 : static_cast<double>(correct) / static_cast<double>(counted);

            return metrics.add(Metric{K_ACCURACY_METRIC, accuracy, true});
        }

        SushiRuntime::span<const std::string_view> CrossEntropyObjective::metric_names()
            const noexcept
        {
            static constexpr std::string_view K_NAMES[] = {K_ACCURACY_METRIC};

            return SushiRuntime::span<const std::string_view>(K_NAMES, 1);
        }

        Result<std::unique_ptr<Objective>> make_cross_entropy(const ObjectiveContext& given,
                                                              const Config::Value& attributes,
                                                              std::string_view context) noexcept
        {
            SA_TRY_ASSIGN(ObjectiveSelection selection, read_selection(attributes, context));

            // Data that states no class count leaves the classes to the width of the output.
            std::int64_t classes = given.classes;
            if (selection.output.is_named())
            {
                SA_TRY_ASSIGN(const OutputShape* const port,
                              selection.output.resolve(given, context));
                if (classes < 1 && port->shape.rank() == 2)
                    classes = port->shape[1];
            }

            SA_FAIL_IF(given.batch < 1 || classes < 1, StatusCode::InvalidDescription,
                       "{} needs a batch and a class count of at least 1; the data gives a "
                       "batch of {} and {} classes",
                       context, given.batch, classes);

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                return std::unique_ptr<Objective>(std::make_unique<CrossEntropyObjective>(
                    given.batch, classes, DType::Float32, std::move(selection)));
            });
        }

    } // namespace Train
} // namespace SushiAI
