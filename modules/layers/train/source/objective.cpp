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
#include <SushiAI/train/one_hot.hpp>

namespace SushiAI
{
    namespace Train
    {
        void ObjectiveLeaves::add(std::string_view role, Graph::ValueID id)
        {
            SA_THROW_IF(has(role),
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

        Graph::ValueID ObjectiveLeaves::at(std::string_view role) const
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

            SA_THROW_IF(true, "the objective declared no leaf '{}'; it declared {}",
                        role, declared);
            return Graph::K_NO_VALUE;   // unreachable: SA_THROW_IF(true) always throws
        }

        void ObjectiveLeaves::require(Data::TargetDescription role)
        {
            required_.push_back(std::move(role));
        }

        Result<SushiBLAS::Tensor*> find_leaf_tensor(const ObjectiveLeaves& leaves,
                                                    std::string_view role,
                                                    Graph::TensorPool& pool)
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

        void CrossEntropyObjective::declare(const Graph::GraphBuilder& builder,
                                            ObjectiveLeaves& leaves) const
        {
            // Leaf matches the expected targets shape and element type.
            leaves.add("targets", builder.input(Shape{batch_, classes_}, "targets", dtype_));
            leaves.require(Data::TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{},
                                                   DType::Int32, classes_});
        }

        Graph::ValueID CrossEntropyObjective::loss(const Graph::GraphBuilder& builder,
                                                   const Graph::Graph& graph,
                                                   const NN::ModelOutputs& outputs,
                                                   const ObjectiveLeaves& leaves) const
        {
            // Cross-entropy reads exactly one output; a model with several must be refused.
            const Result<Graph::ValueID> single = outputs.only();
            SA_THROW_IF(!single.has_value(), "{}", single.status().message());
            const Graph::ValueID output = single.value();

            // By value: every emission below can reallocate the value vector.
            const Shape output_shape = graph.value(output).shape;
            const Shape expected_shape{batch_, classes_};
            SA_THROW_IF(output_shape != expected_shape,
                        "The model produced {} but the loss needs {}",
                        output_shape.to_string(), expected_shape.to_string());

            return NN::CrossEntropyLoss{}.forward(builder, output, leaves.at("targets"));
        }

        Status CrossEntropyObjective::feed(const Data::Batch& batch,
                                           const ObjectiveLeaves& leaves,
                                           Graph::TensorPool& pool) const
        {
            SA_TRY_ASSIGN(const Data::BatchTarget* const labels,
                          require_batch_role(batch, Data::K_ROLE_LABELS));
            SA_TRY_ASSIGN(SushiBLAS::Tensor* const targets,
                          find_leaf_tensor(leaves, "targets", pool));

            return fill_one_hot(labels->values, *targets);
        }

        Status CrossEntropyObjective::measure(const StepView& step, Metrics& metrics) const
        {
            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const logits, only_output(step));
            SA_TRY(require_host_reals(*logits, batch_ * classes_, "the logits"));

            SA_TRY_ASSIGN(const Data::BatchTarget* const labels,
                          require_batch_role(step.batch, Data::K_ROLE_LABELS));
            SA_FAIL_IF(labels->values.dtype != DType::Int32
                       || labels->values.num_elements != batch_,
                       StatusCode::Incompatible,
                       "Accuracy reads the role '{}' as {} {} values; the batch holds {} {}",
                       Data::K_ROLE_LABELS, batch_, dtype_name(DType::Int32),
                       labels->values.num_elements, dtype_name(labels->values.dtype));
            const int32_t* const label = labels->values.data_as<int32_t>();

            std::size_t correct = 0;
            for (int64_t row = 0; row < batch_; ++row)
            {
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

            return metrics.add(Metric{
                "accuracy", static_cast<double>(correct) / static_cast<double>(batch_), true});
        }

        Result<std::unique_ptr<Objective>> make_cross_entropy(const ObjectiveContext& given,
                                                              const Config::Value& attributes,
                                                              std::string_view context)
        {
            SA_FAIL_IF(!attributes.keys().empty(), StatusCode::InvalidDescription,
                       "{} states the key '{}', but a cross_entropy objective has no keys",
                       context, attributes.keys().front());
            SA_FAIL_IF(given.batch < 1 || given.classes < 1, StatusCode::InvalidDescription,
                       "{} needs a batch and a class count of at least 1; the data gives a "
                       "batch of {} and {} classes",
                       context, given.batch, given.classes);

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                return std::unique_ptr<Objective>(
                    std::make_unique<CrossEntropyObjective>(given.batch, given.classes));
            });
        }

    } // namespace Train
} // namespace SushiAI
