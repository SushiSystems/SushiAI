/**************************************************************************/
/* objective.cpp                                                          */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file objective.cpp
 * @brief Implements training objective functions and objective registry lookups.
 * @author Mustafa Garip
 */

#include <SushiAI/train/objective.hpp>

#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/registry.hpp>
#include <SushiAI/nn/loss.hpp>

namespace SushiAI
{
    namespace Train
    {
        void ObjectiveLeaves::add(std::string_view role, Graph::ValueId id)
        {
            SA_THROW_IF(has(role),
                        "the objective declared the leaf '{}' twice; each role names "
                        "one value", role);
            leaves_.emplace_back(std::string(role), id);
        }

        bool ObjectiveLeaves::has(std::string_view role) const noexcept
        {
            for (const std::pair<std::string, Graph::ValueId>& leaf : leaves_)
                if (leaf.first == role) return true;
            return false;
        }

        Graph::ValueId ObjectiveLeaves::at(std::string_view role) const
        {
            for (const std::pair<std::string, Graph::ValueId>& leaf : leaves_)
                if (leaf.first == role) return leaf.second;

            std::string declared;
            for (const std::pair<std::string, Graph::ValueId>& leaf : leaves_)
            {
                if (!declared.empty()) declared += ", ";
                declared += "'" + leaf.first + "'";
            }
            if (declared.empty()) declared = "(none)";

            SA_THROW_IF(true, "the objective declared no leaf '{}'; it declared {}",
                        role, declared);
            return Graph::NO_VALUE;   // unreachable: SA_THROW_IF(true) always throws
        }

        void CrossEntropyObjective::declare(const Graph::GraphBuilder& builder,
                                            ObjectiveLeaves& leaves) const
        {
            // Leaf matches the expected targets shape and element type.
            leaves.add("targets", builder.input(Shape{batch_, classes_}, "targets", dtype_));
        }

        Graph::ValueId CrossEntropyObjective::loss(const Graph::GraphBuilder& builder,
                                                   const Graph::Graph& graph,
                                                   Graph::ValueId output,
                                                   const ObjectiveLeaves& leaves) const
        {
            // By value: every emission below can reallocate the value vector.
            const Shape output_shape = graph.value(output).shape;
            const Shape expected_shape{batch_, classes_};
            SA_THROW_IF(output_shape != expected_shape,
                        "The model produced {} but the loss needs {}",
                        output_shape.to_string(), expected_shape.to_string());

            return NN::CrossEntropyLoss{}.forward(builder, output, leaves.at("targets"));
        }

        std::optional<Metric> CrossEntropyObjective::score(const Data::Batch& batch,
                                                           const SushiBLAS::Tensor& output) const
        {
            // Accuracy evaluation requires readable host float32 or float64 logits.
            SA_THROW_IF(output.dtype != DType::FLOAT32 && output.dtype != DType::FLOAT64,
                        "Accuracy needs a host read of the logits and they are {}; only "
                        "float32 and float64 can be read back.", dtype_name(output.dtype));

            // Per-row argmax is performed on host to avoid whole-tensor reduction.
            const auto at = [&output](int64_t index) -> double
            {
                return output.dtype == DType::FLOAT64
                     ? output.data_as<double>()[index]
                     : static_cast<double>(output.data_as<float>()[index]);
            };

            std::size_t correct = 0;
            for (int64_t row = 0; row < batch_; ++row)
            {
                int64_t best = 0;
                for (int64_t column = 1; column < classes_; ++column)
                    if (at(row * classes_ + column) > at(row * classes_ + best)) best = column;

                if (best == batch.labels[static_cast<std::size_t>(row)]) ++correct;
            }

            return Metric{"accuracy",
                          static_cast<double>(correct) / static_cast<double>(batch_),
                          true};
        }

        std::unique_ptr<IObjective> make_cross_entropy(int64_t batch, int64_t classes,
                                                       const Config::Value& attributes,
                                                       std::string_view context)
        {
            // Cross-entropy takes no configuration attributes.
            static constexpr std::string_view NONE[] = {""};
            attributes.reject_unknown_keys(
                context, SushiRuntime::span<const std::string_view>(NONE,
                                                                    static_cast<std::size_t>(0)));

            return std::make_unique<CrossEntropyObjective>(batch, classes);
        }

        std::unique_ptr<IObjective> make_objective(std::string_view type,
                                                   int64_t batch, int64_t classes,
                                                   const Config::Value& attributes,
                                                   std::string_view context)
        {
            return Config::find_entry(OBJECTIVE_REGISTRY, type, "objective type")
                       .make(batch, classes, attributes, context);
        }

    } // namespace Train
} // namespace SushiAI
