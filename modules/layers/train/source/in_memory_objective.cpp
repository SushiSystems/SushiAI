/**************************************************************************/
/* in_memory_objective.cpp                                                */
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
 * @file in_memory_objective.cpp
 * @brief Implements InMemoryObjective, which declares, feeds, measures and answers from a script.
 * @author Mustafa Garip
 */

#include <SushiAI/train/in_memory_objective.hpp>

#include <cstdint>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Train
    {
        namespace
        {
            /** @brief Writes @p value, converted to Element, into @p count elements. */
            template<typename Element>
            void fill_elements(Element* elements, std::int64_t count, double value) noexcept
            {
                for (std::int64_t index = 0; index < count; ++index)
                    elements[index] = static_cast<Element>(value);
            }

            /**
             * @brief Writes @p value into every element of @p tensor.
             * @return Unsupported for an element type other than float32, float64 and int32.
             */
            [[nodiscard]] Status fill_tensor(SushiBLAS::Tensor& tensor, double value)
            {
                switch (tensor.dtype)
                {
                    case DType::Float32:
                        fill_elements(tensor.data_as<float>(), tensor.num_elements, value);
                        return Status();
                    case DType::Float64:
                        fill_elements(tensor.data_as<double>(), tensor.num_elements, value);
                        return Status();
                    case DType::Int32:
                        fill_elements(tensor.data_as<std::int32_t>(), tensor.num_elements, value);
                        return Status();
                    default:
                        break;
                }

                SA_FAIL_IF(true, StatusCode::Unsupported,
                           "InMemoryObjective::feed: a leaf of {} cannot be filled",
                           dtype_name(tensor.dtype));
                return Status();
            }
        } // namespace

        void InMemoryObjective::declare(const Graph::GraphBuilder& builder,
                                        ObjectiveLeaves& leaves) const
        {
            ++declare_count_;
            for (const auto& leaf : leaves_)
                leaves.add(leaf.first, leaf.second);

            for (const InputLeaf& input : inputs_)
                leaves.add(input.role, builder.input(input.shape, input.role, input.dtype));

            for (const Data::TargetDescription& role : required_)
                leaves.require(role);
        }

        Result<LossTrace> InMemoryObjective::loss(const Graph::GraphBuilder& builder,
                                                  const Graph::Graph& graph,
                                                  const NN::ModelOutputs& outputs,
                                                  const ObjectiveLeaves& leaves) const
        {
            ++loss_count_;
            if (!loss_failure_.is_ok())
                return loss_failure_;

            if (emitter_)
            {
                SA_TRY_ASSIGN(const Graph::ValueID emitted,
                              emitter_(builder, graph, outputs, leaves));
                return LossTrace(emitted);
            }

            SA_FAIL_IF(loss_ == Graph::K_NO_VALUE, StatusCode::InvalidDescription,
                       "InMemoryObjective::loss: no loss value was scripted");
            return LossTrace(loss_);
        }

        Status InMemoryObjective::feed(const Data::Batch& batch, const ObjectiveLeaves& leaves,
                                       Graph::TensorPool& pool) const
        {
            ++feed_count_;
            if (fills_.empty() && feed_failure_.is_ok())
                return Objective::feed(batch, leaves, pool);

            for (const auto& fill : fills_)
            {
                SA_TRY_ASSIGN(SushiBLAS::Tensor* const tensor,
                              find_leaf_tensor(leaves, fill.first, pool));
                SA_TRY(fill_tensor(*tensor, fill.second));
            }

            return feed_failure_;
        }

        Status InMemoryObjective::measure(const StepView&, Metrics& metrics) const
        {
            ++measure_count_;
            for (const auto& metric : metrics_)
                SA_TRY(metrics.add(Metric{metric.first, metric.second}));

            return Status();
        }

    } // namespace Train
} // namespace SushiAI
