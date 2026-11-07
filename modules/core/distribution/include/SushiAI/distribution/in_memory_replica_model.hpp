/**************************************************************************/
/* in_memory_replica_model.hpp                                            */
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
 * @file in_memory_replica_model.hpp
 * @brief Declares InMemoryReplicaModel, the ReplicaModel that holds shapes and no storage.
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/distribution/replica_model.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/graph/tensor_source.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        /**
         * @brief Holds an empty graph, a given count of parameters and of results, each one
         *        float32 element described and given no storage, and the rows it was fed;
         *        touches no device, file, thread or clock.
         *
         * A model with no result is the model a step refuses. Its graph holds no node, so it
         * is not compiled.
         */
        class InMemoryReplicaModel final : public ReplicaModel
        {
            public:
                /** @brief Describes @p parameter_count parameters and @p result_count results. */
                InMemoryReplicaModel(std::size_t parameter_count, std::size_t result_count)
                    : pool_(source_)
                {
                    for (std::size_t index = 0; index < parameter_count; ++index)
                        parameters_.push_back(source_.create(Shape{1}, DType::Float32).value());
                    for (std::size_t index = 0; index < result_count; ++index)
                        results_.push_back(source_.create(Shape{1}, DType::Float32).value());
                }

                InMemoryReplicaModel(const InMemoryReplicaModel&) = delete;
                InMemoryReplicaModel& operator=(const InMemoryReplicaModel&) = delete;
                InMemoryReplicaModel(InMemoryReplicaModel&&) = delete;
                InMemoryReplicaModel& operator=(InMemoryReplicaModel&&) = delete;
                ~InMemoryReplicaModel() override = default;

                /** @brief Returns the graph, which holds no node. */
                [[nodiscard]] const Graph::Graph& graph() const noexcept override
                {
                    return graph_;
                }

                /** @brief Returns the pool, which draws from an in-memory tensor source. */
                [[nodiscard]] Graph::TensorPool& pool() noexcept override
                {
                    return pool_;
                }

                /** @brief Returns the parameters, as many as the constructor was given. */
                [[nodiscard]] SushiRuntime::span<const SushiBLAS::Tensor> parameters()
                    const noexcept override
                {
                    return SushiRuntime::span<const SushiBLAS::Tensor>(parameters_.data(),
                                                                       parameters_.size());
                }

                /** @brief Returns the results, as many as the constructor was given. */
                [[nodiscard]] SushiRuntime::span<const SushiBLAS::Tensor> results()
                    const noexcept override
                {
                    return SushiRuntime::span<const SushiBLAS::Tensor>(results_.data(),
                                                                       results_.size());
                }

                /** @brief Returns 0. */
                [[nodiscard]] std::uint64_t fingerprint() const noexcept override
                {
                    return 0;
                }

                /** @brief Keeps @p rows in fed() and succeeds. */
                [[nodiscard]] Status feed(const ReplicaRows& rows) override
                {
                    fed_.push_back(rows);
                    return Status();
                }

                /** @brief Returns every micro-batch fed, first fed first. */
                [[nodiscard]] const std::vector<ReplicaRows>& fed() const noexcept
                {
                    return fed_;
                }

            private:
                /// @brief Describes the tensors; the pool draws from it.
                Graph::InMemoryTensorSource source_;

                /// @brief The pool over source_.
                Graph::TensorPool pool_;

                /// @brief The graph, with no node.
                Graph::Graph graph_;

                /// @brief The parameters, in order.
                std::vector<SushiBLAS::Tensor> parameters_;

                /// @brief The results, in order.
                std::vector<SushiBLAS::Tensor> results_;

                /// @brief The micro-batches fed, in order.
                std::vector<ReplicaRows> fed_;
        };

    } // namespace Distribution
} // namespace SushiAI
