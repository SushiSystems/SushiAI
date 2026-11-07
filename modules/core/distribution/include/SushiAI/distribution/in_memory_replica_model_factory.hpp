/**************************************************************************/
/* in_memory_replica_model_factory.hpp                                    */
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
 * @file in_memory_replica_model_factory.hpp
 * @brief Declares InMemoryReplicaModelFactory, the model factory that keeps its calls.
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <SushiAI/distribution/replica_draws.hpp>
#include <SushiAI/distribution/replica_model.hpp>
#include <SushiAI/execution/execution_backend.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        /**
         * @brief Builds each model through the factory it was given and keeps the draws and the
         *        micro-batch rows of every call, in the order they came; touches no device,
         *        file, thread or clock of its own.
         *
         * It takes no lock. Read it after the step that called it has returned.
         */
        class InMemoryReplicaModelFactory final
        {
            public:
                /** @brief Holds the arguments of one call, besides the backend. */
                struct Call
                {
                    /** @brief The draws the factory was called with. */
                    ReplicaDraws draws{};

                    /** @brief The micro-batch rows the factory was called with. */
                    std::int64_t rows = 0;
                };

                /** @brief Builds each model through @p inner. */
                explicit InMemoryReplicaModelFactory(ReplicaModelFactory inner)
                    : inner_(std::move(inner))
                {

                }

                /** @brief Keeps the call and returns what the inner factory returns. */
                [[nodiscard]] Result<std::unique_ptr<ReplicaModel>> make(
                    ExecutionBackend& backend, const ReplicaDraws& draws, std::int64_t rows)
                {
                    calls_.push_back(Call{draws, rows});
                    return inner_(backend, draws, rows);
                }

                /** @brief Returns a factory that calls make(); it lives as long as this object. */
                [[nodiscard]] ReplicaModelFactory as_factory()
                {
                    return [this](ExecutionBackend& backend, const ReplicaDraws& draws,
                                  std::int64_t rows)
                    {
                        return make(backend, draws, rows);
                    };
                }

                /** @brief Returns every call received, first received first. */
                [[nodiscard]] const std::vector<Call>& calls() const noexcept
                {
                    return calls_;
                }

            private:
                /// @brief Builds the models.
                ReplicaModelFactory inner_;

                /// @brief Holds the calls received, in order.
                std::vector<Call> calls_;
        };

    } // namespace Distribution
} // namespace SushiAI
