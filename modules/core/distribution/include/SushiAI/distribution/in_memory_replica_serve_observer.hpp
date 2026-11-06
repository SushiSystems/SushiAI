/**************************************************************************/
/* in_memory_replica_serve_observer.hpp                                   */
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
 * @file in_memory_replica_serve_observer.hpp
 * @brief Declares InMemoryReplicaServeObserver, the observer that keeps what it heard in memory.
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <SushiAI/distribution/replica_position.hpp>
#include <SushiAI/distribution/replica_serve_observer.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        /**
         * @brief Keeps every task a kernel served, in the order it was served; touches no device,
         *        file, thread or clock.
         *
         * It takes no lock, as the interface promises one task at a time. Read it after the
         * kernel has returned.
         */
        class InMemoryReplicaServeObserver final : public ReplicaServeObserver
        {
            public:
                /** @brief Holds the replica and the position of one served task. */
                struct Served
                {
                    /** @brief The replica the task ran. */
                    std::uint64_t replica = 0;

                    /** @brief The position the task ran at. */
                    ReplicaPosition position{};
                };

                /** @brief Appends the task to served(). */
                void on_served(std::uint64_t replica, const ReplicaPosition& position) override
                {
                    served_.push_back(Served{replica, position});
                }

                /** @brief Returns every task heard of, first heard first. */
                [[nodiscard]] const std::vector<Served>& served() const noexcept
                {
                    return served_;
                }

                /** @brief Returns how many tasks were heard of. */
                [[nodiscard]] std::size_t served_count() const noexcept
                {
                    return served_.size();
                }

                /** @brief Forgets every task heard of. */
                void clear() noexcept
                {
                    served_.clear();
                }

            private:
                /// @brief Holds the tasks heard of, in order.
                std::vector<Served> served_;
        };

    } // namespace Distribution
} // namespace SushiAI
