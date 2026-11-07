/**************************************************************************/
/* in_memory_worker_watch.hpp                                              */
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
 * @file in_memory_worker_watch.hpp
 * @brief Declares InMemoryWorkerWatch, the WorkerWatch whose workers arrive on a clock
 *        it moves itself.
 * @author Mustafa Garip
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "master_cluster.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @class InMemoryWorkerWatch
         * @brief Lists each worker from the time it arrives and moves its clock to the next
         *        arrival instead of sleeping.
         */
        class InMemoryWorkerWatch final : public WorkerWatch
        {
            public:
                /** @brief Names one worker and the time, after the start, it attaches. */
                struct Arrival
                {
                    /// @brief The name its HELLO carries.
                    std::string name;

                    /// @brief How long after the start it attaches.
                    std::chrono::milliseconds at{0};
                };

                /** @brief Starts the clock at its epoch with @p arrivals still to come. */
                explicit InMemoryWorkerWatch(std::vector<Arrival> arrivals)
                    : arrivals_(std::move(arrivals))
                {

                }

                /** @brief Returns the names that have arrived by the clock's time. */
                [[nodiscard]] std::vector<std::string> attached_names() override
                {
                    std::vector<std::string> names;
                    for (const Arrival& arrival : arrivals_)
                        if (arrival.at <= elapsed_)
                            names.push_back(arrival.name);
                    return names;
                }

                /**
                 * @brief Moves the clock to the arrival that reaches @p count, or by
                 *        @p bound when none does within it.
                 */
                [[nodiscard]] bool wait_for_count(std::size_t count,
                                                  std::chrono::milliseconds bound) override
                {
                    std::vector<std::chrono::milliseconds> times;
                    for (const Arrival& arrival : arrivals_)
                        times.push_back(arrival.at);
                    std::sort(times.begin(), times.end());
                    if (count == 0 || count > times.size()
                        || times[count - 1] > elapsed_ + bound)
                    {
                        elapsed_ += bound;
                        return false;
                    }
                    elapsed_ = std::max(elapsed_, times[count - 1]);
                    return true;
                }

                /** @brief Returns the epoch plus the time the waits have passed. */
                [[nodiscard]] std::chrono::steady_clock::time_point now() override
                {
                    return std::chrono::steady_clock::time_point{} + elapsed_;
                }

                /** @brief Returns how long the waits have passed on the clock. */
                [[nodiscard]] std::chrono::milliseconds elapsed() const noexcept
                {
                    return elapsed_;
                }

            private:
                /// @brief The workers and when they attach.
                std::vector<Arrival> arrivals_;

                /// @brief The time passed since the start.
                std::chrono::milliseconds elapsed_{0};
        };
    } // namespace Apps
} // namespace SushiAI
