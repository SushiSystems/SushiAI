/**************************************************************************/
/* replica_serve_observer.hpp                                             */
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
 * @file replica_serve_observer.hpp
 * @brief Declares ReplicaServeObserver, what a replica kernel tells about each task it serves.
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#pragma once

#include <cstdint>
#include <SushiAI/distribution/replica_position.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        /**
         * @brief Hears of each replica task a kernel serves, so the master learns that a
         *        replica's node ran on its own machine.
         */
        class ReplicaServeObserver
        {
            public:
                virtual ~ReplicaServeObserver() = default;

                /**
                 * @brief Called on the serving thread before the kernel runs replica
                 *        @p replica at @p position; the kernel serves one task at a time.
                 */
                virtual void on_served(std::uint64_t replica, const ReplicaPosition& position) = 0;

            protected:
                ReplicaServeObserver() = default;
                ReplicaServeObserver(const ReplicaServeObserver&) = default;
                ReplicaServeObserver& operator=(const ReplicaServeObserver&) = default;
                ReplicaServeObserver(ReplicaServeObserver&&) = default;
                ReplicaServeObserver& operator=(ReplicaServeObserver&&) = default;
        };

    } // namespace Distribution
} // namespace SushiAI
