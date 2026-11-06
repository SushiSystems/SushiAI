/**************************************************************************/
/* replica_fallback_warning.hpp                                           */
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
 * @file replica_fallback_warning.hpp
 * @brief Declares ReplicaFallbackWarning, the master's WARN for a replica that falls back to it.
 * @author Mustafa Garip
 * @see docs/design/LAYERED_EXECUTION/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <SushiAI/distribution/cluster_section.hpp>
#include <SushiAI/distribution/replica_position.hpp>
#include <SushiAI/distribution/replica_serve_observer.hpp>

#include "master_cluster.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @brief Logs one WARN per replica and cause when a replica the section places on
         *        another machine runs on the master, since from then on the run is not
         *        repeatable.
         *
         * The master's replica kernel calls on_served() for every node that ran on the
         * master; a replica the section places on the master's own machine is not a fallback.
         */
        class ReplicaFallbackWarning final : public Distribution::ReplicaServeObserver
        {
            public:
                /** @brief Reads which replicas the section places off the master's machine. */
                explicit ReplicaFallbackWarning(const Distribution::ClusterSection& section);

                /**
                 * @brief Warns once that @p worker did not attach, and what the "if_absent" of
                 *        its replica makes that replica do.
                 */
                void warn_absent_at_attach(const AwaitedWorker& worker);

                /** @brief Warns the first time a replica placed elsewhere ran on the master. */
                void on_served(std::uint64_t replica,
                               const Distribution::ReplicaPosition& position) override;

            private:
                /** @brief Holds what the section says of one replica, and what was logged. */
                struct Replica
                {
                    /// @brief The worker its rule names.
                    std::string worker;

                    /// @brief What its "if_absent" does, as the WARN names it.
                    std::string absent_action;

                    /// @brief Whether its rule names the master's own machine.
                    bool on_master = false;

                    /// @brief Whether the WARN for its absence at attach was logged.
                    bool warned_at_attach = false;

                    /// @brief Whether the WARN for a run on the master was logged.
                    bool warned_on_master = false;
                };

                /// @brief Entry k describes replica k.
                std::vector<Replica> replicas_;
        };

    } // namespace Apps
} // namespace SushiAI
