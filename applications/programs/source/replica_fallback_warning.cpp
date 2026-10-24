/**************************************************************************/
/* replica_fallback_warning.cpp                                           */
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
 * @file replica_fallback_warning.cpp
 * @brief Implements ReplicaFallbackWarning: one WARN per replica and cause.
 * @author Mustafa Garip
 */

#include "replica_fallback_warning.hpp"

#include <SushiLog/text/text_format.hpp>
#include <SushiRuntime/placement/placement_rule.hpp>
#include <SushiRuntime/placement/placement_target_matching.hpp>
#include <SushiAI/log/log.hpp>

#include "programs_log_category.hpp"

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Returns what @p policy makes a replica do while its worker is absent. */
            [[nodiscard]] std::string describe_absent_action(
                const SushiRuntime::AbsentPolicy& policy)
            {
                switch (policy.action)
                {
                    case SushiRuntime::AbsentAction::Local:
                        return "runs on the master";
                    case SushiRuntime::AbsentAction::Wait:
                        return SushiLog::format_text(
                            "waits up to {} ms for it at each step, then runs on the master",
                            policy.bound.count());
                    case SushiRuntime::AbsentAction::Fail:
                        break;
                }
                return "fails the run";
            }
        } // namespace

        ReplicaFallbackWarning::ReplicaFallbackWarning(
            const Distribution::ClusterSection& section)
        {
            const std::size_t count = section.plan().replica_count();
            replicas_.resize(count);
            for (std::size_t replica = 0; replica < count; ++replica)
            {
                Replica& entry = replicas_[replica];
                entry.worker = section.worker_of(replica);
                entry.absent_action = describe_absent_action(section.absent_policy_of(replica));
                entry.on_master =
                    SushiRuntime::machine_names_match(entry.worker, section.worker_of(0));
            }
        }

        void ReplicaFallbackWarning::warn_absent_at_attach(const AwaitedWorker& worker)
        {
            for (const std::size_t replica : worker.replicas)
            {
                if (replica >= replicas_.size() || replicas_[replica].warned_at_attach)
                    continue;

                Replica& entry = replicas_[replica];
                entry.warned_at_attach = true;
                SA_LOG_WARN(K_PROGRAMS_LOG_CATEGORY,
                            "worker '{}' did not attach; replica {} {}, so the run is not "
                            "repeatable from the step it first runs there",
                            worker.name, replica, entry.absent_action);
            }
        }

        void ReplicaFallbackWarning::on_served(std::uint64_t replica,
                                               const Distribution::ReplicaPosition& position)
        {
            if (replica >= replicas_.size())
                return;

            Replica& entry = replicas_[static_cast<std::size_t>(replica)];
            if (entry.on_master || entry.warned_on_master)
                return;

            entry.warned_on_master = true;
            SA_LOG_WARN(K_PROGRAMS_LOG_CATEGORY,
                        "replica {} ran on the master at step {} (epoch {}, batch {}), not on "
                        "worker '{}' its rule names; the run is not repeatable from this step, "
                        "and this replica is not warned of again",
                        replica, position.step, position.epoch, position.batch_index,
                        entry.worker);
        }

    } // namespace Apps
} // namespace SushiAI
