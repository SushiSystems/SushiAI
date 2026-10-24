/**************************************************************************/
/* cluster_section.hpp                                                    */
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
 * @file cluster_section.hpp
 * @brief Declares ClusterSection, the "cluster" section of a training file.
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/placement/placement_rule.hpp>
#include <SushiRuntime/placement/placement_rules.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/distribution/replica_plan.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        /** @brief Names the key of a training file that holds the cluster section. */
        inline constexpr std::string_view K_CLUSTER_KEY = "cluster";

        /** @brief Holds the master's wait for its workers when "attach_timeout_ms" is absent. */
        inline constexpr std::chrono::milliseconds K_DEFAULT_ATTACH_TIMEOUT{60000};

        /**
         * @brief Holds a checked cluster section: where the master listens, the secret, and what
         *        each replica is, where it runs and what happens when its worker is absent.
         *
         * Replica 0 is the master. Every replica names the machine it runs on, the master's own
         * machine included, so no replica is a special case.
         */
        class ClusterSection final
        {
            public:
                /**
                 * @brief Reads and checks the cluster section of a training file.
                 * @param section The value of the "cluster" key.
                 * @param base The directory a relative "secret_file" resolves against.
                 * @return The section, or InvalidDescription naming the key that breaks a rule
                 *         of the README, or what ReplicaPlan::make refuses.
                 */
                [[nodiscard]] static Result<ClusterSection>
                read(const Config::Value& section, const std::filesystem::path& base);

                /** @brief Returns the address the master listens on, as the file spells it. */
                [[nodiscard]] const std::string& listen_address() const noexcept;

                /** @brief Returns the secret file, resolved against the base of read(). */
                [[nodiscard]] const std::filesystem::path& secret_file() const noexcept;

                /** @brief Returns how long the master waits for every worker to attach. */
                [[nodiscard]] std::chrono::milliseconds attach_timeout() const noexcept;

                /** @brief Returns the split of a batch over the replicas. */
                [[nodiscard]] const ReplicaPlan& plan() const noexcept;

                /**
                 * @brief Returns the name of the machine replica @p replica runs on.
                 * @pre @p replica is below plan().replica_count().
                 */
                [[nodiscard]] const std::string& worker_of(std::size_t replica) const;

                /**
                 * @brief Returns what happens when the machine of replica @p replica is absent.
                 * @pre @p replica is below plan().replica_count().
                 */
                [[nodiscard]] const SushiRuntime::AbsentPolicy& absent_policy_of(
                    std::size_t replica) const;

                /**
                 * @brief Turns the replicas into placement rules, one per replica in order.
                 *
                 * Rule k matches K_REPLICA_PLACEMENT_CLASS and key k, targets worker_of(k), and
                 * carries the priority and absent policy of the replica.
                 * @return The rules, or InvalidDescription naming the replica for a rule the
                 *         runtime refuses.
                 */
                [[nodiscard]] Result<SushiRuntime::PlacementRules> placement_rules() const;

            private:
                /** @brief What one replica states besides its rows. */
                struct Replica
                {
                    /// @brief The machine the replica runs on.
                    std::string worker;

                    /// @brief The priority of the replica's rule.
                    int priority = 0;

                    /// @brief What happens when the machine is absent.
                    SushiRuntime::AbsentPolicy if_absent;
                };

                /** @brief Takes the checked parts of a section. */
                ClusterSection(std::string listen_address, std::filesystem::path secret_file,
                               std::chrono::milliseconds attach_timeout, ReplicaPlan plan,
                               std::vector<Replica> replicas);

                /// @brief The address the master listens on.
                std::string listen_address_;

                /// @brief The secret file.
                std::filesystem::path secret_file_;

                /// @brief How long the master waits for every worker to attach; positive.
                std::chrono::milliseconds attach_timeout_;

                /// @brief The split of a batch.
                ReplicaPlan plan_;

                /// @brief The replicas, in replica order.
                std::vector<Replica> replicas_;
        };

    } // namespace Distribution
} // namespace SushiAI
