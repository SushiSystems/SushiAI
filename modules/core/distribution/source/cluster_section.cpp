/**************************************************************************/
/* cluster_section.cpp                                                    */
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
 * @file cluster_section.cpp
 * @brief Implements ClusterSection.
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#include <SushiAI/distribution/cluster_section.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <SushiAI/config/section_reader.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        namespace
        {
            using Config::Value;

            /** @brief Names the key that states where the master listens. */
            constexpr std::string_view K_LISTEN_KEY = "listen";

            /** @brief Names the key that states the file holding the cluster's secret. */
            constexpr std::string_view K_SECRET_FILE_KEY = "secret_file";

            /** @brief Names the key that lists the replicas. */
            constexpr std::string_view K_REPLICAS_KEY = "replicas";

            /** @brief Names the key of a replica that states the machine it runs on. */
            constexpr std::string_view K_WORKER_KEY = "worker";

            /** @brief Names the key of a replica that states r_k. */
            constexpr std::string_view K_ROWS_KEY = "rows";

            /** @brief Names the key of a replica that states m_k. */
            constexpr std::string_view K_MICRO_BATCH_KEY = "micro_batch";

            /** @brief Names the key of a replica that states the priority of its rule. */
            constexpr std::string_view K_PRIORITY_KEY = "priority";

            /** @brief Names the key of a replica that states what happens without its worker. */
            constexpr std::string_view K_IF_ABSENT_KEY = "if_absent";

            /** @brief Names the key of a replica that bounds a wait, in milliseconds. */
            constexpr std::string_view K_WAIT_MS_KEY = "wait_ms";

            /** @brief Spells the "if_absent" that runs the replica where it would run unruled. */
            constexpr std::string_view K_ABSENT_LOCAL = "local";

            /** @brief Spells the "if_absent" that waits for the worker. */
            constexpr std::string_view K_ABSENT_WAIT = "wait";

            /** @brief Spells the "if_absent" that ends the run. */
            constexpr std::string_view K_ABSENT_FAIL = "fail";

            /** @brief Lists the keys a cluster section has. */
            constexpr std::string_view K_SECTION_KEYS[] = {K_LISTEN_KEY, K_SECRET_FILE_KEY,
                                                           K_REPLICAS_KEY};

            /** @brief Lists the keys a replica has. */
            constexpr std::string_view K_REPLICA_KEYS[] = {K_WORKER_KEY,    K_ROWS_KEY,
                                                           K_MICRO_BATCH_KEY, K_PRIORITY_KEY,
                                                           K_IF_ABSENT_KEY, K_WAIT_MS_KEY};

            /** @brief Holds one replica entry as read, before the split is checked. */
            struct ReadReplica
            {
                /// @brief The rows and the micro-batch.
                ReplicaShare share;

                /// @brief The machine the replica runs on.
                std::string worker;

                /// @brief The priority of the replica's rule.
                int priority = 0;

                /// @brief What happens when the machine is absent.
                SushiRuntime::AbsentPolicy if_absent;
            };

            /** @brief Views a literal key list as the span the section reader takes. */
            template<std::size_t N>
            Config::KeySpan span_of(const std::string_view (&keys)[N]) noexcept
            {
                return Config::KeySpan(keys, N);
            }

            /**
             * @brief Reads the string member @p key of @p reader into @p field.
             * @return Success, or InvalidDescription for an absent key, a value that is no
             *         string and an empty string.
             */
            Status read_text(const Config::SectionReader& reader, std::string_view key,
                             std::string& field)
            {
                const std::string named = reader.member(key);
                SA_TRY_ASSIGN(const Value* const member, reader.find(key));
                SA_TRY_ASSIGN(const std::string* const text, Config::read_string(*member, named));
                SA_FAIL_IF(text->empty(), StatusCode::InvalidDescription,
                           "{} is empty; it states a non-empty text", named);
                field = *text;
                return Status();
            }

            /**
             * @brief Reads the whole-number member @p key of @p reader into @p field.
             * @return Success, or InvalidDescription for an absent key and a value that is no
             *         whole number.
             */
            Status read_number(const Config::SectionReader& reader, std::string_view key,
                               std::int64_t& field)
            {
                SA_TRY_ASSIGN(const Value* const member, reader.find(key));
                SA_TRY_ASSIGN(field, Config::read_whole(*member, reader.member(key)));
                return Status();
            }

            /**
             * @brief Reads "if_absent" and "wait_ms" of a replica into @p policy.
             * @return Success, or InvalidDescription for an absent "if_absent", one outside
             *         local, wait and fail, a wait without a positive "wait_ms", and a
             *         "wait_ms" beside local or fail.
             */
            Status read_absent_policy(const Config::SectionReader& reader,
                                      SushiRuntime::AbsentPolicy& policy)
            {
                std::string stated;
                SA_TRY(read_text(reader, K_IF_ABSENT_KEY, stated));

                const std::string if_absent = reader.member(K_IF_ABSENT_KEY);
                const std::string wait_ms = reader.member(K_WAIT_MS_KEY);
                if (stated == K_ABSENT_WAIT)
                {
                    std::int64_t bound = 0;
                    SA_FAIL_IF(!reader.has(K_WAIT_MS_KEY), StatusCode::InvalidDescription,
                               "{} is 'wait', which needs {} in milliseconds", if_absent,
                               wait_ms);
                    SA_TRY(read_number(reader, K_WAIT_MS_KEY, bound));
                    SA_FAIL_IF(bound <= 0, StatusCode::InvalidDescription,
                               "{} is {}; a wait is bounded by a positive time", wait_ms, bound);

                    policy.action = SushiRuntime::AbsentAction::Wait;
                    policy.bound = std::chrono::milliseconds(bound);
                    return Status();
                }

                SA_FAIL_IF(stated != K_ABSENT_LOCAL && stated != K_ABSENT_FAIL,
                           StatusCode::InvalidDescription,
                           "{} is '{}'; it is one of '{}', '{}', '{}'", if_absent, stated,
                           K_ABSENT_LOCAL, K_ABSENT_WAIT, K_ABSENT_FAIL);
                SA_FAIL_IF(reader.has(K_WAIT_MS_KEY), StatusCode::InvalidDescription,
                           "{} is stated beside {} '{}'; only 'wait' has a bound", wait_ms,
                           if_absent, stated);

                policy.action = stated == K_ABSENT_LOCAL ? SushiRuntime::AbsentAction::Local
                                                         : SushiRuntime::AbsentAction::Fail;
                policy.bound = std::chrono::milliseconds(0);
                return Status();
            }

            /**
             * @brief Reads one entry of "replicas", named @p where, into @p replica.
             * @return Success, or InvalidDescription naming the key of the first refusal.
             */
            Status read_replica(const Value& entry, std::string_view where, ReadReplica& replica)
            {
                const Config::SectionReader reader(entry, where);
                SA_TRY(reader.check_object());
                SA_TRY(reader.check_keys(span_of(K_REPLICA_KEYS)));
                SA_TRY(read_text(reader, K_WORKER_KEY, replica.worker));
                SA_TRY(read_number(reader, K_ROWS_KEY, replica.share.rows));
                SA_TRY(read_number(reader, K_MICRO_BATCH_KEY, replica.share.micro_batch));
                SA_TRY(read_absent_policy(reader, replica.if_absent));

                if (reader.has(K_PRIORITY_KEY))
                {
                    std::int64_t priority = 0;
                    SA_TRY(read_number(reader, K_PRIORITY_KEY, priority));
                    SA_FAIL_IF(priority < std::numeric_limits<int>::min() ||
                                   priority > std::numeric_limits<int>::max(),
                               StatusCode::InvalidDescription, "{} is {}; it fits in {} bits",
                               reader.member(K_PRIORITY_KEY), priority,
                               std::numeric_limits<int>::digits + 1);
                    replica.priority = static_cast<int>(priority);
                }

                return Status();
            }

            /** @brief Returns the path @p text names, resolved against @p base lexically. */
            std::filesystem::path resolve(const std::filesystem::path& base,
                                          const std::string& text)
            {
                const std::filesystem::path spelled(text);
                if (spelled.is_absolute())
                    return spelled;

                return base / spelled;
            }
        } // namespace

        Result<ClusterSection> ClusterSection::read(const Value& section,
                                                    const std::filesystem::path& base)
        {
            const Config::SectionReader reader(section, K_CLUSTER_KEY);
            SA_TRY(reader.check_object());
            SA_TRY(reader.check_keys(span_of(K_SECTION_KEYS)));

            std::string listen;
            SA_TRY(read_text(reader, K_LISTEN_KEY, listen));
            std::string secret;
            SA_TRY(read_text(reader, K_SECRET_FILE_KEY, secret));

            const std::string replicas_name = reader.member(K_REPLICAS_KEY);
            SA_TRY_ASSIGN(const Value* const listed, reader.find(K_REPLICAS_KEY));
            SA_TRY_ASSIGN(const std::vector<Value>* const entries,
                          Config::read_array(*listed, replicas_name));
            SA_FAIL_IF(entries->empty(), StatusCode::InvalidDescription,
                       "{} is empty; a cluster has at least one replica, the master",
                       replicas_name);

            std::vector<ReadReplica> read(entries->size());
            for (std::size_t index = 0; index < entries->size(); ++index)
            {
                const std::string where = Config::entry_of(replicas_name, index);
                SA_TRY(read_replica((*entries)[index], where, read[index]));

                for (std::size_t before = 0; before < index; ++before)
                {
                    SA_FAIL_IF(read[before].worker == read[index].worker,
                               StatusCode::InvalidDescription,
                               "{} is '{}', which entry {} already names; a machine runs one "
                               "replica",
                               Config::member_of(where, K_WORKER_KEY), read[index].worker, before);
                }
            }

            std::vector<ReplicaShare> shares;
            shares.reserve(read.size());
            for (const ReadReplica& replica : read)
                shares.push_back(replica.share);

            Result<ReplicaPlan> made = ReplicaPlan::make(std::move(shares));
            SA_TRY(SA_ANNOTATE(made.status(), "{}", replicas_name));
            ReplicaPlan plan = std::move(made).value();

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                std::vector<Replica> kept;
                kept.reserve(read.size());
                for (ReadReplica& replica : read)
                {
                    Replica one;
                    one.worker = std::move(replica.worker);
                    one.priority = replica.priority;
                    one.if_absent = replica.if_absent;
                    kept.push_back(std::move(one));
                }

                return ClusterSection(std::move(listen), resolve(base, secret), std::move(plan),
                                      std::move(kept));
            });
        }

        ClusterSection::ClusterSection(std::string listen_address,
                                       std::filesystem::path secret_file, ReplicaPlan plan,
                                       std::vector<Replica> replicas)
            : listen_address_(std::move(listen_address)), secret_file_(std::move(secret_file)),
              plan_(std::move(plan)), replicas_(std::move(replicas))
        {

        }

        const std::string& ClusterSection::listen_address() const noexcept
        {
            return listen_address_;
        }

        const std::filesystem::path& ClusterSection::secret_file() const noexcept
        {
            return secret_file_;
        }

        const ReplicaPlan& ClusterSection::plan() const noexcept
        {
            return plan_;
        }

        const std::string& ClusterSection::worker_of(std::size_t replica) const
        {
            SA_FATAL_IF(replica >= replicas_.size(),
                        "Replica {} was asked of a cluster section that has {}", replica,
                        replicas_.size());
            return replicas_[replica].worker;
        }

        const SushiRuntime::AbsentPolicy& ClusterSection::absent_policy_of(
            std::size_t replica) const
        {
            SA_FATAL_IF(replica >= replicas_.size(),
                        "Replica {} was asked of a cluster section that has {}", replica,
                        replicas_.size());
            return replicas_[replica].if_absent;
        }

        Result<SushiRuntime::PlacementRules> ClusterSection::placement_rules() const
        {
            SushiRuntime::PlacementRules rules;
            const Status built = SushiAI::Detail::guard(
                SushiAI::Detail::SeamPhase::Record, [&]() -> Status
            {
                for (std::size_t replica = 0; replica < replicas_.size(); ++replica)
                {
                    SushiRuntime::PlacementRule rule;
                    rule.name = "replica " + std::to_string(replica);
                    rule.match.placement_class = std::string(K_REPLICA_PLACEMENT_CLASS);
                    rule.match.key = static_cast<std::uint64_t>(replica);
                    rule.target.machine_name = replicas_[replica].worker;
                    rule.priority = replicas_[replica].priority;
                    rule.if_absent = replicas_[replica].if_absent;

                    const SushiRuntime::Result<void> added = rules.add(std::move(rule));
                    if (!added)
                    {
                        return SA_ANNOTATE(SushiAI::Detail::status_of_error(
                                               added.error(), SushiAI::Detail::SeamPhase::Record),
                                           "{} replica {}", K_CLUSTER_KEY, replica);
                    }
                }

                return Status();
            });
            SA_TRY(built);

            return Result<SushiRuntime::PlacementRules>(std::move(rules));
        }

    } // namespace Distribution
} // namespace SushiAI
