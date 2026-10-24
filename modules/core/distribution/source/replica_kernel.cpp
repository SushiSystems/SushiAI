/**************************************************************************/
/* replica_kernel.cpp                                                     */
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
 * @file replica_kernel.cpp
 * @brief Implements ReplicaKernel: the task's checks, its steps and its registration.
 * @author Mustafa Garip
 */

#include <SushiAI/distribution/replica_kernel.hpp>

#include <stdexcept>
#include <utility>
#include <SushiBLAS/cluster/bound_pack.hpp>
#include <SushiBLAS/cluster/remote_task.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        Result<std::unique_ptr<ReplicaKernel>> ReplicaKernel::make(
            ReplicaPlan plan, ReplicaModelFactory factory, const ReplicaDraws& draws,
            const SushiRuntime::RuntimeConfig& config)
        {
            SA_FAIL_IF(draws.replica_index >= plan.replica_count(),
                       StatusCode::InvalidDescription,
                       "A replica kernel was asked for replica {} of a plan of {}",
                       draws.replica_index, plan.replica_count());

            const std::int64_t micro_batch =
                plan.micro_batch_of(static_cast<std::size_t>(draws.replica_index));
            SA_TRY_ASSIGN(std::unique_ptr<ReplicaStep> first,
                          ReplicaStep::make(factory, draws, micro_batch, config));

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                return std::unique_ptr<ReplicaKernel>(new ReplicaKernel(
                    std::move(plan), std::move(factory), config, std::move(first)));
            });
        }

        ReplicaKernel::ReplicaKernel(ReplicaPlan plan, ReplicaModelFactory factory,
                                     SushiRuntime::RuntimeConfig config,
                                     std::unique_ptr<ReplicaStep> first)
            : plan_(std::move(plan)), factory_(std::move(factory)), config_(std::move(config)),
              steps_(plan_.replica_count()), fingerprint_(first->fingerprint()),
              reference_(first.get()),
              operation_name_(SushiBLAS::remote_operation_name(
                  std::string(K_REPLICA_PLACEMENT_CLASS), fingerprint_)),
              operation_(SushiBLAS::remote_operation(std::string(K_REPLICA_PLACEMENT_CLASS),
                                                     fingerprint_))
        {
            const std::size_t replica = static_cast<std::size_t>(first->draws().replica_index);
            steps_[replica] = std::move(first);
        }

        const SushiBLAS::PackLayout& ReplicaKernel::parameter_layout() const noexcept
        {
            return reference_->parameter_layout();
        }

        const SushiBLAS::PackLayout& ReplicaKernel::result_layout() const noexcept
        {
            return reference_->result_layout();
        }

        void ReplicaKernel::register_with(SushiRuntime::Distributed::KernelRegistry& registry)
        {
            registry.register_kernel(
                operation_,
                [this](sycl::queue&,
                       SushiRuntime::span<const SushiRuntime::Distributed::BufferView> inputs,
                       SushiRuntime::span<const SushiRuntime::Distributed::BufferView> outputs,
                       const SushiRuntime::NodeOptions& options) -> sycl::event
            {
                // The registry's contract reports a failed task by a throw.
                const Status served = serve(inputs, outputs, options);
                if (!served.is_ok())
                    throw std::runtime_error(std::string(served.message()));

                return sycl::event{};
            });
        }

        Status ReplicaKernel::serve(
            SushiRuntime::span<const SushiRuntime::Distributed::BufferView> inputs,
            SushiRuntime::span<const SushiRuntime::Distributed::BufferView> outputs,
            const SushiRuntime::NodeOptions& options)
        {
            SA_TRY_ASSIGN(const ReplicaTask task, ReplicaTask::decode(options.parameters));
            SA_FAIL_IF(task.replica_count != plan_.replica_count(),
                       StatusCode::InvalidDescription,
                       "A replica node names a step of {} replicas; this kernel's plan has {}",
                       task.replica_count, plan_.replica_count());

            const std::lock_guard<std::mutex> lock(serving_);
            return serve_locked(inputs, outputs, task);
        }

        Result<ReplicaStep*> ReplicaKernel::step_for(const ReplicaDraws& draws)
        {
            std::unique_ptr<ReplicaStep>& slot =
                steps_[static_cast<std::size_t>(draws.replica_index)];
            if (slot != nullptr && slot->draws() == draws)
                return slot.get();

            const std::int64_t micro_batch =
                plan_.micro_batch_of(static_cast<std::size_t>(draws.replica_index));
            SA_TRY_ASSIGN(std::unique_ptr<ReplicaStep> built,
                          ReplicaStep::make(factory_, draws, micro_batch, config_));
            SA_FAIL_IF(built->fingerprint() != fingerprint_, StatusCode::Incompatible,
                       "Replica {}'s model and packs hash to {}; the kernel serves {}",
                       draws.replica_index, built->fingerprint(), fingerprint_);

            // The accessors read reference_, which must not outlive the step it names.
            if (slot.get() == reference_)
                reference_ = built.get();
            slot = std::move(built);
            return slot.get();
        }

        Status ReplicaKernel::serve_locked(
            SushiRuntime::span<const SushiRuntime::Distributed::BufferView> inputs,
            SushiRuntime::span<const SushiRuntime::Distributed::BufferView> outputs,
            const ReplicaTask& task)
        {
            SA_TRY_ASSIGN(ReplicaStep* const step, step_for(task.draws()));

            SA_FAIL_IF(inputs.size() != 1 || outputs.size() != 1,
                       StatusCode::InvalidDescription,
                       "A replica node takes one parameter pack and writes one result pack; it "
                       "names {} inputs and {} outputs",
                       inputs.size(), outputs.size());
            SA_TRY_ASSIGN(const std::vector<SushiBLAS::BoundPack> parameters,
                          SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Execute,
                                                       [&]()
            {
                const SushiRuntime::span<const SushiBLAS::PackLayout> layouts(
                    &step->parameter_layout(), 1);
                return SushiBLAS::bind_views(inputs, layouts);
            }));
            SA_TRY_ASSIGN(const std::vector<SushiBLAS::BoundPack> results,
                          SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Execute,
                                                       [&]()
            {
                const SushiRuntime::span<const SushiBLAS::PackLayout> layouts(
                    &step->result_layout(), 1);
                return SushiBLAS::bind_views(outputs, layouts);
            }));

            SA_TRY(step->begin(parameters.front()));

            const std::size_t replica = static_cast<std::size_t>(task.replica_index);
            const std::uint64_t micro_batches =
                static_cast<std::uint64_t>(plan_.micro_batch_count_of(replica));
            for (std::uint64_t index = 0; index < micro_batches; ++index)
            {
                ReplicaRows rows;
                rows.epoch = task.epoch;
                rows.batch_index = task.batch_index;
                rows.micro_batch_index = index;
                rows.rows = plan_.micro_batch(replica, task.batch_index, index);
                rows.draws = task.draws();
                SA_TRY(SA_ANNOTATE(step->run_micro_batch(rows),
                                   "replica {}, batch {}, micro-batch {}: ", replica,
                                   task.batch_index, index));
            }

            return step->finish(results.front());
        }

    } // namespace Distribution
} // namespace SushiAI
