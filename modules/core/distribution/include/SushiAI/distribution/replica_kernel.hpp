/**************************************************************************/
/* replica_kernel.hpp                                                     */
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
 * @file replica_kernel.hpp
 * @brief Declares ReplicaKernel, the registered kernel that runs one replica of a step.
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/cluster/pack_layout.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/api/graph/node_options.hpp>
#include <SushiRuntime/core/operation_id.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/distributed/buffer_view.hpp>
#include <SushiRuntime/distributed/kernel_registry.hpp>
#include <SushiAI/distribution/replica_draws.hpp>
#include <SushiAI/distribution/replica_model.hpp>
#include <SushiAI/distribution/replica_plan.hpp>
#include <SushiAI/distribution/replica_step.hpp>
#include <SushiAI/distribution/replica_task.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        /**
         * @brief Runs one replica task: unpacks the parameters, runs ReplicaStep over the rows
         *        ReplicaPlan names, and packs the mean of its results.
         *
         * Its inputs are the parameter pack and the position pack; its output is the result
         * pack. It keeps one ReplicaStep per replica served and serves one task at a
         * time.
         */
        class ReplicaKernel final
        {
            public:
                /**
                 * @brief Builds the step of the replica @p draws names and fixes the operation.
                 * @param config The configuration of every step's runtime.
                 * @return The kernel, or the failure of the step; InvalidDescription for a
                 *         replica the plan does not hold.
                 */
                [[nodiscard]] static Result<std::unique_ptr<ReplicaKernel>> make(
                    ReplicaPlan plan, ReplicaModelFactory factory, const ReplicaDraws& draws,
                    const SushiRuntime::RuntimeConfig& config = replica_runtime_config());

                ReplicaKernel(const ReplicaKernel&) = delete;
                ReplicaKernel& operator=(const ReplicaKernel&) = delete;
                ReplicaKernel(ReplicaKernel&&) = delete;
                ReplicaKernel& operator=(ReplicaKernel&&) = delete;
                ~ReplicaKernel() = default;

                /** @brief Returns "sushiai.replica:<fingerprint>", the name it registers under. */
                [[nodiscard]] const std::string& operation_name() const noexcept
                {
                    return operation_name_;
                }

                /** @brief Returns the OperationID of operation_name(). */
                [[nodiscard]] SushiRuntime::OperationID operation() const noexcept
                {
                    return operation_;
                }

                /** @brief Returns the layout of the parameter pack the task reads. */
                [[nodiscard]] const SushiBLAS::PackLayout& parameter_layout() const noexcept;

                /** @brief Returns the layout of the result pack the task writes. */
                [[nodiscard]] const SushiBLAS::PackLayout& result_layout() const noexcept;

                /**
                 * @brief Registers the kernel under operation() in @p registry.
                 * @pre This object outlives @p registry's use of the kernel, and @p registry is
                 *      not sealed.
                 */
                void register_with(SushiRuntime::Distributed::KernelRegistry& registry);

                /**
                 * @brief Runs the task a node carries over its views, on the calling thread.
                 * @return Success, or InvalidDescription for parameters ReplicaTask refuses, a
                 *         replica count other than the plan's, a pack count other than two in
                 *         and one out, or a view the layouts or ReplicaPosition refuse;
                 *         Incompatible for a step whose fingerprint differs from the first;
                 *         or the failure of a step.
                 */
                [[nodiscard]] Status serve(
                    SushiRuntime::span<const SushiRuntime::Distributed::BufferView> inputs,
                    SushiRuntime::span<const SushiRuntime::Distributed::BufferView> outputs,
                    const SushiRuntime::NodeOptions& options);

            private:
                /** @brief Takes the plan, the factory and the first step. */
                ReplicaKernel(ReplicaPlan plan, ReplicaModelFactory factory,
                              SushiRuntime::RuntimeConfig config,
                              std::unique_ptr<ReplicaStep> first);

                /**
                 * @brief Returns the step of the replica @p draws names, built on first use.
                 * @return The step, or the failure of its build; Incompatible for a model whose
                 *         fingerprint differs from the first step's.
                 */
                [[nodiscard]] Result<ReplicaStep*> step_for(const ReplicaDraws& draws);

                /** @brief Runs one task with the lock held. */
                [[nodiscard]] Status serve_locked(
                    SushiRuntime::span<const SushiRuntime::Distributed::BufferView> inputs,
                    SushiRuntime::span<const SushiRuntime::Distributed::BufferView> outputs,
                    const ReplicaTask& task);

                /// @brief Which rows each replica takes of each batch.
                ReplicaPlan plan_;

                /// @brief Builds the model of a replica's step.
                ReplicaModelFactory factory_;

                /// @brief The configuration of each step's runtime.
                SushiRuntime::RuntimeConfig config_;

                /// @brief Entry k holds the step of replica k, or null before its first task.
                std::vector<std::unique_ptr<ReplicaStep>> steps_;

                /// @brief The hash every step's model and layouts must have.
                std::uint64_t fingerprint_ = 0;

                /// @brief The step whose layouts the accessors return; one of steps_.
                const ReplicaStep* reference_ = nullptr;

                /// @brief The name the kernel registers under.
                std::string operation_name_;

                /// @brief The OperationID of operation_name_.
                SushiRuntime::OperationID operation_;

                /// @brief Serialises tasks, which share the steps and the device.
                std::mutex serving_;
        };

    } // namespace Distribution
} // namespace SushiAI
