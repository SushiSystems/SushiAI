/**************************************************************************/
/* runtime_execution_backend.cpp                                          */
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
 * @file runtime_execution_backend.cpp
 * @brief Implements RuntimeExecutionBackend and the private StepTasks that records on a
 *        SushiRuntime graph.
 * @author Mustafa Garip
 */

#include <SushiAI/execution/runtime_execution_backend.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <SushiBLAS/engine/late_count_hold.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace
    {
        /**
         * @class RuntimeStepTasks
         * @brief Records the tasks of one step on a SushiRuntime graph and replays that graph.
         */
        class RuntimeStepTasks final : public StepTasks
        {
            public:
                /**
                 * @brief Makes a graph of @p runtime and a recorder over it.
                 * @param runtime Must outlive this object.
                 * @param engine Built on @p runtime; must outlive this object.
                 */
                RuntimeStepTasks(SushiRuntime::Runtime& runtime, SushiBLAS::Engine& engine)
                    : late_counts_(runtime), graph_(runtime.graph()),
                      recorder_(graph_.recorder(), engine.scratch_pool(), late_counts_,
                                engine.mode())
                {

                }

                RuntimeStepTasks(const RuntimeStepTasks&) = delete;
                RuntimeStepTasks& operator=(const RuntimeStepTasks&) = delete;
                RuntimeStepTasks(RuntimeStepTasks&&) = delete;
                RuntimeStepTasks& operator=(RuntimeStepTasks&&) = delete;
                ~RuntimeStepTasks() override = default;

                /** @brief Returns the recorder over the graph. */
                SushiBLAS::TaskRecorder& recorder() noexcept override
                {
                    return recorder_;
                }

                /** @brief Returns the recorder of the graph. */
                SushiRuntime::Recorder* runtime_recorder() noexcept override
                {
                    return &graph_.recorder();
                }

                /** @brief Returns how many nodes the graph holds. */
                std::size_t task_count() const noexcept override
                {
                    return graph_.recorder().node_count();
                }

                /** @brief Returns the refusal the graph's recorder kept, or success. */
                Status refusal() const override
                {
                    const SushiRuntime::Error* refused = graph_.recorder().refusal();
                    if (refused == nullptr)
                        return Status();

                    return Detail::status_of_error(*refused, Detail::SeamPhase::Record);
                }

                /** @brief Hands @p hook to the graph, which calls it before each step. */
                Status before_each_replay(std::function<void()> hook) override
                {
                    return Detail::guard(Detail::SeamPhase::Allocate, [&]()
                    {
                        graph_.before_each_step([work = std::move(hook)](std::size_t)
                        {
                            work();
                        });
                        return Status();
                    });
                }

                /** @brief Names the graph. */
                Status set_label(std::string label) override
                {
                    return Detail::guard(Detail::SeamPhase::Allocate, [&]()
                    {
                        graph_.set_label(std::move(label));
                        return Status();
                    });
                }

                /** @brief Compiles the graph; a graph with no node is left uncompiled. */
                Status seal() noexcept override
                {
                    return Detail::guard(Detail::SeamPhase::Execute, [&]()
                    {
                        if (task_count() == 0)
                            return Status();

                        const SushiRuntime::Result<void> compiled = graph_.compile();
                        if (!compiled)
                            return Detail::status_of_error(compiled.error(),
                                                           Detail::SeamPhase::Execute);

                        return Status();
                    });
                }

                /** @brief Runs the graph once and waits; a graph with no node runs as nothing. */
                Status replay(SushiRuntime::RunReport& report) noexcept override
                {
                    return Detail::guard(Detail::SeamPhase::Execute, [&]()
                    {
                        if (task_count() == 0)
                            return Status();

                        const SushiRuntime::Result<SushiRuntime::RunReport*> ran =
                            graph_.run(report);
                        if (!ran)
                            return Detail::status_of_error(ran.error(),
                                                           Detail::SeamPhase::Execute);

                        return Status();
                    });
                }

            private:
                /// @brief Keeps the late-count tasks of graph_; declared first so it dies last.
                SushiBLAS::LateCountHold late_counts_;

                /// @brief The graph the tasks are recorded on and replayed from.
                SushiRuntime::Graph graph_;

                /// @brief The recorder over the recorder of graph_.
                SushiBLAS::TaskRecorder recorder_;
        };
    } // namespace

    Result<std::unique_ptr<StepTasks>> RuntimeExecutionBackend::make_tasks() noexcept
    {
        return Detail::guard_value(Detail::SeamPhase::Allocate, [this]() -> std::unique_ptr<StepTasks>
        {
            return std::make_unique<RuntimeStepTasks>(*runtime_, *engine_);
        });
    }

    bool RuntimeExecutionBackend::supports_fp64() const noexcept
    {
        return SushiBLAS::Device::profile_for(
                   runtime_->advanced().queue(SushiRuntime::DeviceIndex{0})).supports_fp64;
    }
} // namespace SushiAI
