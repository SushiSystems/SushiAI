/**************************************************************************/
/* host_execution_backend.cpp                                             */
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
 * @file host_execution_backend.cpp
 * @brief Implements HostExecutionBackend and the private StepTasks that keeps its tasks in a
 *        HostTaskSink.
 * @author Mustafa Garip
 */

#include <SushiAI/execution/host_execution_backend.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <SushiBLAS/engine/host_task_sink.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace
    {
        /**
         * @class HostStepTasks
         * @brief Keeps the tasks of one recording in a HostTaskSink and replays them on the host.
         */
        class HostStepTasks final : public StepTasks
        {
            public:
                /**
                 * @brief Binds a recorder over a new sink to the scratch pool and mode of @p engine.
                 * @param engine Must outlive this object.
                 * @throws SushiBLAS::UnsupportedExecution when the mode of @p engine is not
                 *         ForceReference.
                 */
                explicit HostStepTasks(SushiBLAS::Engine& engine)
                    : recorder_(sink_, &engine.scratch_pool(), nullptr, engine.mode())
                {

                }

                HostStepTasks(const HostStepTasks&) = delete;
                HostStepTasks& operator=(const HostStepTasks&) = delete;
                HostStepTasks(HostStepTasks&&) = delete;
                HostStepTasks& operator=(HostStepTasks&&) = delete;
                ~HostStepTasks() override = default;

                /** @brief Returns the recorder over the sink. */
                SushiBLAS::TaskRecorder& recorder() noexcept override
                {
                    return recorder_;
                }

                /** @brief Returns null: the host recording has no SushiRuntime graph. */
                SushiRuntime::Recorder* runtime_recorder() noexcept override
                {
                    return nullptr;
                }

                /** @brief Returns how many tasks the sink keeps. */
                std::size_t task_count() const noexcept override
                {
                    return sink_.recorded_count();
                }

                /** @brief Returns the origin of each task the sink keeps. */
                std::vector<std::string> task_origins() const override
                {
                    std::vector<std::string> origins;
                    for (const SushiBLAS::HostTaskRecord& record : sink_.records())
                        origins.push_back(record.origin);
                    return origins;
                }

                /** @brief Returns success: the sink throws at record time and keeps no refusal. */
                Status refusal() const override
                {
                    return Status();
                }

                /** @brief Stores @p hook after the hooks stored before it. */
                Status before_each_replay(std::function<void()> hook) override
                {
                    return Detail::guard(Detail::SeamPhase::Allocate, [&]()
                    {
                        hooks_.push_back(std::move(hook));
                        return Status();
                    });
                }

                /** @brief Returns success: a host recording has no report to name. */
                Status set_label(std::string) override
                {
                    return Status();
                }

                /** @brief Returns success: the sink replays its tasks as recorded. */
                Status seal() noexcept override
                {
                    return Status();
                }

                /** @brief Runs the hooks, then the sink's tasks; leaves @p report as it was. */
                Status replay(SushiRuntime::RunReport&) noexcept override
                {
                    return Detail::guard(Detail::SeamPhase::Execute, [&]()
                    {
                        if (task_count() == 0)
                            return Status();

                        for (const std::function<void()>& hook : hooks_)
                            hook();

                        const std::optional<SushiBLAS::HostTaskFailure> failed = sink_.replay();
                        if (!failed.has_value())
                            return Status();

                        return Status::from_message(StatusCode::DeviceFailed, [&]()
                        {
                            return "task " + std::to_string(failed->index) + " ("
                                 + failed->name + ") failed: " + failed->message;
                        });
                    });
                }

            private:
                /// @brief Keeps the tasks in record order; declared before recorder_, which uses it.
                SushiBLAS::HostTaskSink sink_;

                /// @brief The recorder over sink_.
                SushiBLAS::TaskRecorder recorder_;

                /// @brief The hooks run before each replay, in the order added.
                std::vector<std::function<void()>> hooks_;
        };
    } // namespace

    HostExecutionBackend::HostExecutionBackend(SushiBLAS::Core::Layout layout)
        : engine_(SushiBLAS::Engine::Host{}, layout)
    {

    }

    SushiBLAS::Engine& HostExecutionBackend::engine() noexcept
    {
        return engine_;
    }

    const SushiBLAS::Engine& HostExecutionBackend::engine() const noexcept
    {
        return engine_;
    }

    Result<std::unique_ptr<StepTasks>> HostExecutionBackend::make_tasks() noexcept
    {
        return Detail::guard_value(Detail::SeamPhase::Allocate, [this]() -> std::unique_ptr<StepTasks>
        {
            return std::make_unique<HostStepTasks>(engine_);
        });
    }

    bool HostExecutionBackend::supports_fp64() const noexcept
    {
        return true;
    }
} // namespace SushiAI
