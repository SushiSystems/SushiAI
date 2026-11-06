/**************************************************************************/
/* in_memory_step_recorder.hpp                                            */
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
 * @file in_memory_step_recorder.hpp
 * @brief Declares InMemoryStepRecorder, a StepRecorder that records no device work.
 * @author Mustafa Garip
 * @see modules/core/graph/README.md
 */

#pragma once

#include <cstddef>
#include <utility>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/graph/step_recording.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @class InMemoryStepRecorder
         * @brief Journals the calls it receives and answers record_into_step() with a scripted
         *        task count or a scripted failure; touches no device, file, thread or clock.
         *
         * It never reads the StepRecording it is given, so it records no task into the step.
         */
        class InMemoryStepRecorder final : public StepRecorder
        {
            public:
                /** @brief Names one call the recorder received. */
                enum class Call
                {
                    Record,       /**< record_into_step() was called. */
                    BeforeReplay, /**< before_replay() was called. */
                    AfterReplay   /**< after_replay() was called. */
                };

                /**
                 * @brief Builds a recorder whose record_into_step() reports @p tasks tasks.
                 * @param tasks The count record_into_step() returns until fail_with() is called.
                 */
                explicit InMemoryStepRecorder(std::size_t tasks = 0) noexcept : tasks_(tasks)
                {

                }

                /**
                 * @brief Makes every later record_into_step() return @p failure.
                 * @param failure A failure; a success restores the task count.
                 */
                void fail_with(Status failure) noexcept
                {
                    failure_ = std::move(failure);
                }

                /** @brief Appends Call::Record and returns the scripted count or failure. */
                [[nodiscard]] Result<std::size_t> record_into_step(const StepRecording&) override
                {
                    calls_.push_back(Call::Record);
                    if (!failure_.is_ok())
                        return failure_;

                    return tasks_;
                }

                /** @brief Appends Call::BeforeReplay. */
                void before_replay() override
                {
                    calls_.push_back(Call::BeforeReplay);
                }

                /** @brief Appends Call::AfterReplay. */
                void after_replay() override
                {
                    calls_.push_back(Call::AfterReplay);
                }

                /** @brief Returns every call received, first received first. */
                [[nodiscard]] const std::vector<Call>& calls() const noexcept
                {
                    return calls_;
                }

            private:
                /// @brief Holds the count a successful record_into_step() returns.
                std::size_t tasks_;

                /// @brief Holds the scripted failure; a success when none is scripted.
                Status failure_;

                /// @brief Holds the calls received, in order.
                std::vector<Call> calls_;
        };

    } // namespace Graph
} // namespace SushiAI
