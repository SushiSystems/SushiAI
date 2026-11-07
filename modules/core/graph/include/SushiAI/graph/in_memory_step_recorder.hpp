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
         *        task count or failure; touches no device, file, thread or clock.
         *
         * It never reads its StepRecording. Members built over one Journal also append to it,
         * so the order of the calls across members can be read.
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

                /** @brief Names one call and the member that received it. */
                struct Entry
                {
                    int member = 0; /**< The number the member was built with. */
                    Call call = Call::Record; /**< The call it received. */

                    bool operator==(const Entry& other) const noexcept
                    {
                        return member == other.member && call == other.call;
                    }
                };

                /** @brief Holds the calls several members received, in the order they came. */
                using Journal = std::vector<Entry>;

                /**
                 * @brief Builds a recorder whose record_into_step() reports @p tasks tasks.
                 * @param tasks The count record_into_step() returns until fail_with() is called.
                 */
                explicit InMemoryStepRecorder(std::size_t tasks = 0) noexcept : tasks_(tasks)
                {

                }

                /**
                 * @brief Builds a recorder that also appends each call to @p journal.
                 * @param member The number its entries carry.
                 * @param journal Shared by the members of one test; must outlive the recorder.
                 * @param tasks The count record_into_step() returns until fail_with() is called.
                 */
                InMemoryStepRecorder(int member, Journal& journal, std::size_t tasks = 0) noexcept
                    : tasks_(tasks), member_(member), journal_(&journal)
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
                    note(Call::Record);
                    if (!failure_.is_ok())
                        return failure_;

                    return tasks_;
                }

                /** @brief Appends Call::BeforeReplay. */
                void before_replay() override
                {
                    note(Call::BeforeReplay);
                }

                /** @brief Appends Call::AfterReplay. */
                void after_replay() override
                {
                    note(Call::AfterReplay);
                }

                /** @brief Returns every call received, first received first. */
                [[nodiscard]] const std::vector<Call>& calls() const noexcept
                {
                    return calls_;
                }

            private:
                /** @brief Appends @p call to the calls received and to the shared journal. */
                void note(Call call)
                {
                    calls_.push_back(call);
                    if (journal_ != nullptr)
                        journal_->push_back(Entry{member_, call});
                }

                /// @brief Holds the count a successful record_into_step() returns.
                std::size_t tasks_;

                /// @brief Holds the scripted failure; a success when none is scripted.
                Status failure_;

                /// @brief Holds the number the entries of the shared journal carry.
                int member_ = 0;

                /// @brief Holds the journal several members share, or null for none.
                Journal* journal_ = nullptr;

                /// @brief Holds the calls received, in order.
                std::vector<Call> calls_;
        };

    } // namespace Graph
} // namespace SushiAI
