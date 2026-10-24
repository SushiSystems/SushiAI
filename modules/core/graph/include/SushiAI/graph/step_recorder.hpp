/**************************************************************************/
/* step_recorder.hpp                                                      */
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
 * @file step_recorder.hpp
 * @brief Declares the StepRecorder interface and the list that drives several as one.
 * @author Mustafa Garip
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/step_recording.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @class StepRecorder
         * @brief Records device work into a compiled step and follows each of its replays.
         */
        class StepRecorder
        {
            public:
                virtual ~StepRecorder() = default;

                /**
                 * @brief Records this recorder's work through @p recording and returns the
                 *        task count.
                 */
                [[nodiscard]] virtual Result<std::size_t> record_into_step(
                    const StepRecording& recording) = 0;

                /**
                 * @brief Refreshes the host state that varies between replays.
                 *
                 * Runs before the plan is dispatched and records no device work.
                 */
                virtual void before_replay() {}

                /** @brief Runs after the replay has completed and been waited on. */
                virtual void after_replay() {}
        };

        /**
         * @brief Runs several recorders as one, in the order they were added.
         *
         * Holds each member by address and owns none of them.
         */
        class StepRecorderList final : public StepRecorder
        {
            public:
                /**
                 * @brief Appends @p recorder behind the members already held.
                 * @param recorder Must outlive the list, and is neither the list nor a direct
                 *                 member. A recorder reached through a nested list is not checked.
                 * @return Success, or StatusCode::OutOfMemory when the host allocation fails.
                 * @pre No list of one tree holds a recorder twice, and no list reaches itself.
                 */
                [[nodiscard]] Status add(StepRecorder& recorder)
                {
                    SA_FATAL_IF(&recorder == this,
                                "StepRecorderList::add: a list cannot hold itself");
                    SA_FATAL_IF(std::find(members_.begin(), members_.end(), &recorder)
                                    != members_.end(),
                                "StepRecorderList::add: the recorder is already one of the {} "
                                "members",
                                members_.size());

                    return ::SushiAI::Detail::guard(
                        ::SushiAI::Detail::SeamPhase::Allocate, [&]()
                    {
                        members_.push_back(&recorder);
                        return Status();
                    });
                }

                /** @brief Returns whether the list holds no member. */
                [[nodiscard]] bool empty() const noexcept
                {
                    return members_.empty();
                }

                /**
                 * @brief Records every member's work into the step, first added first.
                 * @return Sum of the device tasks the members recorded.
                 */
                [[nodiscard]] Result<std::size_t> record_into_step(
                    const StepRecording& recording) override
                {
                    std::size_t recorded = 0;
                    for (StepRecorder* member : members_)
                    {
                        SA_TRY_ASSIGN(const std::size_t count, member->record_into_step(recording));
                        recorded += count;
                    }

                    return recorded;
                }

                /** @brief Refreshes every member's host state, first added first. */
                void before_replay() override
                {
                    for (StepRecorder* member : members_)
                        member->before_replay();
                }

                /** @brief Notifies every member that the replay finished, first added first. */
                void after_replay() override
                {
                    for (StepRecorder* member : members_)
                        member->after_replay();
                }

            private:
                /// @brief Members in the order add() received them; none is owned.
                std::vector<StepRecorder*> members_;
        };

    } // namespace Graph
} // namespace SushiAI
