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
 * @brief Declares the IStepRecorder interface and the list that drives several as one.
 * @author Mustafa Garip
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>
#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @class IStepRecorder
         * @brief Device work recorded into the compiled step rather than flushed alone.
         *
         * Records replay-invariant tasks into the graph builder before compilation.
         */
        class IStepRecorder
        {
            public:
                virtual ~IStepRecorder() = default;

                /**
                 * @brief Records this work into the engine's task-graph builder.
                 *
                 * Called once before the step's single compile call.
                 *
                 * @return Count of recorded device tasks.
                 */
                virtual std::size_t record_into_step() = 0;

                /**
                 * @brief Refreshes state varying between replays on host.
                 *
                 * Executes before plan dispatch without recording device work.
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
        class StepRecorderList final : public IStepRecorder
        {
            public:
                /**
                 * @brief Appends @p recorder behind the members already held.
                 * @param recorder Must outlive the list, and is neither the list nor a direct
                 *                 member. A recorder reached through a nested list is not checked.
                 * @pre No list of one tree holds a recorder twice, and no list reaches itself.
                 */
                void add(IStepRecorder& recorder)
                {
                    SA_THROW_IF(&recorder == this,
                                "StepRecorderList::add: a list cannot hold itself");
                    SA_THROW_IF(std::find(members_.begin(), members_.end(), &recorder)
                                    != members_.end(),
                                "StepRecorderList::add: the recorder is already one of the {} "
                                "members",
                                members_.size());

                    members_.push_back(&recorder);
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
                std::size_t record_into_step() override
                {
                    std::size_t recorded = 0;
                    for (IStepRecorder* member : members_)
                        recorded += member->record_into_step();

                    return recorded;
                }

                /** @brief Refreshes every member's host state, first added first. */
                void before_replay() override
                {
                    for (IStepRecorder* member : members_)
                        member->before_replay();
                }

                /** @brief Notifies every member that the replay finished, first added first. */
                void after_replay() override
                {
                    for (IStepRecorder* member : members_)
                        member->after_replay();
                }

            private:
                /// @brief Members in the order add() received them; none is owned.
                std::vector<IStepRecorder*> members_;
        };

    } // namespace Graph
} // namespace SushiAI
