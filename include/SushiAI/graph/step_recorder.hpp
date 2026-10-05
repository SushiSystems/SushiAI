/****************************************************************/
/* step_recorder.hpp                                            */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file step_recorder.hpp
 * @brief Declares the IStepRecorder interface for joining compiled steps.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>

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

    } // namespace Graph
} // namespace SushiAI
