/**************************************************************************/
/* step_recorder.hpp                                                      */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

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
