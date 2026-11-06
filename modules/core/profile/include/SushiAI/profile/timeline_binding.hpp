/**************************************************************************/
/* timeline_binding.hpp                                                   */
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
 * @file timeline_binding.hpp
 * @brief Declares TimelineBinding, which points every Timeline at a session's timeline of the same name.
 * @author Mustafa Garip
 * @see modules/core/profile/README.md
 */

#pragma once

namespace SushiProfiler
{
    class Session;
} // namespace SushiProfiler

namespace SushiAI
{
    namespace Profile
    {
        /**
         * @brief Points every Timeline at the timeline of one session, for as long as it lives.
         *
         * The binding is process-wide, as the session is; a second binding while one lives is a
         * programming error. Binding and unbinding happen on the thread that runs the step
         * loop, with no mark in flight.
         */
        class TimelineBinding final
        {
            public:
                /**
                 * @brief Points every Timeline at the timeline of @p session with its name.
                 * @pre @p session outlives the binding; no other binding lives.
                 */
                explicit TimelineBinding(SushiProfiler::Session& session);

                TimelineBinding(const TimelineBinding&) = delete;
                TimelineBinding& operator=(const TimelineBinding&) = delete;
                TimelineBinding(TimelineBinding&&) = delete;
                TimelineBinding& operator=(TimelineBinding&&) = delete;

                /** @brief Unbinds every Timeline; mark_period does nothing afterwards. */
                ~TimelineBinding();
        };
    } // namespace Profile
} // namespace SushiAI
