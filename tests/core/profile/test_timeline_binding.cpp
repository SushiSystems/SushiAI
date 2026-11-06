/**************************************************************************/
/* test_timeline_binding.cpp                                              */
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
 * @file test_timeline_binding.cpp
 * @brief Tests that a TimelineBinding unbinds when it ends and that a later one binds again.
 * @author Mustafa Garip
 * @see tests/core/profile/test_timeline_binding.md
 */

#include <SushiAI/profile/timeline_binding.hpp>

#include <gtest/gtest.h>
#include <SushiProfiler/registry/registry.hpp>
#include <SushiProfiler/session/session.hpp>

#include <SushiAI/profile/timelines.hpp>
#include <SushiAI/profile_records/profile_records.hpp>

namespace
{
    using SushiAI::Profile::mark_period;
    using SushiAI::Profile::Timeline;
    using SushiAI::Profile::TimelineBinding;

    /** @brief Returns a configuration that follows `train.step` for four periods. */
    SushiProfiler::ProfilerConfig following_train_step()
    {
        return SushiAI::Test::following_timeline("train.step", 0U, 4U);
    }
} // namespace

/** @brief Tests that a mark after a TimelineBinding has ended changes no timeline. */
TEST(Medium_TimelineBinding, MarksStopWhenTheBindingEnds)
{
    const SushiProfiler::Registry registry;
    SushiProfiler::SessionStart started =
        SushiProfiler::Session::start(following_train_step(), registry);
    ASSERT_TRUE(started.ok()) << started.error;
    SushiProfiler::Session& session = *started.session;
    SushiProfiler::PeriodTimeline& step = session.timeline("train.step");
    {
        const TimelineBinding binding(session);
        mark_period(Timeline::TrainStep);
    }
    EXPECT_EQ(step.current(), 0U);

    mark_period(Timeline::TrainStep);
    EXPECT_EQ(step.current(), 0U);
    static_cast<void>(session.stop());
}

/** @brief Tests that a TimelineBinding made after another one ended marks the next period. */
TEST(Medium_TimelineBinding, ALaterBindingBindsAgain)
{
    const SushiProfiler::Registry registry;
    SushiProfiler::SessionStart started =
        SushiProfiler::Session::start(following_train_step(), registry);
    ASSERT_TRUE(started.ok()) << started.error;
    SushiProfiler::Session& session = *started.session;
    SushiProfiler::PeriodTimeline& step = session.timeline("train.step");
    {
        const TimelineBinding first(session);
        mark_period(Timeline::TrainStep);
    }
    {
        const TimelineBinding second(session);
        mark_period(Timeline::TrainStep);
    }

    EXPECT_EQ(step.current(), 1U);
    static_cast<void>(session.stop());
}
