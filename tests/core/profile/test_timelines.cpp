/**************************************************************************/
/* test_timelines.cpp                                                     */
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
 * @file test_timelines.cpp
 * @brief Tests the names of the timelines and mark_period with and without a bound session.
 * @author Mustafa Garip
 * @see tests/core/profile/test_timelines.md
 */

#include <SushiAI/profile/timelines.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <SushiProfiler/periods/period_timeline.hpp>
#include <SushiProfiler/registry/registry.hpp>
#include <SushiProfiler/session/session.hpp>

#include <SushiAI/profile/timeline_binding.hpp>
#include <SushiAI/profile_records/profile_records.hpp>

namespace
{
    using SushiAI::Profile::mark_period;
    using SushiAI::Profile::Timeline;
    using SushiAI::Profile::TimelineBinding;

    /** @brief Returns a configuration that follows `train.step` for one warm-up and three periods. */
    SushiProfiler::ProfilerConfig train_step_window()
    {
        return SushiAI::Test::following_timeline("train.step", 1U, 3U);
    }

    /** @brief Marks the train step @p count times in the binding in force. */
    void mark_train_steps(std::uint32_t count)
    {
        for (std::uint32_t index = 0U; index < count; ++index)
            mark_period(Timeline::TrainStep);
    }

    /** @brief Returns the index of each period of @p record, in the order the record holds them. */
    std::vector<std::size_t> period_indices(const SushiProfiler::RunRecord& record)
    {
        std::vector<std::size_t> indices;
        for (const SushiProfiler::PeriodRecord& period : record.periods)
            indices.push_back(static_cast<std::size_t>(period.index));
        return indices;
    }
} // namespace

TEST(Small_Timelines, MarkWithNoBindingDoesNothing)
{
    const SushiProfiler::Registry registry;
    SushiProfiler::SessionStart started =
        SushiProfiler::Session::start(train_step_window(), registry);
    ASSERT_TRUE(started.ok()) << started.error;
    SushiProfiler::PeriodTimeline& step = started.session->timeline("train.step");

    mark_period(Timeline::TrainStep);
    mark_period(Timeline::InferenceBatch);

    EXPECT_EQ(step.current(), SushiProfiler::PeriodTimeline::K_NO_PERIOD);
    static_cast<void>(started.session->stop());
}

/** @brief Tests that each mark under a bound session opens one period of its timeline. */
TEST(Medium_Timelines, ABoundSessionRecordsOnePeriodPerMark)
{
    const SushiProfiler::Registry registry;
    SushiProfiler::SessionStart started =
        SushiProfiler::Session::start(train_step_window(), registry);
    ASSERT_TRUE(started.ok()) << started.error;
    SushiProfiler::Session& session = *started.session;
    SushiProfiler::PeriodTimeline& step = session.timeline("train.step");
    SushiProfiler::PeriodTimeline& batch = session.timeline("inference.batch");
    {
        const TimelineBinding binding(session);
        mark_train_steps(5U);
        mark_period(Timeline::InferenceBatch);
        mark_period(Timeline::InferenceBatch);
    }
    mark_period(Timeline::TrainStep);

    const SushiProfiler::SessionStop stopped = session.stop();

    EXPECT_EQ(step.current(), 4U);
    EXPECT_EQ(batch.current(), 1U);
    EXPECT_TRUE(stopped.problems.empty());
    ASSERT_EQ(stopped.record.periods.size(), 4U);
    EXPECT_TRUE(stopped.record.periods[0].warmup);
    EXPECT_EQ(period_indices(stopped.record), (std::vector<std::size_t>{0U, 1U, 2U, 3U}));
}
