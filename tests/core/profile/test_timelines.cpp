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
 * @file test_timelines.cpp
 * @brief Tests the names of the timelines and mark_period with and without a bound session.
 * @author Mustafa Garip
 * @see tests/core/profile/test_timelines.md
 */

#include <SushiAI/profile/timelines.hpp>

#include <cstdint>

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
    using SushiAI::Profile::timeline_name;

    /** @brief Returns a configuration that follows `train.step` for one warm-up and three periods. */
    SushiProfiler::ProfilerConfig train_step_window()
    {
        return SushiAI::Test::following_timeline("train.step", 1U, 3U);
    }
} // namespace

TEST(Small_Timelines, NamesAreTrainStepAndInferenceBatch)
{
    EXPECT_EQ(timeline_name(Timeline::TrainStep), "train.step");
    EXPECT_EQ(timeline_name(Timeline::InferenceBatch), "inference.batch");
}

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
        for (std::uint32_t index = 0U; index < 5U; ++index)
            mark_period(Timeline::TrainStep);
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
    for (std::size_t index = 0U; index < stopped.record.periods.size(); ++index)
        EXPECT_EQ(stopped.record.periods[index].index, index);
}
