/**************************************************************************/
/* test_profile.cpp                                                       */
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
 * @file test_profile.cpp
 * @brief Tests that SA_ZONE records a site named by its group and stage.
 * @author Mustafa Garip
 * @see tests/core/profile/test_profile.md
 */

#include <SushiAI/profile/profile.hpp>
#include <SushiAI/profile_records/profile_records.hpp>

#include <algorithm>

#include <gtest/gtest.h>
#include <SushiProfiler/registry/registry.hpp>
#include <SushiProfiler/session/session.hpp>

// SA_ZONE compiles to nothing below level 1, so the test exists only where a zone records.
#if SA_PROFILE_LEVEL >= 1
namespace
{
    /** @brief Opens the next period of @p timeline and runs one zone inside it. */
    void run_zoned_period(SushiProfiler::PeriodTimeline& timeline)
    {
        static_cast<void>(timeline.mark());
        SA_ZONE("data", "read");
    }
} // namespace

/** @brief Tests that a zone run in a period is recorded as a site named by its group and stage. */
TEST(Medium_Profile, AZoneCarriesItsGroupAndStageName)
{
    const SushiProfiler::Registry registry;
    SushiProfiler::SessionStart started =
        SushiProfiler::Session::start(SushiAI::Test::following_timeline("zones", 0U, 2U), registry);
    ASSERT_TRUE(started.ok()) << started.error;
    SushiProfiler::PeriodTimeline& timeline = started.session->timeline("zones");
    run_zoned_period(timeline);
    run_zoned_period(timeline);
    run_zoned_period(timeline);

    const SushiProfiler::SessionStop stopped = started.session->stop();

    const auto found = std::find_if(stopped.record.sites.begin(), stopped.record.sites.end(),
                                    [](const SushiProfiler::SiteRecord& site)
    {
        return site.name == "data.read";
    });
    ASSERT_NE(found, stopped.record.sites.end());
    EXPECT_EQ(found->group, "data");
    EXPECT_EQ(found->role, SushiProfiler::SiteRole::Zone);
}
#endif
