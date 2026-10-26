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
 * @file test_profile.cpp
 * @brief Tests that SA_ZONE records a site named by its group and stage.
 * @author Mustafa Garip
 * @see tests/core/profile/test_profile.md
 */

#include <SushiAI/profile/profile.hpp>
#include <SushiAI/profile_records/profile_records.hpp>

#include <gtest/gtest.h>
#include <SushiProfiler/registry/registry.hpp>
#include <SushiProfiler/session/session.hpp>

TEST(Medium_Profile, AZoneCarriesItsGroupAndStageName)
{
    if (SA_PROFILE_LEVEL < 1)
        GTEST_SKIP() << "the build compiles every zone away";
    const SushiProfiler::Registry registry;
    SushiProfiler::SessionStart started =
        SushiProfiler::Session::start(SushiAI::Test::following_timeline("zones", 0U, 2U), registry);
    ASSERT_TRUE(started.ok()) << started.error;
    SushiProfiler::PeriodTimeline& timeline = started.session->timeline("zones");
    for (int index = 0; index < 3; ++index)
    {
        static_cast<void>(timeline.mark());
        SA_ZONE("data", "read");
    }

    const SushiProfiler::SessionStop stopped = started.session->stop();

    const SushiProfiler::SiteRecord* found = nullptr;
    for (const SushiProfiler::SiteRecord& site : stopped.record.sites)
    {
        if (site.name == "data.read")
            found = &site;
    }
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->group, "data");
    EXPECT_EQ(found->role, SushiProfiler::SiteRole::Zone);
}
