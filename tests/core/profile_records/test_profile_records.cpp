/**************************************************************************/
/* test_profile_records.cpp                                               */
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
 * @file test_profile_records.cpp
 * @brief Tests the profiling helpers that the tests of other modules share.
 * @author Mustafa Garip
 * @see tests/core/profile_records/test_profile_records.md
 */

#include <SushiAI/profile_records/profile_records.hpp>

#include <filesystem>
#include <gtest/gtest.h>
#include <SushiProfiler/config/profiler_config.hpp>
#include <SushiAI/scratch_files/scratch_directory.hpp>

namespace SaTest = SushiAI::Test;

/** @brief Tests that the timeline, the window and a zero linger land in the configuration. */
TEST(Small_ProfileRecords, FollowingTimelineSetsTheWindowAndNoLinger)
{
    const SushiProfiler::ProfilerConfig config = SaTest::following_timeline("train.step", 1U, 3U);

    EXPECT_EQ(config.timeline, "train.step");
    EXPECT_EQ(config.window.warmup, 1U);
    EXPECT_EQ(config.window.active, 3U);
    EXPECT_EQ(config.buffers.linger_ms, 0U);
}

/** @brief Tests that zones_of maps each zone site of a period to its group. */
TEST(Small_ProfileRecords, ZonesOfMapsEachSiteNameToItsGroup)
{
    SushiProfiler::RunRecord record;
    record.sites.resize(3);
    record.sites[1].name = "data.read";
    record.sites[1].group = "data";
    record.sites[2].name = "host.feed";
    record.sites[2].group = "host";
    SushiProfiler::PeriodRecord period;
    SushiProfiler::ZoneTree tree;
    SushiProfiler::ZoneNode first;
    first.site = 1U;
    SushiProfiler::ZoneNode second;
    second.site = 2U;
    tree.nodes = {first, second};
    period.zones.push_back(tree);

    const std::map<std::string, std::string> zones = SaTest::zones_of(record, period);

    const std::map<std::string, std::string> expected = {{"data.read", "data"},
                                                         {"host.feed", "host"}};
    EXPECT_EQ(zones, expected);
}

/** @brief Tests that the written configuration file reads back through SushiProfiler. */
TEST(Small_ProfileRecords, TheWrittenConfigurationIsReadBySushiProfiler)
{
    const SaTest::ScratchDirectory folder("sushiai_profile_records");

    const std::filesystem::path path = SaTest::write_profile_config(folder.path(), "run.sprec.json");

    const SushiProfiler::ConfigReadResult read =
        SushiProfiler::read_profiler_config(SaTest::read_text(path));
    ASSERT_TRUE(read.ok()) << read.error;
    EXPECT_EQ(read.config->timeline, "train.step");
    EXPECT_EQ(read.config->window.active, 2U);
    EXPECT_EQ(read.config->writers.size(), 1u);
}
