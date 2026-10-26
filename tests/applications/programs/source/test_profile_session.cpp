/**************************************************************************/
/* test_profile_session.cpp                                               */
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
 * @file test_profile_session.cpp
 * @brief Tests that ProfileSession refuses a bad configuration naming its file and writes the record of a run that fails.
 * @author Mustafa Garip
 * @see tests/applications/programs/README.md
 */

#include "profile_session.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <gtest/gtest.h>
#include <SushiProfiler/record_json/record_json.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/profile/timelines.hpp>
#include <SushiAI/profile_records/profile_records.hpp>
#include <SushiAI/scratch_files/scratch_directory.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/applications/programs/README.md.
namespace SaTest = SushiAI::Test;
using Apps::ProfileSession;

namespace
{
    /** @brief Provides the Engine, and so the runtime, every case opens a session on. */
    class Medium_ProfileSession : public SaTest::EngineTest
    {

    };

    /** @brief Holds the file name of the record the configuration of write_config() asks for. */
    constexpr const char* K_RECORD_NAME = "run.sprec.json";

    /** @brief Writes the configuration that follows `train.step` for two periods under @p folder. */
    std::filesystem::path write_config(const std::filesystem::path& folder)
    {
        return SaTest::write_profile_config(folder, K_RECORD_NAME);
    }

    /** @brief Opens three periods of the training timeline. */
    void run_three_steps()
    {
        for (int step = 0; step < 3; ++step)
            Profile::mark_period(Profile::Timeline::TrainStep);
    }

    /** @brief Expects the record under @p folder to hold two periods of `train.step`. */
    void expect_two_periods_written(const std::filesystem::path& folder)
    {
        const std::string text = SaTest::read_text(folder / K_RECORD_NAME);
        ASSERT_FALSE(text.empty()) << "no record at " << (folder / K_RECORD_NAME).string();
        const SushiProfiler::RecordReadResult read = SushiProfiler::read_record(text);
        ASSERT_TRUE(read.ok()) << read.error;
        EXPECT_EQ(read.record->periods.size(), 2u);
    }
} // namespace

/** @brief Tests that a configuration path that does not exist is refused naming the file. */
TEST_F(Medium_ProfileSession, AMissingConfigNamesTheFile)
{
    const SaTest::ScratchDirectory folder("sushiai_profile_missing");

    SA_EXPECT_FAILURE(ProfileSession::open(folder.path() / "no_such_profile.json",
                                           engine->runtime()),
                      StatusCode::FileNotFound, "no_such_profile.json");
}

/** @brief Tests that a file that is not JSON is refused naming the file. */
TEST_F(Medium_ProfileSession, ABadJsonNamesTheFile)
{
    const SaTest::ScratchDirectory folder("sushiai_profile_bad_json");
    const std::filesystem::path path = folder.path() / "broken_profile.json";
    SaTest::write_text(path, "{\"sushiprofiler\": 1,");

    SA_EXPECT_FAILURE(ProfileSession::open(path, engine->runtime()),
                      StatusCode::InvalidDescription, "broken_profile.json");
}

/** @brief Tests that an unknown key is refused naming the file and the key. */
TEST_F(Medium_ProfileSession, AnUnknownKeyNamesTheFile)
{
    const SaTest::ScratchDirectory folder("sushiai_profile_unknown_key");
    const std::filesystem::path path = folder.path() / "unknown_profile.json";
    SaTest::write_text(path, "{\"sushiprofiler\": 1, \"bogus_key\": true}");

    SA_EXPECT_FAILURE(ProfileSession::open(path, engine->runtime()),
                      StatusCode::InvalidDescription, "unknown_profile.json", "bogus_key");
}

/** @brief Tests that a run that fails still gets its record, and keeps its own failure. */
TEST_F(Medium_ProfileSession, ARunThatFailsStillWritesItsRecord)
{
    const SaTest::ScratchDirectory folder("sushiai_profile_failed_run");
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ProfileSession> session,
                            ProfileSession::open(write_config(folder.path()), engine->runtime()));
    run_three_steps();
    const Status failed = Status::from_message(StatusCode::DeviceFailed,
                                               []() { return std::string("a step failed"); });

    const Status ended = Apps::end_profiled_run(session, failed);

    SA_EXPECT_FAILURE(ended, StatusCode::DeviceFailed, "a step failed");
    expect_two_periods_written(folder.path());
}

/** @brief Tests that a session dropped without close(), as an exception does, writes its record. */
TEST_F(Medium_ProfileSession, ADroppedSessionStillWritesItsRecord)
{
    const SaTest::ScratchDirectory folder("sushiai_profile_dropped");
    {
        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ProfileSession> session,
                                ProfileSession::open(write_config(folder.path()),
                                                     engine->runtime()));
        run_three_steps();
    }

    expect_two_periods_written(folder.path());
}

/** @brief Tests that a second close() succeeds and writes nothing again. */
TEST_F(Medium_ProfileSession, ASecondCloseSucceeds)
{
    const SaTest::ScratchDirectory folder("sushiai_profile_second_close");
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ProfileSession> session,
                            ProfileSession::open(write_config(folder.path()), engine->runtime()));
    run_three_steps();

    SA_EXPECT_OK(session->close());
    SA_EXPECT_OK(session->close());

    expect_two_periods_written(folder.path());
}
