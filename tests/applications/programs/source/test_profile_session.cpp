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

    /** @brief Returns the failure of a step of a run. */
    Status failed_step()
    {
        return Status::from_message(StatusCode::DeviceFailed,
                                    []() { return std::string("a step failed"); });
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
                                           backend().runtime()),
                      StatusCode::FileNotFound, "no_such_profile.json");
}

/** @brief Tests that text that is not JSON is refused naming its origin. */
TEST(Small_ProfileSession, ABadJsonNamesTheFile)
{
    SA_EXPECT_FAILURE(Apps::check_profile_config("{\"sushiprofiler\": 1,", "broken_profile.json"),
                      StatusCode::InvalidDescription, "broken_profile.json");
}

/** @brief Tests that an unknown key is refused naming the origin and the key. */
TEST(Small_ProfileSession, AnUnknownKeyNamesTheFile)
{
    SA_EXPECT_FAILURE(
        Apps::check_profile_config("{\"sushiprofiler\": 1, \"bogus_key\": true}",
                                   "unknown_profile.json"),
        StatusCode::InvalidDescription, "unknown_profile.json", "bogus_key");
}

/** @brief Tests that ending a run that failed returns the failure of the run. */
TEST_F(Medium_ProfileSession, EndingAFailedRunReturnsItsOwnFailure)
{
    const SaTest::ScratchDirectory folder("sushiai_profile_failed_status");
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ProfileSession> session,
                            ProfileSession::open(write_config(folder.path()), backend().runtime()));
    run_three_steps();

    const Status ended = Apps::end_profiled_run(session, failed_step());

    SA_EXPECT_FAILURE(ended, StatusCode::DeviceFailed, "a step failed");
}

/** @brief Tests that ending a profiled run that failed returns the failure of the run. */
TEST_F(Medium_ProfileSession, EndingAFailedRunReturnsItsFailure)
{
    const SaTest::ScratchDirectory folder("sushiai_profile_failure_returned");
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ProfileSession> session,
                            ProfileSession::open(write_config(folder.path()), backend().runtime()));
    run_three_steps();

    EXPECT_FALSE(Apps::end_profiled_run(session, failed_step()).is_ok());
}

/** @brief Tests that a run that fails still gets its record written. */
TEST_F(Medium_ProfileSession, ARunThatFailsStillWritesItsRecord)
{
    const SaTest::ScratchDirectory folder("sushiai_profile_failed_run");
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ProfileSession> session,
                            ProfileSession::open(write_config(folder.path()), backend().runtime()));
    run_three_steps();

    (void)Apps::end_profiled_run(session, failed_step());

    expect_two_periods_written(folder.path());
}

/** @brief Tests that a session dropped without close(), as an exception does, writes its record. */
TEST_F(Medium_ProfileSession, ADroppedSessionStillWritesItsRecord)
{
    const SaTest::ScratchDirectory folder("sushiai_profile_dropped");
    {
        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ProfileSession> session,
                                ProfileSession::open(write_config(folder.path()),
                                                     backend().runtime()));
        run_three_steps();
    }

    expect_two_periods_written(folder.path());
}

/** @brief Tests that a second close() succeeds and writes nothing again. */
TEST_F(Medium_ProfileSession, ASecondCloseSucceeds)
{
    const SaTest::ScratchDirectory folder("sushiai_profile_second_close");
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ProfileSession> session,
                            ProfileSession::open(write_config(folder.path()), backend().runtime()));
    run_three_steps();

    SA_EXPECT_OK(session->close());
    SA_EXPECT_OK(session->close());

    expect_two_periods_written(folder.path());
}
