/**************************************************************************/
/* test_run_progress.cpp                                                  */
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
 * @file test_run_progress.cpp
 * @brief Tests the default of a run's progress and the refusal of a resume under another seed.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/run_progress.hpp>

namespace
{
    using SushiAI::StatusCode;
    using SushiAI::IO::RunProgress;

    /** @brief Holds the seed the saved run of the cases was started with. */
    constexpr std::uint64_t K_SAVED_SEED = 20261011ull;

    /** @brief Holds a record name constant and the text the checkpoint format states for it. */
    struct NameRow
    {
        std::string name;                 /**< The row's name. */
        std::string_view constant;        /**< The constant of the header. */
        std::string_view stated;          /**< The name the format states. */
    };

    /** @brief Runs the record name check over each of the four records. */
    class Small_RunProgressName : public ::testing::TestWithParam<NameRow>
    {
    };

    /** @brief Holds a saved progress, the seed of the resuming run and the words of the refusal. */
    struct SeedRow
    {
        std::string name;          /**< The row's name. */
        RunProgress saved;         /**< What the checkpoint holds. */
        std::uint64_t seed;        /**< The seed of the run that resumes. */
        std::string saved_text;    /**< The saved seed as the message states it. */
        std::string seed_text;     /**< The resuming seed as the message states it. */
    };

    /** @brief Runs the seed refusal over each pair of seeds that differ. */
    class Small_RunProgressSeed : public ::testing::TestWithParam<SeedRow>
    {
    };

    /** @brief Returns a progress saved under the seed @p seed. */
    RunProgress saved_under(std::uint64_t seed)
    {
        RunProgress saved;
        saved.seed = seed;
        return saved;
    }
} // namespace

/** @brief Tests that each run-state record has the name the checkpoint format states. */
TEST_P(Small_RunProgressName, ARecordHasTheNameTheFormatStates)
{
    EXPECT_EQ(GetParam().constant, GetParam().stated);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_RunProgressName,
    ::testing::Values(NameRow{"Seed", SushiAI::IO::K_RUN_SEED_STATE, "run.seed"},
                      NameRow{"StepRuns", SushiAI::IO::K_RUN_STEP_RUNS_STATE, "run.step_runs"},
                      NameRow{"Epoch", SushiAI::IO::K_RUN_EPOCH_STATE, "run.epoch"},
                      NameRow{"Batch", SushiAI::IO::K_RUN_BATCH_STATE, "run.batch"}),
    [](const ::testing::TestParamInfo<NameRow>& info) { return info.param.name; });

/** @brief Tests that the seed a checkpoint holds is accepted. */
TEST(Small_RunProgress, AResumeUnderTheSavedSeedIsAccepted)
{
    SA_EXPECT_OK(
        SushiAI::IO::check_run_seed(saved_under(K_SAVED_SEED), K_SAVED_SEED, "last.sushi"));
}

/** @brief Tests that any other seed is refused, naming the file and both seeds. */
TEST_P(Small_RunProgressSeed, AResumeUnderAnotherSeedIsRefusedWithBothSeeds)
{
    const SeedRow& row = GetParam();
    SA_EXPECT_FAILURE(SushiAI::IO::check_run_seed(row.saved, row.seed, "last.sushi"),
                      StatusCode::Incompatible, "'last.sushi'", row.saved_text, row.seed_text);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_RunProgressSeed,
    ::testing::Values(SeedRow{"AnotherSeed", saved_under(K_SAVED_SEED), 7, "seed 20261011",
                              "seed 7"},
                      SeedRow{"ADefaultProgress", RunProgress{}, K_SAVED_SEED, "seed 0",
                              "seed 20261011"}),
    [](const ::testing::TestParamInfo<SeedRow>& info) { return info.param.name; });
