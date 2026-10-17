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
 * @file test_run_progress.cpp
 * @brief Tests the default of a run's progress and the refusal of a resume under another seed.
 * @author Mustafa Garip
 */

#include <cstdint>
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
}

/** @brief Tests that a progress nobody filled is the start of a run under the seed 0. */
TEST(Small_RunProgress, ADefaultProgressIsTheStartOfARun)
{
    const RunProgress progress;

    EXPECT_EQ(progress.seed, 0u);
    EXPECT_EQ(progress.step_runs, 0u);
    EXPECT_EQ(progress.epoch, 0u);
    EXPECT_EQ(progress.batch, 0u);
}

/** @brief Tests that the four records have four names, none of them an optimizer's. */
TEST(Small_RunProgress, TheRecordsHaveDistinctNames)
{
    EXPECT_EQ(SushiAI::IO::K_RUN_SEED_STATE, "run.seed");
    EXPECT_EQ(SushiAI::IO::K_RUN_STEP_RUNS_STATE, "run.step_runs");
    EXPECT_EQ(SushiAI::IO::K_RUN_EPOCH_STATE, "run.epoch");
    EXPECT_EQ(SushiAI::IO::K_RUN_BATCH_STATE, "run.batch");
}

/** @brief Tests that the seed a checkpoint holds is accepted and any other is refused. */
TEST(Small_RunProgress, AResumeUnderAnotherSeedIsRefusedWithBothSeeds)
{
    RunProgress saved;
    saved.seed = K_SAVED_SEED;

    SA_EXPECT_OK(SushiAI::IO::check_run_seed(saved, K_SAVED_SEED, "last.sushi"));
    SA_EXPECT_FAILURE(SushiAI::IO::check_run_seed(saved, 7, "last.sushi"),
                      StatusCode::Incompatible, "'last.sushi'", "seed 20261011", "seed 7");
    SA_EXPECT_FAILURE(SushiAI::IO::check_run_seed(RunProgress{}, K_SAVED_SEED, "last.sushi"),
                      StatusCode::Incompatible, "seed 0", "seed 20261011");
}
