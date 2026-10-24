/**************************************************************************/
/* test_worker_arguments.cpp                                              */
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
 * @file test_worker_arguments.cpp
 * @brief Tests read_worker_arguments, which sorts the command line of sushiai_worker.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>

#include "worker_arguments.hpp"

namespace
{
    using SushiAI::Apps::WorkerArguments;
    using SushiAI::Apps::WorkerCall;

    /** @brief Sorts @p arguments as the worker's command line after its program name. */
    WorkerArguments read(const std::vector<const char*>& arguments)
    {
        return SushiAI::Apps::read_worker_arguments(
            SushiRuntime::span<const char* const>(arguments.data(), arguments.size()));
    }
} // namespace

/** @brief Tests that the version flags, short and long, ask for the version. */
TEST(Small_WorkerArguments, RegressionTheVersionFlagsComeFirstAndPrint)
{
    for (const char* flag : {"--version", "-V"})
    {
        const WorkerArguments read_flag = read({flag});
        EXPECT_EQ(read_flag.call, WorkerCall::Version) << flag;
        ASSERT_EQ(read_flag.forwarded.size(), 1u) << flag;
        EXPECT_EQ(std::string(read_flag.forwarded[0]), "--version") << flag;
    }
}

/** @brief Tests that the help flags, short and long, ask for the usage. */
TEST(Small_WorkerArguments, RegressionTheHelpFlagsComeFirstAndPrint)
{
    for (const char* flag : {"--help", "-h"})
    {
        const WorkerArguments read_flag = read({flag});
        EXPECT_EQ(read_flag.call, WorkerCall::Help) << flag;
        ASSERT_EQ(read_flag.forwarded.size(), 1u) << flag;
        EXPECT_EQ(std::string(read_flag.forwarded[0]), "--help") << flag;
    }
}

/** @brief Tests that another flag in the place of the training file is refused. */
TEST(Small_WorkerArguments, RegressionAFlagInThePlaceOfTheTrainingFileIsRefused)
{
    EXPECT_EQ(read({"--master", "host:1", "train.json"}).call, WorkerCall::Refused);
    EXPECT_EQ(read({}).call, WorkerCall::Refused);
    EXPECT_NE(std::string(SushiAI::Apps::K_WORKER_USAGE_LINE).find("<train.json> --master"),
              std::string::npos);
}

/** @brief Tests that a training file first is served, with every later argument forwarded. */
TEST(Small_WorkerArguments, ATrainingFileFirstIsServedWithTheRestForwarded)
{
    const WorkerArguments served = read({"train.json", "--master", "host:1", "--name", "w"});

    EXPECT_EQ(served.call, WorkerCall::Serve);
    EXPECT_EQ(served.spec_path, "train.json");
    ASSERT_EQ(served.forwarded.size(), 4u);
    EXPECT_EQ(std::string(served.forwarded[0]), "--master");
    EXPECT_EQ(std::string(served.forwarded[3]), "w");
}
