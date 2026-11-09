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
 * @file test_worker_arguments.cpp
 * @brief Tests read_worker_arguments, which sorts the command line of sushiai_worker.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <support/test_names.hpp>

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

namespace
{
/** @brief Holds a flag and the command line it forms alone. */
struct FlagCase
{
    /** @brief Names the case. */
    const char* name;

    /** @brief The flag, which is the whole command line. */
    const char* flag;
};
} // namespace

namespace
{
/** @brief Holds a command line the worker refuses. */
struct RefusedCase
{
    /** @brief Names the case. */
    const char* name;

    /** @brief The command line after the program name. */
    std::vector<const char*> arguments;
};
} // namespace

/** @brief Runs the rows of flags that ask for the version. */
class Small_WorkerArgumentsVersion : public ::testing::TestWithParam<FlagCase> {};

/** @brief Runs the rows of flags that ask for the usage. */
class Small_WorkerArgumentsHelp : public ::testing::TestWithParam<FlagCase> {};

/** @brief Runs the rows of command lines the worker refuses. */
class Small_WorkerArgumentsRefused : public ::testing::TestWithParam<RefusedCase> {};

/** @brief Tests that the version flags, short and long, ask for the version. */
TEST_P(Small_WorkerArgumentsVersion, RegressionTheVersionFlagsComeFirstAndPrint)
{
    const WorkerArguments read_flag = read({GetParam().flag});

    EXPECT_EQ(read_flag.call, WorkerCall::Version);
    ASSERT_EQ(read_flag.forwarded.size(), 1u);
    EXPECT_EQ(std::string(read_flag.forwarded[0]), "--version");
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_WorkerArgumentsVersion,
                         ::testing::Values(FlagCase{"TheLongFlag", "--version"},
                                           FlagCase{"TheShortFlag", "-V"}),
                         [](const ::testing::TestParamInfo<FlagCase>& info)
                         {
                             return ::SushiAI::Test::parameter_name(info.param.name);
                         });

/** @brief Tests that the help flags, short and long, ask for the usage. */
TEST_P(Small_WorkerArgumentsHelp, RegressionTheHelpFlagsComeFirstAndPrint)
{
    const WorkerArguments read_flag = read({GetParam().flag});

    EXPECT_EQ(read_flag.call, WorkerCall::Help);
    ASSERT_EQ(read_flag.forwarded.size(), 1u);
    EXPECT_EQ(std::string(read_flag.forwarded[0]), "--help");
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_WorkerArgumentsHelp,
                         ::testing::Values(FlagCase{"TheLongFlag", "--help"},
                                           FlagCase{"TheShortFlag", "-h"}),
                         [](const ::testing::TestParamInfo<FlagCase>& info)
                         {
                             return ::SushiAI::Test::parameter_name(info.param.name);
                         });

/** @brief Tests that another flag in the place of the training file, or no argument, is refused. */
TEST_P(Small_WorkerArgumentsRefused, RegressionAFlagInThePlaceOfTheTrainingFileIsRefused)
{
    EXPECT_EQ(read(GetParam().arguments).call, WorkerCall::Refused);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_WorkerArgumentsRefused,
    ::testing::Values(RefusedCase{"AFlagInThePlaceOfTheFile", {"--master", "host:1", "train.json"}},
                      RefusedCase{"AnEmptyCommandLine", {}}),
    [](const ::testing::TestParamInfo<RefusedCase>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.name);
    });

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
