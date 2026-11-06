/**************************************************************************/
/* test_scratch_directory.cpp                                             */
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
 * @file test_scratch_directory.cpp
 * @brief Tests that ScratchDirectory names a directory under the temporary one and removes it.
 * @author Mustafa Garip
 */

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <SushiAI/scratch_files/scratch_directory.hpp>

namespace
{
    using SushiAI::Test::ScratchDirectory;

    /** @brief Holds the directory name every case uses. */
    constexpr const char* K_NAME = "sushiai_scratch_directory_test";

    /** @brief Creates @p directory with a nested folder and a file in each. */
    void fill(const std::filesystem::path& directory)
    {
        std::filesystem::create_directories(directory / "nested");
        std::ofstream(directory / "top.txt") << "scratch\n";
        std::ofstream(directory / "nested" / "deep.txt") << "scratch\n";
    }
} // namespace

/** @brief Tests that the path under a base is the name below it. */
TEST(Small_ScratchDirectory, ThePathUnderABaseIsTheNameBelowIt)
{
    const std::filesystem::path base = "base";

    EXPECT_EQ(ScratchDirectory::path_under(base, K_NAME), base / K_NAME);
}

/** @brief Tests that a directory made under a base removes the leftover tree of that base. */
TEST(Medium_ScratchDirectory, ConstructionUnderABaseRemovesTheLeftoverTreeOfThatBase)
{
    const ScratchDirectory base(K_NAME);
    fill(base.path() / "inner");
    ASSERT_TRUE(std::filesystem::exists(base.path() / "inner" / "nested" / "deep.txt"));

    const ScratchDirectory directory(base.path(), "inner");

    EXPECT_EQ(directory.path(), base.path() / "inner");
    EXPECT_FALSE(std::filesystem::exists(directory.path()));
}

/** @brief Tests that construction creates nothing, so the directory need not exist. */
TEST(Medium_ScratchDirectory, ConstructionLeavesTheDirectoryAbsent)
{
    const ScratchDirectory directory(K_NAME);

    EXPECT_FALSE(std::filesystem::exists(directory.path()));
}

/** @brief Tests that construction removes a directory and its contents from a previous run. */
TEST(Medium_ScratchDirectory, ConstructionRemovesALeftoverTree)
{
    const std::filesystem::path leftover = std::filesystem::temp_directory_path() / K_NAME;
    fill(leftover);
    ASSERT_TRUE(std::filesystem::exists(leftover / "nested" / "deep.txt"));

    const ScratchDirectory directory(K_NAME);

    EXPECT_FALSE(std::filesystem::exists(directory.path()));
}

/** @brief Tests that destruction removes the directory with the folders and files below it. */
TEST(Medium_ScratchDirectory, DestructionRemovesTheWholeTree)
{
    std::filesystem::path written;
    {
        const ScratchDirectory directory(K_NAME);
        written = directory.path();
        fill(written);
        ASSERT_TRUE(std::filesystem::exists(written / "nested" / "deep.txt"));
    }

    EXPECT_FALSE(std::filesystem::exists(written));
}
