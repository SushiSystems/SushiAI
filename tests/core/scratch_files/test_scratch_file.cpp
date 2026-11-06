/**************************************************************************/
/* test_scratch_file.cpp                                                  */
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
 * @file test_scratch_file.cpp
 * @brief Tests that ScratchFile names a file under the temporary directory and removes it.
 * @author Mustafa Garip
 */

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <SushiAI/scratch_files/scratch_file.hpp>

namespace
{
    using SushiAI::Test::ScratchFile;

    /** @brief Holds the file name every case uses. */
    constexpr const char* K_NAME = "sushiai_scratch_file_test.txt";

    /** @brief Writes one line to @p path. */
    void write_line(const std::filesystem::path& path)
    {
        std::ofstream out(path);
        out << "scratch\n";
    }
} // namespace

/** @brief Tests that the path is the name under the temporary directory. */
TEST(Medium_ScratchFile, ThePathIsTheNameUnderTheTemporaryDirectory)
{
    const ScratchFile file(K_NAME);

    EXPECT_EQ(file.path(), std::filesystem::temp_directory_path() / K_NAME);
}

/** @brief Tests that construction removes a file a previous run left under the name. */
TEST(Medium_ScratchFile, ConstructionRemovesALeftoverFile)
{
    const std::filesystem::path leftover = std::filesystem::temp_directory_path() / K_NAME;
    write_line(leftover);
    ASSERT_TRUE(std::filesystem::exists(leftover));

    const ScratchFile file(K_NAME);

    EXPECT_FALSE(std::filesystem::exists(file.path()));
}

/** @brief Tests that destruction removes the file a case wrote. */
TEST(Medium_ScratchFile, DestructionRemovesTheFile)
{
    std::filesystem::path written;
    {
        const ScratchFile file(K_NAME);
        written = file.path();
        write_line(written);
        ASSERT_TRUE(std::filesystem::exists(written));
    }

    EXPECT_FALSE(std::filesystem::exists(written));
}
