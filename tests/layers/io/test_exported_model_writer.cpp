/**************************************************************************/
/* test_exported_model_writer.cpp                                         */
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
 * @file test_exported_model_writer.cpp
 * @brief Tests the writer of an export: an existing file, the replace mode and a repeated name.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/exported_model.hpp>
#include <SushiAI/io/exported_model_writer.hpp>
#include <SushiAI/scratch_files/scratch_directory.hpp>

using namespace SushiAI;
using IO::ExistingFiles;

namespace
{
    using SushiAI::Test::ScratchDirectory;

    /** @brief Returns a file named @p name that holds the bytes of @p text. */
    IO::ExportedFile make_file(const std::string& name, const std::string& text)
    {
        IO::ExportedFile file;
        file.name = name;
        for (const char c : text)
            file.bytes.push_back(static_cast<std::byte>(c));
        return file;
    }

    /** @brief Returns the bytes of the file at @p path; empty when it cannot be opened. */
    std::string read_bytes(const std::filesystem::path& path)
    {
        std::ifstream in(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
} // namespace

/** @brief Tests that every file is written in order under its name. */
TEST(Small_ExportedModelWriter, WritesEveryFile)
{
    const ScratchDirectory scratch("sushiai_export_files");
    IO::ExportedModel exported;
    exported.files = {make_file("net.cfg", "[net]"), make_file("net.weights", "abc")};

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<std::filesystem::path> written,
                            IO::write_exported_model(exported, scratch.path(),
                                                     ExistingFiles::Refuse));
    ASSERT_EQ(written.size(), 2u);
    EXPECT_EQ(read_bytes(written[0]), "[net]");
    EXPECT_EQ(read_bytes(written[1]), "abc");
}

/** @brief Tests that an existing file stops the export before any file is written. */
TEST(Small_ExportedModelWriter, AnExistingFileIsRefusedBeforeAnyIsWritten)
{
    const ScratchDirectory scratch("sushiai_export_existing");
    std::filesystem::create_directories(scratch.path());
    std::ofstream(scratch.path() / "net.weights") << "old";
    IO::ExportedModel exported;
    exported.files = {make_file("net.cfg", "[net]"), make_file("net.weights", "abc")};

    SA_EXPECT_FAILURE(IO::write_exported_model(exported, scratch.path(), ExistingFiles::Refuse),
                      StatusCode::Incompatible, "net.weights", "exists already");
    EXPECT_FALSE(std::filesystem::exists(scratch.path() / "net.cfg"));
    EXPECT_EQ(read_bytes(scratch.path() / "net.weights"), "old");
}

/** @brief Tests that the replace mode overwrites an existing file. */
TEST(Small_ExportedModelWriter, ForceReplacesAnExistingFile)
{
    const ScratchDirectory scratch("sushiai_export_force");
    std::filesystem::create_directories(scratch.path());
    std::ofstream(scratch.path() / "net.weights") << "old";
    IO::ExportedModel exported;
    exported.files = {make_file("net.weights", "new")};

    SA_EXPECT_OK(IO::write_exported_model(exported, scratch.path(), ExistingFiles::Replace));
    EXPECT_EQ(read_bytes(scratch.path() / "net.weights"), "new");
}

/** @brief Tests that two files of one name and a name with a directory are refused. */
TEST(Small_ExportedModelWriter, TwoFilesOfOneNameAreRefused)
{
    const ScratchDirectory scratch("sushiai_export_twice");
    IO::ExportedModel twice;
    twice.files = {make_file("net.cfg", "a"), make_file("net.cfg", "b")};
    SA_EXPECT_FAILURE(IO::write_exported_model(twice, scratch.path(), ExistingFiles::Replace),
                      StatusCode::InvalidDescription, "'net.cfg' twice");

    IO::ExportedModel nested;
    nested.files = {make_file("sub/net.cfg", "a")};
    SA_EXPECT_FAILURE(IO::write_exported_model(nested, scratch.path(), ExistingFiles::Replace),
                      StatusCode::InvalidDescription, "no plain file name");
    EXPECT_FALSE(std::filesystem::exists(scratch.path() / "net.cfg"));
}
