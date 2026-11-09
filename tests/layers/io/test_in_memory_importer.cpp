/**************************************************************************/
/* test_in_memory_importer.cpp                                            */
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
 * @file test_in_memory_importer.cpp
 * @brief Tests the importer rows that read no file.
 * @author Mustafa Garip
 */

#include <filesystem>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/in_memory_importer.hpp>

using namespace SushiAI;

/** @brief Tests that the counting importer notes how many files it was given. */
TEST(Small_InMemoryImporter, TheCountingImporterNotesTheFilesItWasGiven)
{
    const std::vector<std::filesystem::path> files = {"a.cfg", "a.weights"};

    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            IO::K_COUNTING_IMPORTER_ROW.read(files));

    EXPECT_EQ(imported.notes, (std::vector<std::string>{"2"}));
}

/** @brief Tests that the refusing importer refuses a call without files. */
TEST(Small_InMemoryImporter, TheRefusingImporterRefusesACallWithoutFiles)
{
    SA_EXPECT_FAILURE(IO::K_REFUSING_IMPORTER_ROW.read({}), StatusCode::InvalidFile,
                      "reads no file");
}

/** @brief Tests that the refusing importer accepts a call that names a file. */
TEST(Small_InMemoryImporter, TheRefusingImporterAcceptsACallWithAFile)
{
    const std::vector<std::filesystem::path> files = {"a.cfg"};

    SA_EXPECT_OK(IO::K_REFUSING_IMPORTER_ROW.read(files));
}
