/**************************************************************************/
/* test_importer_registry.cpp                                             */
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
 * @file test_importer_registry.cpp
 * @brief Tests the importer registry: insertion order, lookup, a repeated format and the import of
 *        a model through the row a format names.
 * @author Mustafa Garip
 */

#include <filesystem>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/in_memory_importer.hpp>
#include <SushiAI/io/importer_registry.hpp>

using namespace SushiAI;

namespace
{
    using IO::K_COUNTING_IMPORTER_ROW;
    using IO::K_REFUSING_IMPORTER_ROW;
} // namespace

/** @brief Tests that rows() returns the rows in the order add() received them. */
TEST(Small_ImporterRegistry, AddKeepsInsertionOrder)
{
    IO::ImporterRegistry registry;
    SA_EXPECT_OK(registry.add(K_COUNTING_IMPORTER_ROW));
    SA_EXPECT_OK(registry.add(K_REFUSING_IMPORTER_ROW));

    const SushiRuntime::span<const IO::ImporterEntry> rows = registry.rows();
    ASSERT_EQ(rows.size(), 2u);
    EXPECT_EQ(rows[0].format, "counting");
    EXPECT_EQ(rows[1].format, "refusing");
}

/** @brief Tests that find returns null for a format no row holds and the row of one it holds. */
TEST(Small_ImporterRegistry, FindReturnsNullForAnUnknownFormat)
{
    IO::ImporterRegistry registry;
    EXPECT_EQ(registry.find("counting"), nullptr);

    SA_EXPECT_OK(registry.add(K_COUNTING_IMPORTER_ROW));
    EXPECT_EQ(registry.find("onnx"), nullptr);

    const IO::ImporterEntry* row = registry.find("counting");
    ASSERT_NE(row, nullptr);
    EXPECT_EQ(row->format, "counting");
}

/** @brief Tests that add refuses a format a row already holds and names the format. */
TEST(Small_ImporterRegistry, AFormatAddedTwiceIsRefused)
{
    IO::ImporterRegistry registry;
    SA_EXPECT_OK(registry.add(K_COUNTING_IMPORTER_ROW));

    SA_EXPECT_FAILURE(registry.add(K_COUNTING_IMPORTER_ROW), StatusCode::InvalidDescription,
                      "'counting'", "already registered");
}

/** @brief Tests that import_model hands the files to the row the format names. */
TEST(Small_ImporterRegistry, ImportModelCallsTheRowOfTheFormat)
{
    IO::ImporterRegistry registry;
    SA_EXPECT_OK(registry.add(K_COUNTING_IMPORTER_ROW));
    const std::vector<std::filesystem::path> files = {"a.cfg", "a.weights"};

    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            IO::import_model(registry, "counting", files));
    ASSERT_EQ(imported.notes.size(), 1u);
    EXPECT_EQ(imported.notes[0], "2");
}

/** @brief Tests that the failure of a row reaches the caller unchanged. */
TEST(Small_ImporterRegistry, ImportModelReturnsTheFailureOfTheRow)
{
    IO::ImporterRegistry registry;
    SA_EXPECT_OK(registry.add(K_REFUSING_IMPORTER_ROW));

    SA_EXPECT_FAILURE(IO::import_model(registry, "refusing", {}), StatusCode::InvalidFile,
                      "the importer reads no file");
}

/** @brief Tests that an unknown format is refused with every format the registry holds. */
TEST(Small_ImporterRegistry, AnUnknownFormatListsTheFormatsTheRegistryHolds)
{
    IO::ImporterRegistry registry;
    SA_EXPECT_OK(registry.add(K_COUNTING_IMPORTER_ROW));
    SA_EXPECT_OK(registry.add(K_REFUSING_IMPORTER_ROW));

    SA_EXPECT_FAILURE(IO::import_model(registry, "onnx", {}), StatusCode::InvalidDescription,
                      "'onnx'", "'counting', 'refusing'");
}
