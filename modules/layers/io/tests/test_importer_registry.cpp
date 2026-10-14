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
 * @file test_importer_registry.cpp
 * @brief Tests the importer registry: insertion order, lookup, a repeated format and the
 *        import of a model through the row a format names.
 * @author Mustafa Garip
 */

#include <filesystem>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/importer_registry.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Returns a model whose only note is the number of files it was given. */
    Result<IO::ImportedModel> count_files(SushiRuntime::span<const std::filesystem::path> files)
    {
        IO::ImportedModel imported;
        imported.notes.push_back(std::to_string(files.size()));
        return imported;
    }

    /** @brief Refuses a call without files, as an importer does for files it cannot read. */
    Result<IO::ImportedModel> refuse_no_file(
        SushiRuntime::span<const std::filesystem::path> files)
    {
        SA_FAIL_IF(files.empty(), StatusCode::InvalidFile, "the stub reads no file");
        return IO::ImportedModel{};
    }

    /** @brief A row under a format no module registers. */
    constexpr IO::ImporterEntry K_FIRST_ROW{"first", &count_files};

    /** @brief A second row, whose importer refuses. */
    constexpr IO::ImporterEntry K_SECOND_ROW{"second", &refuse_no_file};
} // namespace

/** @brief Tests that rows() returns the rows in the order add() received them. */
TEST(Unit_ImporterRegistry, AddKeepsInsertionOrder)
{
    IO::ImporterRegistry registry;
    SA_EXPECT_OK(registry.add(K_FIRST_ROW));
    SA_EXPECT_OK(registry.add(K_SECOND_ROW));

    const SushiRuntime::span<const IO::ImporterEntry> rows = registry.rows();
    ASSERT_EQ(rows.size(), 2u);
    EXPECT_EQ(rows[0].format, "first");
    EXPECT_EQ(rows[1].format, "second");
}

/** @brief Tests that find returns null for a format no row holds and the row of one it holds. */
TEST(Unit_ImporterRegistry, FindReturnsNullForAnUnknownFormat)
{
    IO::ImporterRegistry registry;
    EXPECT_EQ(registry.find("first"), nullptr);

    SA_EXPECT_OK(registry.add(K_FIRST_ROW));
    EXPECT_EQ(registry.find("onnx"), nullptr);

    const IO::ImporterEntry* row = registry.find("first");
    ASSERT_NE(row, nullptr);
    EXPECT_EQ(row->format, "first");
}

/** @brief Tests that add refuses a format a row already holds and names the format. */
TEST(Unit_ImporterRegistry, AFormatAddedTwiceIsRefused)
{
    IO::ImporterRegistry registry;
    SA_EXPECT_OK(registry.add(K_FIRST_ROW));

    SA_EXPECT_FAILURE(registry.add(K_FIRST_ROW), StatusCode::InvalidDescription, "'first'",
                      "already registered");
}

/** @brief Tests that import_model hands the files to the row the format names. */
TEST(Unit_ImporterRegistry, ImportModelCallsTheRowOfTheFormat)
{
    IO::ImporterRegistry registry;
    SA_EXPECT_OK(registry.add(K_FIRST_ROW));
    const std::vector<std::filesystem::path> files = {"a.cfg", "a.weights"};

    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            IO::import_model(registry, "first", files));
    ASSERT_EQ(imported.notes.size(), 1u);
    EXPECT_EQ(imported.notes[0], "2");
}

/** @brief Tests that the failure of a row reaches the caller unchanged. */
TEST(Unit_ImporterRegistry, ImportModelReturnsTheFailureOfTheRow)
{
    IO::ImporterRegistry registry;
    SA_EXPECT_OK(registry.add(K_SECOND_ROW));

    SA_EXPECT_FAILURE(IO::import_model(registry, "second", {}), StatusCode::InvalidFile,
                      "the stub reads no file");
}

/** @brief Tests that an unknown format is refused with every format the registry holds. */
TEST(Unit_ImporterRegistry, AnUnknownFormatListsTheFormatsTheRegistryHolds)
{
    IO::ImporterRegistry registry;
    SA_EXPECT_OK(registry.add(K_FIRST_ROW));
    SA_EXPECT_OK(registry.add(K_SECOND_ROW));

    SA_EXPECT_FAILURE(IO::import_model(registry, "onnx", {}), StatusCode::InvalidDescription,
                      "'onnx'", "'first', 'second'");
}
