/**************************************************************************/
/* test_datasets_tokens_section.cpp                                       */
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
 * @file test_datasets_tokens_section.cpp
 * @brief Tests that a "tokens" data block builds a next-token dataset and refuses bad keys.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/language/datasets/token_dataset.hpp>
#include <SushiAI/language/datasets/token_file.hpp>
#include <SushiAI/language/datasets/tokens_section.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include "datasets_test_support.hpp"

using namespace SushiAI;
using namespace SushiAI::Language::Datasets;
using Language::Text::TokenID;

namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the ids of the file these tests read, and its vocabulary size. */
    constexpr std::int32_t K_IDS = 40;

    /** @brief Provides an engine and the token file these tests read. */
    class Unit_LanguageDatasetsTokensSection : public SaTest::EngineTest
    {
        protected:
            /** @brief Writes the ids 0 to 39 and returns the path with forward slashes. */
            std::string rising_file(const Support::ScratchDirectory& scratch)
            {
                const std::filesystem::path file = scratch.path() / "rising.satk";
                const std::vector<TokenID> ids = Support::rising_ids(K_IDS);
                SA_EXPECT_OK(write_token_file(
                    file, SushiRuntime::span<const TokenID>(ids.data(), ids.size()), K_IDS));
                return file.generic_string();
            }

            /** @brief Parses a data block whose "file" is @p file and whose rest is @p keys. */
            Config::Value section_of(const std::string& file, const std::string& keys)
            {
                return Config::parse_json("{\"file\": \"" + file + "\"" + keys + "}", "<test>");
            }
    };
} // namespace

/** @brief Tests that the block's window, stride and batch size reach the dataset. */
TEST_F(Unit_LanguageDatasetsTokensSection, TheKeysDescribeTheDataset)
{
    const Support::ScratchDirectory scratch("sushiai_tokens_section_keys");
    const std::string file = rising_file(scratch);

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        make_tokens_dataset(*engine,
                            section_of(file, R"(, "window": 4, "stride": 5, "batch_size": 2)"),
                            "the data block"));

    EXPECT_EQ(dataset->batch_size(), 2);
    EXPECT_EQ(dataset->sample_shape(), (Shape{4}));
    EXPECT_EQ(dataset->sample_dtype(), DType::Int32);
    EXPECT_EQ(dataset->batch_count(), 4u);

    const Data::TargetDescription* const labels =
        Data::find_target(dataset->targets(), Data::K_ROLE_LABELS);
    ASSERT_NE(labels, nullptr);
    EXPECT_EQ(labels->classes, K_IDS);

    Data::Batch batch;
    SA_EXPECT_OK(dataset->read(1, batch));
    const TokenID* const inputs = batch.inputs.data_as<TokenID>();
    EXPECT_EQ(inputs[0], 10);
    EXPECT_EQ(inputs[4], 15);
}

/** @brief Tests that a block without "batch_size" has that of TokenDatasetOptions. */
TEST_F(Unit_LanguageDatasetsTokensSection, AnAbsentBatchSizeIsTheDefault)
{
    const Support::ScratchDirectory scratch("sushiai_tokens_section_default");
    const std::string file = rising_file(scratch);

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        make_tokens_dataset(*engine, section_of(file, R"(, "window": 1, "stride": 1)"),
                            "the data block"));

    EXPECT_EQ(dataset->batch_size(), TokenDatasetOptions{}.batch_size);
}

/** @brief Tests that an absent, a mistyped and an unknown key are each refused by name. */
TEST_F(Unit_LanguageDatasetsTokensSection, AKeyThatDoesNotFitIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_tokens_section_refused");
    const std::string file = rising_file(scratch);

    SA_EXPECT_FAILURE(make_tokens_dataset(*engine, section_of(file, ""), "the data block"),
                      StatusCode::InvalidDescription, "the data block", "has no 'window'");
    SA_EXPECT_FAILURE(make_tokens_dataset(*engine,
                                          Config::parse_json(R"({"window": 4})", "<test>"),
                                          "the data block"),
                      StatusCode::InvalidDescription, "the data block", "has no 'file'");
    SA_EXPECT_FAILURE(make_tokens_dataset(*engine, section_of(file, R"(, "window": "four")"),
                                          "the data block"),
                      StatusCode::InvalidDescription, "'window'", "whole number");
    SA_EXPECT_FAILURE(make_tokens_dataset(*engine, section_of(file, R"(, "window": 0)"),
                                          "the data block"),
                      StatusCode::InvalidDescription, "'window'", "positive");
    SA_EXPECT_FAILURE(make_tokens_dataset(
                          *engine, section_of(file, R"(, "window": 4, "shuffle": true)"),
                          "the data block"),
                      StatusCode::InvalidDescription, "the data block", "'shuffle'",
                      "does not have");
}

/** @brief Tests that a file that is not there fails and names no dataset. */
TEST_F(Unit_LanguageDatasetsTokensSection, AMissingFileFails)
{
    const Support::ScratchDirectory scratch("sushiai_tokens_section_missing");
    const std::string file = (scratch.path() / "absent.satk").generic_string();

    const Result<std::unique_ptr<Data::Dataset>> made =
        make_tokens_dataset(*engine, section_of(file, R"(, "window": 4)"), "the data block");

    EXPECT_FALSE(made.status().is_ok());
}
