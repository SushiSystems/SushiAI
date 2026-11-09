/**************************************************************************/
/* test_tokens_section.cpp                                                */
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
 * @file test_tokens_section.cpp
 * @brief Tests that a "tokens" data block builds a next-token dataset and refuses bad keys.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "datasets_test_support.hpp"
#include <SushiAI/failure/result.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/datasets/token_dataset.hpp>
#include <SushiAI/language/datasets/token_file.hpp>
#include <SushiAI/language/datasets/tokens_section.hpp>
#include <SushiAI/language/text/token.hpp>
#include <support/test_names.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Datasets;
using Language::Text::TokenID;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the ids of the file these tests read, and its vocabulary size. */
    constexpr std::int32_t K_IDS = 40;

    /** @brief Provides an engine and the token file these tests read. */
    class Medium_LanguageDatasetsTokensSection : public SaTest::EngineTest
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
    };

    /** @brief Parses a data block whose "file" is @p file and whose rest is @p keys. */
    Config::Value section_of(const std::string& file, const std::string& keys)
    {
        return SaTest::require_value(
            Config::parse_json("{\"file\": \"" + file + "\"" + keys + "}", "<test>"));
    }

    /** @brief Names a data block that does not fit, and what the refusal says. */
    struct BlockCase
    {
        const char* label;  /**< The name the row prints. */
        const char* json;   /**< The text of the data block. */
        const char* first;  /**< A fragment the refusal names. */
        const char* second; /**< A second fragment, or an empty text. */
        const char* third;  /**< A third fragment, or an empty text. */
    };

    /** @brief Provides the data blocks read_tokens_section refuses. */
    class Small_LanguageDatasetsTokensSectionRefusal : public ::testing::TestWithParam<BlockCase>
    {

    };
} // namespace

/** @brief Tests that the block's window, stride and batch size reach the dataset. */
TEST_F(Medium_LanguageDatasetsTokensSection, TheKeysDescribeTheDataset)
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
}

/** @brief Tests that a read of the dataset a block builds returns the windows at the stride. */
TEST_F(Medium_LanguageDatasetsTokensSection, AReadReturnsTheWindowsAtTheStride)
{
    const Support::ScratchDirectory scratch("sushiai_tokens_section_read");
    const std::string file = rising_file(scratch);

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        make_tokens_dataset(*engine,
                            section_of(file, R"(, "window": 4, "stride": 5, "batch_size": 2)"),
                            "the data block"));
    Data::Batch batch;
    SA_EXPECT_OK(dataset->read(1, batch));

    const TokenID* const inputs = batch.inputs.data_as<TokenID>();
    EXPECT_EQ(inputs[0], 10);
    EXPECT_EQ(inputs[4], 15);
}

/** @brief Tests that the keys of a block are read, and that a stride is left to the window. */
TEST(Small_LanguageDatasetsTokensSection, TheKeysOfABlockAreRead)
{
    const Config::Value section =
        section_of("a.satk", R"(, "window": 4, "stride": 5, "batch_size": 2)");

    SA_ASSERT_OK_AND_ASSIGN(const TokensSection keys,
                            read_tokens_section(section, "the data block"));

    EXPECT_EQ(keys.file, "a.satk");
    EXPECT_EQ(keys.window, 4);
    EXPECT_EQ(keys.options.stride, std::optional<std::int64_t>(5));
    EXPECT_EQ(keys.options.batch_size, 2);
}

/** @brief Tests that a block without "batch_size" or "stride" states 16 windows and no stride. */
TEST(Small_LanguageDatasetsTokensSection, AnAbsentBatchSizeIsSixteen)
{
    const Config::Value section = section_of("a.satk", R"(, "window": 1)");

    SA_ASSERT_OK_AND_ASSIGN(const TokensSection keys,
                            read_tokens_section(section, "the data block"));

    EXPECT_EQ(keys.options.batch_size, 16);
    EXPECT_FALSE(keys.options.stride.has_value());
}

/** @brief Tests that an absent, a mistyped and an unknown key are each refused by name. */
TEST_P(Small_LanguageDatasetsTokensSectionRefusal, AKeyThatDoesNotFitIsRefused)
{
    const BlockCase& row = GetParam();
    const Config::Value section = SaTest::require_value(Config::parse_json(row.json, "<test>"));

    SA_EXPECT_FAILURE(read_tokens_section(section, "the data block"),
                      StatusCode::InvalidDescription, row.first, row.second, row.third);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LanguageDatasetsTokensSectionRefusal,
    ::testing::Values(
        BlockCase{"NoWindow", R"({"file": "a.satk"})", "the data block", "has no 'window'", ""},
        BlockCase{"NoFile", R"({"window": 4})", "the data block", "has no 'file'", ""},
        BlockCase{"AWindowThatIsNoNumber", R"({"file": "a.satk", "window": "four"})", "'window'",
                  "whole number", ""},
        BlockCase{"AWindowOfZero", R"({"file": "a.satk", "window": 0})", "'window'", "positive",
                  ""},
        BlockCase{"AnUnknownKey", R"({"file": "a.satk", "window": 4, "shuffle": true})",
                  "the data block", "'shuffle'", "does not have"}),
    [](const ::testing::TestParamInfo<BlockCase>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.label);
    });

/** @brief Tests that a file that is not there fails and names no dataset. */
TEST_F(Medium_LanguageDatasetsTokensSection, AMissingFileFails)
{
    const Support::ScratchDirectory scratch("sushiai_tokens_section_missing");
    const std::string file = (scratch.path() / "absent.satk").generic_string();

    const Result<std::unique_ptr<Data::Dataset>> made =
        make_tokens_dataset(*engine, section_of(file, R"(, "window": 4)"), "the data block");

    EXPECT_FALSE(made.status().is_ok());
}
