/**************************************************************************/
/* test_pipeline_table.cpp                                                */
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
 * @file test_pipeline_table.cpp
 * @brief Tests the pipeline table.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "datasets_test_support.hpp"
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/darknet_pipeline.hpp>
#include <SushiAI/vision/datasets/augment_pipeline.hpp>
#include <SushiAI/vision/datasets/fit_mask.hpp>
#include <SushiAI/vision/datasets/fit_pipeline.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/pipeline_table.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/sample_pipeline.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <support/test_names.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
using Vision::Imaging::Image;
using Vision::Imaging::ImageSize;
using Vision::Imaging::NormalizedBox;

namespace
{
    /** @brief Holds the training size of the tests. */
    constexpr std::int32_t K_SIDE = 16;

    /** @brief Holds the seed of every pipeline below. */
    constexpr std::uint64_t K_SEED = 20261011;

    /** @brief Holds the name of each row of the table, one row of a test each. */
    class Small_DatasetsPipelineTableName : public ::testing::TestWithParam<std::string_view>
    {
    };

    /** @brief Names a test after the name of its row. */
    std::string table_row_name(const ::testing::TestParamInfo<std::string_view>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param);
    }
} // namespace

/** @brief Tests that the table lists none, default, then darknet. */
TEST(Small_DatasetsPipelineTable, ItListsNoneDefaultThenDarknet)
{
    const SushiRuntime::span<const PipelineEntry> rows = pipeline_table();

    ASSERT_EQ(rows.size(), 3u);
    EXPECT_EQ(rows[0].name, "none");
    EXPECT_EQ(rows[1].name, "default");
    EXPECT_EQ(rows[2].name, "darknet");
}

/** @brief Tests that the darknet row lists its four keys and the other rows list none. */
TEST(Small_DatasetsPipelineTable, OnlyTheDarknetRowListsKeys)
{
    EXPECT_TRUE(find_pipeline(K_PIPELINE_NONE)->keys.empty());
    EXPECT_TRUE(find_pipeline(K_PIPELINE_DEFAULT)->keys.empty());

    const PipelineEntry* const row = find_pipeline(K_PIPELINE_DARKNET);
    ASSERT_NE(row, nullptr);
    EXPECT_EQ(std::vector<std::string_view>(row->keys.begin(), row->keys.end()),
              (std::vector<std::string_view>{"jitter", "hue", "saturation", "exposure"}));
}

/** @brief Tests that the darknet row reads a key as a float32 and refuses what it cannot read. */
TEST(Small_DatasetsPipelineTable, TheDarknetRowRefusesAKeyItCannotUse)
{
    const PipelineEntry* const row = find_pipeline(K_PIPELINE_DARKNET);
    ASSERT_NE(row, nullptr);
    PipelineRequest request;
    request.fit.size = ImageSize{K_SIDE, K_SIDE};
    request.channels = 3;
    request.context = "data";

    request.section = Config::Value::object({"jitter", "hue", "saturation", "exposure"},
                                            {Config::Value::real(0.3), Config::Value::real(0.1),
                                             Config::Value::real(1.5), Config::Value::real(1.5)});
    SA_EXPECT_OK(row->make(request));

    request.section = Config::Value::object({"jitter"}, {Config::Value::string("wide")});
    SA_EXPECT_FAILURE(row->make(request), StatusCode::InvalidDescription, "data 'jitter'");

    request.section = Config::Value::object({"jitter"}, {Config::Value::real(0.5)});
    SA_EXPECT_FAILURE(row->make(request), StatusCode::InvalidDescription, "the jitter is 0.5");

    request.section = Config::Value::object({"exposure"}, {Config::Value::real(0.9)});
    SA_EXPECT_FAILURE(row->make(request), StatusCode::InvalidDescription, "the exposure 0.9");
}

/** @brief Tests that find_pipeline finds the row of a name in the table. */
TEST_P(Small_DatasetsPipelineTableName, FindPipelineFindsEachNameInTheTable)
{
    const PipelineEntry* const found = find_pipeline(GetParam());

    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->name, GetParam());
}

/** @brief Tests that each row builds the pipeline of its name. */
TEST_P(Small_DatasetsPipelineTableName, ARowBuildsThePipelineOfItsName)
{
    const PipelineEntry* const row = find_pipeline(GetParam());
    ASSERT_NE(row, nullptr);
    PipelineRequest request;
    request.fit.size = ImageSize{K_SIDE, K_SIDE};
    request.channels = 3;
    request.seed = K_SEED;

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<const SamplePipeline> pipeline,
                            row->make(request));

    EXPECT_EQ(pipeline->name(), row->name);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_DatasetsPipelineTableName,
                         ::testing::Values(std::string_view("none"), std::string_view("default"),
                                           std::string_view("darknet")),
                         table_row_name);

/** @brief Tests that find_pipeline gives no row for a name the table lacks. */
TEST(Small_DatasetsPipelineTable, AnUnknownNameFindsNoRow)
{
    EXPECT_EQ(find_pipeline("mixup"), nullptr);
}

/** @brief Tests that default refuses an input of another channel count than three. */
TEST(Small_DatasetsPipelineTable, DefaultRefusesAnInputOfOneChannel)
{
    const PipelineEntry* const row = find_pipeline(K_PIPELINE_DEFAULT);
    ASSERT_NE(row, nullptr);
    PipelineRequest request;
    request.fit.size = ImageSize{4, 4};
    request.channels = 1;

    SA_EXPECT_FAILURE(row->make(request), StatusCode::InvalidDescription, "'default'",
                      "3 channels", "has 1");
}

/** @brief Tests that default builds its pipeline for an input that is not square. */
TEST(Small_DatasetsPipelineTable, DefaultBuildsForANonSquareInput)
{
    const PipelineEntry* const row = find_pipeline(K_PIPELINE_DEFAULT);
    ASSERT_NE(row, nullptr);
    PipelineRequest request;
    request.fit.size = ImageSize{4, 6};
    request.channels = 3;
    request.seed = K_SEED;

    SA_EXPECT_OK(row->make(request));
}
