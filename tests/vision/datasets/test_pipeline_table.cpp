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
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/augment_pipeline.hpp>
#include <SushiAI/vision/datasets/fit_mask.hpp>
#include <SushiAI/vision/datasets/fit_pipeline.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/pipeline_table.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/sample_pipeline.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/image.hpp>

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

    /** @brief Holds an input the default row refuses and the text of the refusal. */
    struct InputRefusal
    {
        /** @brief Names the row. */
        const char* name;

        /** @brief Holds the size of the request. */
        ImageSize size;

        /** @brief Holds the channels of the request. */
        std::int32_t channels;

        /** @brief Holds a text the refusal carries. */
        const char* message;

        /** @brief Holds another text the refusal carries. */
        const char* detail;
    };

    /** @brief Holds one refused input per row. */
    class Small_DatasetsPipelineTableRefusal : public ::testing::TestWithParam<InputRefusal>
    {
    };

    /** @brief Names a test after the name of its row. */
    std::string table_row_name(const ::testing::TestParamInfo<std::string_view>& info)
    {
        return std::string(info.param);
    }
} // namespace

/** @brief Tests that the table lists none, then default. */
TEST(Small_DatasetsPipelineTable, ItListsNoneThenDefault)
{
    const SushiRuntime::span<const PipelineEntry> rows = pipeline_table();

    ASSERT_EQ(rows.size(), 2u);
    EXPECT_EQ(rows[0].name, "none");
    EXPECT_EQ(rows[1].name, "default");
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
                         ::testing::Values(std::string_view("none"), std::string_view("default")),
                         table_row_name);

/** @brief Tests that find_pipeline gives no row for a name the table lacks. */
TEST(Small_DatasetsPipelineTable, AnUnknownNameFindsNoRow)
{
    EXPECT_EQ(find_pipeline("mixup"), nullptr);
}

/** @brief Tests that default refuses an input that is not square or not of three channels. */
TEST_P(Small_DatasetsPipelineTableRefusal, DefaultRefusesAnInputItDoesNotMake)
{
    const InputRefusal& refusal = GetParam();
    const PipelineEntry* const row = find_pipeline(K_PIPELINE_DEFAULT);
    ASSERT_NE(row, nullptr);
    PipelineRequest request;
    request.fit.size = refusal.size;
    request.channels = refusal.channels;

    SA_EXPECT_FAILURE(row->make(request), StatusCode::InvalidDescription, "'default'",
                      refusal.message, refusal.detail);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DatasetsPipelineTableRefusal,
    ::testing::Values(InputRefusal{"ANonSquareSize", ImageSize{4, 6}, 3, "square", "4 by 6"},
                      InputRefusal{"OneChannel", ImageSize{4, 4}, 1, "3 channels", "has 1"}),
    [](const ::testing::TestParamInfo<InputRefusal>& info) { return std::string(info.param.name); });
