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
#include <SushiAI/core/status.hpp>
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

} // namespace

/** @brief Tests that the table lists none, then default, and finds each by its name. */
TEST(Unit_DatasetsPipelineTable, ItListsNoneThenDefault)
{
    const SushiRuntime::span<const PipelineEntry> rows = pipeline_table();
    ASSERT_EQ(rows.size(), 2u);
    EXPECT_EQ(rows[0].name, "none");
    EXPECT_EQ(rows[1].name, "default");

    EXPECT_EQ(find_pipeline("none"), &rows[0]);
    EXPECT_EQ(find_pipeline("default"), &rows[1]);
    EXPECT_EQ(find_pipeline("mixup"), nullptr);
}

/** @brief Tests that each row builds the pipeline of its name. */
TEST(Unit_DatasetsPipelineTable, ARowBuildsThePipelineOfItsName)
{
    PipelineRequest request;
    request.fit.size = ImageSize{K_SIDE, K_SIDE};
    request.channels = 3;
    request.seed = K_SEED;

    for (const PipelineEntry& row : pipeline_table())
    {
        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<const SamplePipeline> pipeline,
                                row.make(request));
        EXPECT_EQ(pipeline->name(), row.name);
    }
}

/** @brief Tests that default refuses an input that is not square or not of three channels. */
TEST(Unit_DatasetsPipelineTable, DefaultRefusesAnInputItDoesNotMake)
{
    const PipelineEntry* const row = find_pipeline(K_PIPELINE_DEFAULT);
    ASSERT_NE(row, nullptr);

    PipelineRequest request;
    request.fit.size = ImageSize{4, 6};
    request.channels = 3;
    SA_EXPECT_FAILURE(row->make(request), StatusCode::InvalidDescription, "'default'",
                      "square", "4 by 6");

    request.fit.size = ImageSize{4, 4};
    request.channels = 1;
    SA_EXPECT_FAILURE(row->make(request), StatusCode::InvalidDescription, "'default'",
                      "3 channels", "has 1");
}
