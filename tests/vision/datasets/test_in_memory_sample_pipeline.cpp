/**************************************************************************/
/* test_in_memory_sample_pipeline.cpp                                     */
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
 * @file test_in_memory_sample_pipeline.cpp
 * @brief Tests that InMemorySamplePipeline keeps the promise of the SamplePipeline interface.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/in_memory_sample_pipeline.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/sample_pipeline.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;

namespace
{
    /** @brief Returns a list of three samples, in the files 4 to 6, of the classes 0, 1 and 2. */
    SampleList three_samples()
    {
        SampleList list;
        for (std::uint32_t i = 0; i < 3; ++i)
        {
            Sample sample;
            sample.file = 4 + i;
            sample.class_index = static_cast<std::int32_t>(i);
            list.samples.push_back(sample);
        }

        return list;
    }

    /** @brief Views @p values as the span produce() takes. */
    SushiRuntime::span<const std::size_t> view_of(const std::vector<std::size_t>& values)
    {
        return SushiRuntime::span<const std::size_t>(values.data(), values.size());
    }

    /** @brief Views @p values as the span produce() fills. */
    SushiRuntime::span<FittedSample> fill_of(std::vector<FittedSample>& values)
    {
        return SushiRuntime::span<FittedSample>(values.data(), values.size());
    }
} // namespace

/** @brief Tests that the pipeline names itself. */
TEST(Small_InMemorySamplePipeline, NamesItself)
{
    const InMemorySamplePipeline pipeline;

    EXPECT_EQ(pipeline.name(), "in-memory");
}

/** @brief Tests that produce through the interface fills one fitted sample per index. */
TEST(Small_InMemorySamplePipeline, ProduceFillsOneFittedSamplePerIndex)
{
    const InMemorySamplePipeline concrete;
    const SamplePipeline& pipeline = concrete;
    const SampleList list = three_samples();
    const std::vector<std::size_t> indices = {2, 0};
    std::vector<FittedSample> fitted(2);

    SA_EXPECT_OK(pipeline.produce(list, view_of(indices), 0, fill_of(fitted)));

    EXPECT_EQ(fitted[0].file, 6u);
    EXPECT_EQ(fitted[0].class_index, 2);
    EXPECT_EQ(fitted[1].file, 4u);
    EXPECT_EQ(fitted[1].class_index, 0);
}

/** @brief Tests that an index the list lacks is refused, naming the index. */
TEST(Small_InMemorySamplePipeline, AnIndexTheListLacksIsRefused)
{
    const InMemorySamplePipeline pipeline;
    const SampleList list = three_samples();
    const std::vector<std::size_t> indices = {3};
    std::vector<FittedSample> fitted(1);

    SA_EXPECT_FAILURE(pipeline.produce(list, view_of(indices), 0, fill_of(fitted)),
                      StatusCode::InvalidDescription, "sample 3");
}

/** @brief Tests that indices and fitted samples of different lengths are refused. */
TEST(Small_InMemorySamplePipeline, DifferentLengthsAreRefused)
{
    const InMemorySamplePipeline pipeline;
    const SampleList list = three_samples();
    const std::vector<std::size_t> indices = {0, 1};
    std::vector<FittedSample> fitted(1);

    SA_EXPECT_FAILURE(pipeline.produce(list, view_of(indices), 0, fill_of(fitted)),
                      StatusCode::InvalidDescription, "2 indices for 1 fitted");
}

/** @brief Tests that a scripted failure is returned and no fitted sample is touched. */
TEST(Small_InMemorySamplePipeline, AScriptedFailureIsReturnedBeforeAnySampleIsTouched)
{
    const InMemorySamplePipeline pipeline(Status::from_message(StatusCode::InvalidFile, []()
    {
        return std::string("a.png is truncated");
    }));
    const SampleList list = three_samples();
    const std::vector<std::size_t> indices = {1};
    std::vector<FittedSample> fitted(1);

    SA_EXPECT_FAILURE(pipeline.produce(list, view_of(indices), 0, fill_of(fitted)),
                      StatusCode::InvalidFile, "a.png");
    EXPECT_EQ(fitted[0].file, 0u);
}
