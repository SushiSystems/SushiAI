/**************************************************************************/
/* test_sample_pipeline.cpp                                               */
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
 * @file test_sample_pipeline.cpp
 * @brief Tests the sample pipeline interface through a fake pipeline driven by the base.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/sample_pipeline.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;

namespace
{
    /** @brief Copies the file and the class of each listed sample, offset by the epoch. */
    class EpochPipeline final : public SamplePipeline
    {
        public:
            /** @brief Returns "fake_epoch". */
            [[nodiscard]] std::string_view name() const noexcept override
            {
                return "fake_epoch";
            }

            /** @brief Fills one fitted sample per index; refuses an index the list lacks. */
            [[nodiscard]] Status produce(const SampleList& list,
                                         SushiRuntime::span<const std::size_t> samples,
                                         std::int64_t epoch,
                                         SushiRuntime::span<FittedSample> fitted) const override
            {
                for (std::size_t row = 0; row < samples.size(); ++row)
                {
                    SA_FAIL_IF(samples[row] >= list.samples.size(),
                               StatusCode::InvalidDescription, "sample {} is outside the list",
                               samples[row]);
                    const Sample& sample = list.samples[samples[row]];
                    fitted[row].file = sample.file;
                    fitted[row].class_index =
                        sample.class_index.value_or(0) + static_cast<std::int32_t>(epoch);
                }

                return Status();
            }
    };

    /** @brief Returns a list of three samples of the classes 0, 1 and 2 in the files 0 to 2. */
    SampleList three_samples()
    {
        SampleList list;
        list.files = {"a.png", "b.png", "c.png"};
        for (std::uint32_t i = 0; i < 3; ++i)
        {
            Sample sample;
            sample.file = i;
            sample.class_index = static_cast<std::int32_t>(i);
            list.samples.push_back(sample);
        }

        return list;
    }
} // namespace

/** @brief Tests that produce() fills one fitted sample per index, in the order given. */
TEST(Small_DatasetsSamplePipeline, ProduceFillsOneSamplePerIndexInOrder)
{
    const std::unique_ptr<const SamplePipeline> pipeline = std::make_unique<EpochPipeline>();
    const std::vector<std::size_t> indices = {2, 0};
    std::vector<FittedSample> fitted(indices.size());

    SA_EXPECT_OK(pipeline->produce(three_samples(), indices, 0, fitted));

    EXPECT_EQ(pipeline->name(), "fake_epoch");
    EXPECT_EQ(fitted[0].file, 2u);
    EXPECT_EQ(fitted[0].class_index, 2);
    EXPECT_EQ(fitted[1].file, 0u);
    EXPECT_EQ(fitted[1].class_index, 0);
}

/** @brief Tests that the epoch the caller passes reaches the pipeline. */
TEST(Small_DatasetsSamplePipeline, TheEpochReachesThePipeline)
{
    const std::unique_ptr<const SamplePipeline> pipeline = std::make_unique<EpochPipeline>();
    const std::vector<std::size_t> indices = {1};
    std::vector<FittedSample> fitted(1);

    SA_EXPECT_OK(pipeline->produce(three_samples(), indices, 3, fitted));

    EXPECT_EQ(fitted[0].class_index, 4);
}

/** @brief Tests that the failure of a sample reaches the caller through the interface. */
TEST(Small_DatasetsSamplePipeline, AFailureReachesTheCaller)
{
    const std::unique_ptr<const SamplePipeline> pipeline = std::make_unique<EpochPipeline>();
    const std::vector<std::size_t> indices = {0, 7};
    std::vector<FittedSample> fitted(indices.size());

    SA_EXPECT_FAILURE(pipeline->produce(three_samples(), indices, 0, fitted),
                      StatusCode::InvalidDescription, "sample 7");
}
