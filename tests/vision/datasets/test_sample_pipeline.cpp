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
 * @file test_sample_pipeline.cpp
 * @brief Tests the contract of the sample pipeline interface over each implementation.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "datasets_test_support.hpp"
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/augment_pipeline.hpp>
#include <SushiAI/vision/datasets/fit_pipeline.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/in_memory_sample_pipeline.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/sample_pipeline.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
using Vision::Imaging::ImageSize;

namespace
{
    /** @brief Names the implementations of SamplePipeline the contract runs over. */
    enum class Implementation
    {
        InMemory,
        Fit,
        Augment,
    };

    /** @brief Holds the training side of the image pipelines the contract builds. */
    constexpr std::int32_t K_SIDE = 16;

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

    /** @brief Returns the list of three samples whose files are three raw images on disk. */
    SampleList three_images(const Support::ScratchDirectory& scratch)
    {
        SampleList list = three_samples();
        const char* const names[] = {"a.saim", "b.saim", "c.saim"};
        for (std::size_t i = 0; i < 3; ++i)
            list.files[i] = Support::write_raw_image(
                scratch, names[i],
                Support::uniform_image(ImageSize{K_SIDE, K_SIDE}, 3,
                                       static_cast<std::uint8_t>(40 * i)));

        return list;
    }

    /** @brief Builds the pipeline of one implementation. */
    std::unique_ptr<const SamplePipeline> make_pipeline(Implementation implementation)
    {
        if (implementation == Implementation::Fit)
        {
            SampleFitOptions fit;
            fit.size = ImageSize{K_SIDE, K_SIDE};
            auto made = FitPipeline::make(fit, 3);
            EXPECT_TRUE(made.status().is_ok());
            return std::move(made).value();
        }

        if (implementation == Implementation::Augment)
        {
            AugmentOptions options;
            options.image_size = K_SIDE;
            auto made = AugmentPipeline::make(options);
            EXPECT_TRUE(made.status().is_ok());
            return std::move(made).value();
        }

        return std::make_unique<InMemorySamplePipeline>();
    }

    /** @brief Expects produce() to fill one fitted sample per index, in the order given. */
    void expect_one_fitted_sample_per_index_in_order(const SamplePipeline& pipeline,
                                                     const SampleList& list)
    {
        const std::vector<std::size_t> indices = {2, 0};
        std::vector<FittedSample> fitted(indices.size());

        SA_EXPECT_OK(pipeline.produce(list, indices, 0, fitted));

        EXPECT_EQ(fitted[0].file, 2u);
        EXPECT_EQ(fitted[1].file, 0u);
    }

    /** @brief Reports whether two productions hold the same file and class in every row. */
    bool same_files_and_classes(const std::vector<FittedSample>& a,
                                const std::vector<FittedSample>& b)
    {
        if (a.size() != b.size())
            return false;

        for (std::size_t row = 0; row < a.size(); ++row)
        {
            if (a[row].file != b[row].file || a[row].class_index != b[row].class_index)
                return false;
        }
        return true;
    }

    /** @brief Holds the implementation of each row that reaches no file. */
    class Small_SamplePipelineContract : public ::testing::TestWithParam<Implementation>
    {
    };

    /** @brief Holds the implementation of each row that reads image files. */
    class Medium_SamplePipelineContract : public ::testing::TestWithParam<Implementation>
    {
    };

    /** @brief Names a test after the implementation of its row. */
    std::string implementation_name(const ::testing::TestParamInfo<Implementation>& info)
    {
        switch (info.param)
        {
            case Implementation::InMemory:
                return "InMemory";
            case Implementation::Fit:
                return "Fit";
            case Implementation::Augment:
                return "Augment";
        }

        return "Unknown";
    }
} // namespace

/** @brief Tests that produce() fills one fitted sample per index, in the order given. */
TEST_P(Small_SamplePipelineContract, ProduceFillsOneSamplePerIndexInOrder)
{
    const std::unique_ptr<const SamplePipeline> pipeline = make_pipeline(GetParam());

    expect_one_fitted_sample_per_index_in_order(*pipeline, three_samples());
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_SamplePipelineContract,
                         ::testing::Values(Implementation::InMemory), implementation_name);

/** @brief Tests that produce() of an image pipeline fills one fitted sample per index, in order. */
TEST_P(Medium_SamplePipelineContract, ProduceFillsOneSamplePerIndexInOrder)
{
    const Support::ScratchDirectory scratch(
        ("sushiai_test_datasets_contract_" + implementation_name({GetParam(), 0})).c_str());
    const std::unique_ptr<const SamplePipeline> pipeline = make_pipeline(GetParam());

    expect_one_fitted_sample_per_index_in_order(*pipeline, three_images(scratch));
}

INSTANTIATE_TEST_SUITE_P(Rows, Medium_SamplePipelineContract,
                         ::testing::Values(Implementation::Fit, Implementation::Augment),
                         implementation_name);

/** @brief Tests that two threads calling produce() at once get the single-thread result. */
TEST(Medium_SamplePipelineThreads, TwoThreadsGetTheResultOfOne)
{
    const InMemorySamplePipeline pipeline;
    const SampleList list = three_samples();
    const std::vector<std::size_t> indices = {2, 0, 1};
    std::vector<FittedSample> single(indices.size());
    ASSERT_TRUE(pipeline.produce(list, indices, 0, single).is_ok());

    std::vector<FittedSample> first(indices.size());
    std::vector<FittedSample> second(indices.size());
    Status first_status;
    Status second_status;
    std::thread other([&] { second_status = pipeline.produce(list, indices, 0, second); });
    first_status = pipeline.produce(list, indices, 0, first);
    other.join();

    SA_EXPECT_OK(first_status);
    SA_EXPECT_OK(second_status);
    EXPECT_TRUE(same_files_and_classes(first, single));
    EXPECT_TRUE(same_files_and_classes(second, single));
}
