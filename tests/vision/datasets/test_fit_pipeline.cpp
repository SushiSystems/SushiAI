/**************************************************************************/
/* test_fit_pipeline.cpp                                                  */
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
 * @file test_fit_pipeline.cpp
 * @brief Tests the fit pipeline.
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
    /** @brief Returns an RGB image whose bytes rise from a first value, wrapping at 256. */
    Image rising_image(ImageSize size, std::uint8_t first)
    {
        Image image(size, 3);
        for (std::size_t i = 0; i < image.byte_count(); ++i)
            image.data()[i] = static_cast<std::uint8_t>(first + i);

        return image;
    }

    /** @brief Returns four images of four sizes, sample i holding one object of class i. */
    SampleList four_images(const Support::ScratchDirectory& scratch)
    {
        const ImageSize sizes[] = {{20, 16}, {16, 20}, {16, 16}, {10, 20}};
        const char* const names[] = {"a.saim", "b.saim", "c.saim", "d.saim"};

        SampleList list;
        list.class_count = 4;
        for (std::size_t i = 0; i < 4; ++i)
        {
            Sample sample;
            sample.file = static_cast<std::uint32_t>(list.files.size());
            sample.objects.add(static_cast<std::int32_t>(i),
                               NormalizedBox{0.5f, 0.5f, 0.8f, 0.8f});
            list.files.push_back(Support::write_raw_image(
                scratch, names[i], rising_image(sizes[i], static_cast<std::uint8_t>(40 * i))));
            list.samples.push_back(sample);
        }

        return list;
    }

    /** @brief Produces the fitted samples of some indices in one epoch. */
    std::vector<FittedSample> produce(const SamplePipeline& pipeline, const SampleList& list,
                                      const std::vector<std::size_t>& samples,
                                      std::int64_t epoch)
    {
        std::vector<FittedSample> fitted(samples.size());
        const Status status = pipeline.produce(
            list, SushiRuntime::span<const std::size_t>(samples.data(), samples.size()), epoch,
            SushiRuntime::span<FittedSample>(fitted.data(), fitted.size()));
        EXPECT_TRUE(status.is_ok()) << status.message();
        return fitted;
    }

    /** @brief Reports whether two images have the same size, channels and bytes. */
    bool same_pixels(const Image& a, const Image& b)
    {
        if (a.size() != b.size() || a.channels() != b.channels())
            return false;

        for (std::size_t i = 0; i < a.byte_count(); ++i)
        {
            if (a.data()[i] != b.data()[i])
                return false;
        }
        return true;
    }

    /** @brief Expects two fitted samples to hold the same image, objects and mask. */
    void expect_same(const FittedSample& a, const FittedSample& b)
    {
        EXPECT_TRUE(same_pixels(a.image, b.image));
        EXPECT_TRUE(same_pixels(a.mask, b.mask));
        EXPECT_EQ(a.file, b.file);
        EXPECT_EQ(a.objects.classes(), b.objects.classes());
        ASSERT_EQ(a.objects.size(), b.objects.size());

        for (std::size_t i = 0; i < a.objects.size(); ++i)
        {
            EXPECT_EQ(a.objects.boxes()[i].x1, b.objects.boxes()[i].x1);
            EXPECT_EQ(a.objects.boxes()[i].y2, b.objects.boxes()[i].y2);
        }
    }


    /** @brief Holds the sample index of each row of a test over the four images. */
    class Medium_DatasetsFitPipelineSample : public ::testing::TestWithParam<std::size_t>
    {
    };

    /** @brief Names a test after the sample index of its row. */
    std::string sample_row_name(const ::testing::TestParamInfo<std::size_t>& info)
    {
        return "Sample" + std::to_string(info.param);
    }

    /** @brief Names the scratch directory of one row of a test, so rows never share one. */
    std::string row_directory(const char* test, std::size_t row)
    {
        return std::string("sushiai_test_datasets_pipeline_fit_") + test + "_"
               + std::to_string(row);
    }

    /** @brief Returns the letterbox fit of the 8 by 8 input the tests use. */
    SampleFitOptions letterbox_of_eight()
    {
        SampleFitOptions fit;
        fit.size = ImageSize{8, 8};
        fit.fit = SampleFit::Letterbox;
        return fit;
    }
} // namespace

/** @brief Tests that a fitted sample has the input size and three channels. */
TEST_P(Medium_DatasetsFitPipelineSample, AFittedSampleHasTheInputSizeAndThreeChannels)
{
    const std::size_t row = GetParam();
    const Support::ScratchDirectory scratch(row_directory("shape", row).c_str());
    const SampleList list = four_images(scratch);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<FitPipeline> pipeline,
                            FitPipeline::make(letterbox_of_eight(), 3));
    EXPECT_EQ(pipeline->name(), "none");

    const std::vector<FittedSample> made = produce(*pipeline, list, {row}, 0);

    ASSERT_EQ(made.size(), 1u);
    EXPECT_EQ(made[0].image.size(), (ImageSize{8, 8}));
    EXPECT_EQ(made[0].image.channels(), 3);
}

/** @brief Tests that a fitted sample keeps the one object, of the class of its row. */
TEST_P(Medium_DatasetsFitPipelineSample, AFittedSampleKeepsTheClassOfItsRow)
{
    const std::size_t row = GetParam();
    const Support::ScratchDirectory scratch(row_directory("classes", row).c_str());
    const SampleList list = four_images(scratch);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<FitPipeline> pipeline,
                            FitPipeline::make(letterbox_of_eight(), 3));

    const std::vector<FittedSample> made = produce(*pipeline, list, {row}, 0);

    ASSERT_EQ(made.size(), 1u);
    ASSERT_EQ(made[0].objects.size(), 1u);
    EXPECT_EQ(made[0].objects.classes()[0], static_cast<std::int32_t>(row));
}

/** @brief Tests that the fit of a sample is the same in every epoch. */
TEST_P(Medium_DatasetsFitPipelineSample, TheFitIsTheSameInEveryEpoch)
{
    const std::size_t row = GetParam();
    const Support::ScratchDirectory scratch(row_directory("replay", row).c_str());
    const SampleList list = four_images(scratch);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<FitPipeline> pipeline,
                            FitPipeline::make(letterbox_of_eight(), 3));

    const std::vector<FittedSample> first = produce(*pipeline, list, {row}, 0);
    const std::vector<FittedSample> later = produce(*pipeline, list, {row}, 7);

    ASSERT_EQ(first.size(), 1u);
    ASSERT_EQ(later.size(), 1u);
    expect_same(first[0], later[0]);
}

INSTANTIATE_TEST_SUITE_P(Rows, Medium_DatasetsFitPipelineSample,
                         ::testing::Values<std::size_t>(3, 0), sample_row_name);

/** @brief Tests that make refuses an input size of no pixel. */
TEST(Small_DatasetsFitPipeline, MakeRefusesASizeOfNoPixel)
{
    SampleFitOptions fit;
    fit.size = ImageSize{0, 8};

    SA_EXPECT_FAILURE(FitPipeline::make(fit, 3), StatusCode::InvalidDescription, "FitPipeline",
                      "0 by 8");
}

/** @brief Tests that produce refuses a count of samples that differs from the fitted count. */
TEST(Small_DatasetsFitPipeline, ProduceRefusesCountsThatDiffer)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<FitPipeline> pipeline,
                            FitPipeline::make(letterbox_of_eight(), 3));
    const SampleList list;
    const std::size_t samples[] = {0, 1};
    FittedSample fitted;

    SA_EXPECT_FAILURE(pipeline->produce(list, SushiRuntime::span<const std::size_t>(samples, 2),
                                        0, SushiRuntime::span<FittedSample>(&fitted, 1)),
                      StatusCode::InvalidDescription, "2 samples", "1 fitted");
}

/** @brief Tests that a mask of another size than its image is refused with the mask file. */
TEST(Medium_DatasetsFitPipeline, AMaskOfAnotherSizeThanItsImageNamesItsFile)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_pipeline_fit_mask_size");
    SampleList list;
    list.files.push_back(Support::write_raw_image(
        scratch, "images/train/a.saim", Support::uniform_image(ImageSize{2, 4}, 3, 255)));
    list.files.push_back(Support::write_raw_image(
        scratch, "masks/train/a.png", Support::uniform_image(ImageSize{2, 2}, 1, 0)));
    Sample sample;
    sample.file = 0;
    sample.mask = 1u;
    list.samples.push_back(sample);

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<FitPipeline> pipeline,
                            FitPipeline::make(letterbox_of_eight(), 3));
    const std::size_t samples[] = {0};
    FittedSample fitted;

    SA_EXPECT_FAILURE(pipeline->produce(list, SushiRuntime::span<const std::size_t>(samples, 1),
                                        0, SushiRuntime::span<FittedSample>(&fitted, 1)),
                      StatusCode::Incompatible, "masks/train/a.png", "fit_mask");
}
