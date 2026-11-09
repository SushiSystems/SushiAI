/**************************************************************************/
/* test_augment_pipeline.cpp                                              */
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
 * @file test_augment_pipeline.cpp
 * @brief Tests the training transforms of the augment pipeline.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
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

    /** @brief Returns an RGB image whose bytes rise from a first value, wrapping at 256. */
    Image rising_image(ImageSize size, std::uint8_t first)
    {
        Image image(size, 3);
        for (std::size_t i = 0; i < image.byte_count(); ++i)
            image.data()[i] = static_cast<std::uint8_t>(first + i);

        return image;
    }

    /** @brief Returns four images of four sizes, sample i holding one object of class i. */
    SampleList four_images(MemorySource& source)
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
            list.files.push_back(Support::add_raw_image(
                source, names[i], rising_image(sizes[i], static_cast<std::uint8_t>(40 * i))));
            list.samples.push_back(sample);
        }

        return list;
    }

    /** @brief Gives each sample of four_images a mask of its size that holds its index. */
    void add_masks(MemorySource& source, SampleList& list)
    {
        const ImageSize sizes[] = {{20, 16}, {16, 20}, {16, 16}, {10, 20}};
        const char* const names[] = {"a.png", "b.png", "c.png", "d.png"};

        for (std::size_t i = 0; i < 4; ++i)
        {
            list.samples[i].mask = static_cast<std::uint32_t>(list.files.size());
            list.files.push_back(Support::add_raw_image(
                source, names[i],
                Support::uniform_image(sizes[i], 1, static_cast<std::uint8_t>(i))));
        }
    }

    /** @brief Returns the training pipeline of K_SIDE and K_SEED. */
    std::unique_ptr<AugmentPipeline> augment(std::int32_t mosaic_epochs,
                                              const ByteSource* source = nullptr)
    {
        AugmentOptions options;
        options.image_size = K_SIDE;
        options.seed = K_SEED;
        options.mosaic_epochs = mosaic_epochs;
        options.source = source;

        auto made = AugmentPipeline::make(options);
        EXPECT_TRUE(made.has_value()) << made.status().message();
        return made.has_value() ? std::move(made).value() : nullptr;
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


    /** @brief Names a test after the sample index of its row. */
    std::string sample_row_name(const ::testing::TestParamInfo<std::size_t>& info)
    {
        return "Sample" + std::to_string(info.param);
    }

    /** @brief Reports whether every box of a fitted sample lies inside the training square. */
    bool boxes_lie_inside(const FittedSample& sample)
    {
        return std::all_of(sample.objects.boxes().begin(), sample.objects.boxes().end(),
                           [](const Vision::Imaging::PixelBox& box)
                           {
                               return box.x1 >= 0.0f && box.y1 >= 0.0f
                                      && box.x2 <= static_cast<float>(K_SIDE)
                                      && box.y2 <= static_cast<float>(K_SIDE);
                           });
    }

    /** @brief Reports whether every byte of a mask is a class below four or the void byte. */
    bool holds_classes_or_void(const Image& mask)
    {
        return std::all_of(mask.data(), mask.data() + mask.byte_count(),
                           [](std::uint8_t value) { return value < 4 || value == K_MASK_VOID; });
    }

    /** @brief Counts the rows whose image differs between two productions. */
    std::size_t changed_images(const std::vector<FittedSample>& a,
                               const std::vector<FittedSample>& b)
    {
        std::size_t changed = 0;
        for (std::size_t row = 0; row < a.size(); ++row)
            changed += same_pixels(a[row].image, b[row].image) ? 0 : 1;

        return changed;
    }

    /** @brief Holds the sample index of each row of a test over the four images. */
    class Small_DatasetsAugmentPipelineSample : public ::testing::TestWithParam<std::size_t>
    {
    };

    /** @brief Holds the epoch and the sample index of each row of a test over the four images. */
    class Small_DatasetsAugmentPipelineEpochSample
        : public ::testing::TestWithParam<std::tuple<std::int64_t, std::size_t>>
    {
    };

    /** @brief Holds the options and the message of each refused make. */
    struct MakeRefusal
    {
        /** @brief Names the row. */
        const char* name;

        /** @brief Holds the training size. */
        std::int32_t image_size;

        /** @brief Holds the count of epochs of mosaic. */
        std::int32_t mosaic_epochs;

        /** @brief Holds the text the refusal carries. */
        const char* message;
    };

    /** @brief Holds one refused make per row. */
    class Small_DatasetsAugmentPipelineMakeRefusal : public ::testing::TestWithParam<MakeRefusal>
    {
    };

    /** @brief Returns a list of one sample that names no readable file. */
    SampleList list_of_one_sample()
    {
        SampleList list;
        list.files.push_back("never_read.saim");
        list.samples.push_back(Sample());
        return list;
    }
} // namespace

/** @brief Tests that a training sample has the training size and three channels. */
TEST_P(Small_DatasetsAugmentPipelineSample, ASampleHasTheTrainingSizeAndThreeChannels)
{
    const std::size_t row = GetParam();
    MemorySource source;
    const SampleList list = four_images(source);
    const std::unique_ptr<AugmentPipeline> pipeline = augment(1000, &source);
    ASSERT_NE(pipeline, nullptr);
    EXPECT_EQ(pipeline->name(), "default");

    const std::vector<FittedSample> made = produce(*pipeline, list, {row}, 0);

    ASSERT_EQ(made.size(), 1u);
    EXPECT_EQ(made[0].image.size(), (ImageSize{K_SIDE, K_SIDE}));
    EXPECT_EQ(made[0].image.channels(), 3);
}

/** @brief Tests that the boxes of a training sample lie inside the training square. */
TEST_P(Small_DatasetsAugmentPipelineSample, TheBoxesOfASampleLieInsideTheImage)
{
    const std::size_t row = GetParam();
    MemorySource source;
    const SampleList list = four_images(source);
    const std::unique_ptr<AugmentPipeline> pipeline = augment(1000, &source);
    ASSERT_NE(pipeline, nullptr);

    const std::vector<FittedSample> made = produce(*pipeline, list, {row}, 0);

    ASSERT_EQ(made.size(), 1u);
    EXPECT_TRUE(boxes_lie_inside(made[0]));
}

/** @brief Tests that a sample of a list without masks comes out with an empty mask. */
TEST_P(Small_DatasetsAugmentPipelineSample, ASampleWithoutAMaskHasAnEmptyMask)
{
    const std::size_t row = GetParam();
    MemorySource source;
    const SampleList list = four_images(source);
    const std::unique_ptr<AugmentPipeline> pipeline = augment(1000, &source);
    ASSERT_NE(pipeline, nullptr);

    const std::vector<FittedSample> made = produce(*pipeline, list, {row}, 0);

    ASSERT_EQ(made.size(), 1u);
    EXPECT_EQ(made[0].mask.byte_count(), 0u);
}

/** @brief Tests that a sample is the same whether it is made again, alone or in a reversed batch. */
TEST_P(Small_DatasetsAugmentPipelineSample, ASampleDependsOnItsIndexAndItsEpochAlone)
{
    const std::size_t row = GetParam();
    MemorySource source;
    const SampleList list = four_images(source);
    const std::unique_ptr<AugmentPipeline> pipeline = augment(1000, &source);
    ASSERT_NE(pipeline, nullptr);

    const std::vector<FittedSample> whole = produce(*pipeline, list, {0, 1, 2, 3}, 2);
    const std::vector<FittedSample> again = produce(*pipeline, list, {0, 1, 2, 3}, 2);
    const std::vector<FittedSample> alone = produce(*pipeline, list, {row}, 2);
    const std::vector<FittedSample> reversed = produce(*pipeline, list, {3, 2, 1, 0}, 2);

    expect_same(whole[row], again[row]);
    expect_same(whole[row], reversed[3 - row]);
    expect_same(whole[row], alone[0]);
}

/** @brief Tests that a mask comes out at the training size holding classes or the void byte. */
TEST_P(Small_DatasetsAugmentPipelineSample, AMaskTakesTheSizeOfItsSample)
{
    const std::size_t row = GetParam();
    MemorySource source;
    SampleList list = four_images(source);
    add_masks(source, list);
    const std::unique_ptr<AugmentPipeline> pipeline = augment(1000, &source);
    ASSERT_NE(pipeline, nullptr);

    const std::vector<FittedSample> made = produce(*pipeline, list, {row}, 1);

    ASSERT_EQ(made.size(), 1u);
    ASSERT_EQ(made[0].mask.size(), (ImageSize{K_SIDE, K_SIDE}));
    ASSERT_EQ(made[0].mask.channels(), 1);
    EXPECT_TRUE(holds_classes_or_void(made[0].mask));
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_DatasetsAugmentPipelineSample,
                         ::testing::Range<std::size_t>(0, 4), sample_row_name);

/** @brief Tests that the next epoch changes at least one image of the four. */
TEST(Small_DatasetsAugmentPipeline, ANewEpochChangesTheImages)
{
    MemorySource source;
    const SampleList list = four_images(source);
    const std::unique_ptr<AugmentPipeline> pipeline = augment(1000, &source);
    ASSERT_NE(pipeline, nullptr);

    const std::vector<FittedSample> second = produce(*pipeline, list, {0, 1, 2, 3}, 2);
    const std::vector<FittedSample> third = produce(*pipeline, list, {0, 1, 2, 3}, 3);

    EXPECT_GT(changed_images(second, third), 0u);
}

/** @brief Tests that past the mosaic epochs a sample holds the objects of its own image. */
TEST_P(Small_DatasetsAugmentPipelineEpochSample, PastTheMosaicEpochsASampleHoldsItsOwnObjects)
{
    const auto [epoch, row] = GetParam();
    MemorySource source;
    const SampleList list = four_images(source);
    const std::unique_ptr<AugmentPipeline> pipeline = augment(0, &source);
    ASSERT_NE(pipeline, nullptr);

    const std::vector<FittedSample> made = produce(*pipeline, list, {row}, epoch);

    ASSERT_EQ(made.size(), 1u);
    EXPECT_EQ(made[0].objects.classes(),
              std::vector<std::int32_t>{static_cast<std::int32_t>(row)});
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DatasetsAugmentPipelineEpochSample,
    ::testing::Combine(::testing::Range<std::int64_t>(0, 3), ::testing::Range<std::size_t>(0, 4)),
    [](const ::testing::TestParamInfo<std::tuple<std::int64_t, std::size_t>>& info)
    {
        return "Epoch" + std::to_string(std::get<0>(info.param)) + "Sample"
               + std::to_string(std::get<1>(info.param));
    });

/** @brief Tests that an image that cannot be read is the failure, with its file. */
TEST(Small_DatasetsAugmentPipeline, AnImageThatCannotBeReadIsTheFailure)
{
    MemorySource source;
    SampleList list = four_images(source);
    list.files[1] = "gone.saim";
    const std::unique_ptr<AugmentPipeline> pipeline = augment(0, &source);
    ASSERT_NE(pipeline, nullptr);

    const std::size_t samples[] = {1};
    FittedSample fitted;
    SA_EXPECT_FAILURE(pipeline->produce(list, SushiRuntime::span<const std::size_t>(samples, 1),
                                        0, SushiRuntime::span<FittedSample>(&fitted, 1)),
                      StatusCode::FileNotFound, "gone.saim");
}

/** @brief Tests that make refuses a training size of 0 and a negative count of mosaic epochs. */
TEST_P(Small_DatasetsAugmentPipelineMakeRefusal, MakeRefusesWhatItCannotBuild)
{
    const MakeRefusal& row = GetParam();
    AugmentOptions options;
    options.image_size = row.image_size;
    options.mosaic_epochs = row.mosaic_epochs;

    SA_EXPECT_FAILURE(AugmentPipeline::make(options), StatusCode::InvalidDescription,
                      "AugmentPipeline", row.message);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DatasetsAugmentPipelineMakeRefusal,
    ::testing::Values(MakeRefusal{"ASizeOfZero", 0, 1, "a training size of 0"},
                      MakeRefusal{"ANegativeEpochCount", K_SIDE, -1, "-1 epochs of mosaic"}),
    [](const ::testing::TestParamInfo<MakeRefusal>& info) { return ::SushiAI::Test::parameter_name(info.param.name); });

/** @brief Tests that produce refuses a negative epoch. */
TEST(Small_DatasetsAugmentPipeline, ProduceRefusesANegativeEpoch)
{
    const std::unique_ptr<AugmentPipeline> pipeline = augment(0);
    ASSERT_NE(pipeline, nullptr);
    const SampleList list = list_of_one_sample();
    const std::size_t samples[] = {0};
    FittedSample fitted;

    SA_EXPECT_FAILURE(pipeline->produce(list, SushiRuntime::span<const std::size_t>(samples, 1),
                                        -1, SushiRuntime::span<FittedSample>(&fitted, 1)),
                      StatusCode::InvalidDescription, "epoch -1");
}

/** @brief Tests that produce refuses a list of no sample. */
TEST(Small_DatasetsAugmentPipeline, ProduceRefusesAnEmptyList)
{
    const std::unique_ptr<AugmentPipeline> pipeline = augment(0);
    ASSERT_NE(pipeline, nullptr);
    const std::size_t samples[] = {0};
    FittedSample fitted;

    SA_EXPECT_FAILURE(pipeline->produce(SampleList(),
                                        SushiRuntime::span<const std::size_t>(samples, 1), 0,
                                        SushiRuntime::span<FittedSample>(&fitted, 1)),
                      StatusCode::InvalidDescription, "a list of 0 samples");
}
