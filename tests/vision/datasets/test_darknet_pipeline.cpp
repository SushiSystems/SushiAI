/**************************************************************************/
/* test_darknet_pipeline.cpp                                              */
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
 * @file test_darknet_pipeline.cpp
 * @brief Tests the pipeline of Darknet's detector augmentation.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "datasets_test_support.hpp"
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/darknet_pipeline.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/sample_pipeline.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/darknet_augment.hpp>
#include <SushiAI/vision/imaging/darknet_resize.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/pipeline.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
using Vision::Imaging::DarknetAugmentSettings;
using Vision::Imaging::Image;
using Vision::Imaging::ImageSize;
using Vision::Imaging::NormalizedBox;

namespace
{
    /** @brief Holds the side of the input of the tests. */
    constexpr std::int32_t K_SIDE = 16;

    /** @brief Holds the seed of every pipeline below. */
    constexpr std::uint64_t K_SEED = 20261115;

    /** @brief Holds the settings of a .cfg that augments: jitter .3, hue .1, saturation 1.5. */
    DarknetAugmentSettings augmenting()
    {
        DarknetAugmentSettings settings;
        settings.jitter = 0.3f;
        settings.hue = 0.1f;
        settings.saturation = 1.5f;
        settings.exposure = 1.5f;
        return settings;
    }

    /** @brief Returns an RGB image whose bytes rise from a first value, wrapping at 256. */
    Image rising_image(ImageSize size, std::uint8_t first)
    {
        Image image(size, 3);
        for (std::size_t i = 0; i < image.byte_count(); ++i)
            image.data()[i] = static_cast<std::uint8_t>(first + 3 * i);

        return image;
    }

    /** @brief Returns three images of three sizes, sample i holding one object of class i. */
    SampleList three_images(MemorySource& source)
    {
        const ImageSize sizes[] = {{20, 24}, {24, 20}, {32, 32}};
        const char* const names[] = {"a.saim", "b.saim", "c.saim"};

        SampleList list;
        list.class_count = 3;
        for (std::size_t i = 0; i < 3; ++i)
        {
            Sample sample;
            sample.file = static_cast<std::uint32_t>(list.files.size());
            sample.objects.add(static_cast<std::int32_t>(i),
                               NormalizedBox{0.5f, 0.5f, 0.5f, 0.5f});
            list.files.push_back(Support::add_raw_image(
                source, names[i], rising_image(sizes[i], static_cast<std::uint8_t>(50 * i))));
            list.samples.push_back(sample);
        }

        return list;
    }

    /** @brief Returns the pipeline of K_SIDE, K_SEED and @p settings over @p source. */
    std::unique_ptr<DarknetPipeline> darknet(const DarknetAugmentSettings& settings,
                                             const ByteSource& source)
    {
        DarknetOptions options;
        options.size = ImageSize{K_SIDE, K_SIDE};
        options.seed = K_SEED;
        options.augment = settings;
        options.source = &source;

        auto made = DarknetPipeline::make(options);
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
} // namespace

/** @brief Tests that the pipeline is named darknet. */
TEST(Small_DatasetsDarknetPipeline, ItIsNamedDarknet)
{
    MemorySource source;
    const std::unique_ptr<DarknetPipeline> pipeline = darknet(augmenting(), source);
    ASSERT_NE(pipeline, nullptr);

    EXPECT_EQ(pipeline->name(), "darknet");
}

/** @brief Tests that settings of no change resize each sample with Darknet's kernel and its boxes with its scale. */
TEST(Small_DatasetsDarknetPipeline, TheIdentitySettingsOnlyResize)
{
    MemorySource source;
    const SampleList list = three_images(source);
    const std::unique_ptr<DarknetPipeline> pipeline = darknet(DarknetAugmentSettings{}, source);
    ASSERT_NE(pipeline, nullptr);

    const std::vector<FittedSample> fitted = produce(*pipeline, list, {0, 1, 2}, 0);

    const ImageSize sizes[] = {{20, 24}, {24, 20}, {32, 32}};
    for (std::size_t i = 0; i < 3; ++i)
    {
        Image expected;
        const Image original = rising_image(sizes[i], static_cast<std::uint8_t>(50 * i));
        ASSERT_TRUE(Vision::Imaging::resize_darknet(original.view(), ImageSize{K_SIDE, K_SIDE},
                                                    expected)
                        .is_ok());
        EXPECT_TRUE(same_pixels(fitted[i].image, expected)) << "sample " << i;

        // The object spans [0.25, 0.75] of each side, so it spans [4, 12] of the input.
        ASSERT_EQ(fitted[i].objects.size(), 1u);
        EXPECT_EQ(fitted[i].objects.classes()[0], static_cast<std::int32_t>(i));
        EXPECT_FLOAT_EQ(fitted[i].objects.boxes()[0].x1, 4.0f);
        EXPECT_FLOAT_EQ(fitted[i].objects.boxes()[0].y1, 4.0f);
        EXPECT_FLOAT_EQ(fitted[i].objects.boxes()[0].x2, 12.0f);
        EXPECT_FLOAT_EQ(fitted[i].objects.boxes()[0].y2, 12.0f);
    }
}

/** @brief Tests that a sample is the crop and the colour of the draws keyed by its seed, epoch and index. */
TEST(Small_DatasetsDarknetPipeline, ASampleIsTheAugmentationOfItsKeyedDraws)
{
    MemorySource source;
    const SampleList list = three_images(source);
    const std::unique_ptr<DarknetPipeline> pipeline = darknet(augmenting(), source);
    ASSERT_NE(pipeline, nullptr);

    const std::vector<FittedSample> fitted = produce(*pipeline, list, {1}, 3);

    Vision::Imaging::Random random = Vision::Imaging::make_sample_random(K_SEED, 3, 1);
    const ImageSize size{24, 20};
    SA_ASSERT_OK_AND_ASSIGN(
        const Vision::Imaging::DarknetAugmentPlan plan,
        Vision::Imaging::plan_darknet_augment(augmenting(), size, random));
    Image expected;
    ASSERT_TRUE(Vision::Imaging::apply_darknet_augment(rising_image(size, 50).view(), plan,
                                                       ImageSize{K_SIDE, K_SIDE}, expected)
                    .is_ok());

    EXPECT_TRUE(same_pixels(fitted[0].image, expected));
    EXPECT_EQ(fitted[0].file, 1u);
}

/** @brief Tests that a sample depends on its index and its epoch alone. */
TEST(Small_DatasetsDarknetPipeline, ASampleDoesNotDependOnItsBatch)
{
    MemorySource source;
    const SampleList list = three_images(source);
    const std::unique_ptr<DarknetPipeline> pipeline = darknet(augmenting(), source);
    ASSERT_NE(pipeline, nullptr);

    const std::vector<FittedSample> batch = produce(*pipeline, list, {0, 1, 2}, 2);
    const std::vector<FittedSample> reversed = produce(*pipeline, list, {2, 1, 0}, 2);
    const std::vector<FittedSample> alone = produce(*pipeline, list, {1}, 2);

    EXPECT_TRUE(same_pixels(batch[1].image, reversed[1].image));
    EXPECT_TRUE(same_pixels(batch[1].image, alone[0].image));
    EXPECT_TRUE(same_pixels(batch[0].image, reversed[2].image));
}

/** @brief Tests that a new epoch draws again. */
TEST(Small_DatasetsDarknetPipeline, ANewEpochChangesTheImage)
{
    MemorySource source;
    const SampleList list = three_images(source);
    const std::unique_ptr<DarknetPipeline> pipeline = darknet(augmenting(), source);
    ASSERT_NE(pipeline, nullptr);

    const std::vector<FittedSample> first = produce(*pipeline, list, {2}, 0);
    const std::vector<FittedSample> second = produce(*pipeline, list, {2}, 1);

    EXPECT_FALSE(same_pixels(first[0].image, second[0].image));
}

/** @brief Tests that a sample with a mask is refused, naming its file and the pipeline. */
TEST(Small_DatasetsDarknetPipeline, ASampleWithAMaskIsRefused)
{
    MemorySource source;
    SampleList list = three_images(source);
    list.samples[0].mask = static_cast<std::uint32_t>(list.files.size());
    list.files.push_back(Support::add_raw_image(
        source, "mask.saim", Support::uniform_image(ImageSize{20, 24}, 1, 0)));

    const std::unique_ptr<DarknetPipeline> pipeline = darknet(augmenting(), source);
    ASSERT_NE(pipeline, nullptr);

    const std::vector<std::size_t> samples = {0};
    std::vector<FittedSample> fitted(1);
    SA_EXPECT_FAILURE(pipeline->produce(list,
                                        SushiRuntime::span<const std::size_t>(samples.data(), 1), 0,
                                        SushiRuntime::span<FittedSample>(fitted.data(), 1)),
                      StatusCode::InvalidDescription, "a.saim", "has a mask", "'darknet'");
}

/** @brief Tests that make refuses a size, a channel count and settings it cannot serve. */
TEST(Small_DatasetsDarknetPipeline, MakeRefusesWhatItCannotBuild)
{
    DarknetOptions options;
    options.size = ImageSize{0, 8};
    SA_EXPECT_FAILURE(DarknetPipeline::make(options), StatusCode::InvalidDescription,
                      "DarknetPipeline", "0 by 8");

    options.size = ImageSize{8, 8};
    options.channels = 5;
    SA_EXPECT_FAILURE(DarknetPipeline::make(options), StatusCode::InvalidDescription,
                      "5 channels");

    options.channels = 3;
    options.augment.jitter = 0.5f;
    SA_EXPECT_FAILURE(DarknetPipeline::make(options), StatusCode::InvalidDescription,
                      "DarknetPipeline", "the jitter is 0.5");
}

/** @brief Tests that produce refuses an epoch below zero and counts that differ. */
TEST(Small_DatasetsDarknetPipeline, ProduceRefusesAnEpochAndCountsItCannotServe)
{
    MemorySource source;
    const SampleList list = three_images(source);
    const std::unique_ptr<DarknetPipeline> pipeline = darknet(augmenting(), source);
    ASSERT_NE(pipeline, nullptr);

    const std::vector<std::size_t> samples = {0, 1};
    std::vector<FittedSample> fitted(2);
    SA_EXPECT_FAILURE(pipeline->produce(list,
                                        SushiRuntime::span<const std::size_t>(samples.data(), 2),
                                        -1, SushiRuntime::span<FittedSample>(fitted.data(), 2)),
                      StatusCode::InvalidDescription, "epoch -1");
    SA_EXPECT_FAILURE(pipeline->produce(list,
                                        SushiRuntime::span<const std::size_t>(samples.data(), 2),
                                        0, SushiRuntime::span<FittedSample>(fitted.data(), 1)),
                      StatusCode::InvalidDescription, "2 samples", "1 fitted");
}
