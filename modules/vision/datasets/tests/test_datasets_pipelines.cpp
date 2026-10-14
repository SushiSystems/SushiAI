/**************************************************************************/
/* test_datasets_pipelines.cpp                                            */
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
 * @file test_datasets_pipelines.cpp
 * @brief Tests the pipeline table and its two pipelines: the fit and the training transforms.
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

#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/datasets/augment_pipeline.hpp>
#include <SushiAI/vision/datasets/fit_mask.hpp>
#include <SushiAI/vision/datasets/fit_pipeline.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/pipeline_table.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/sample_pipeline.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/image.hpp>

#include <SushiAI/failure_expectations/status_expectations.hpp>

#include "datasets_test_support.hpp"

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

    /** @brief Holds the absolute tolerance of a box that repeats the same arithmetic. */
    constexpr float K_REPLAY_TOLERANCE = 0.0f;

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

    /** @brief Gives each sample of four_images a mask of its size that holds its index. */
    void add_masks(const Support::ScratchDirectory& scratch, SampleList& list)
    {
        const ImageSize sizes[] = {{20, 16}, {16, 20}, {16, 16}, {10, 20}};
        const char* const names[] = {"a.png", "b.png", "c.png", "d.png"};

        for (std::size_t i = 0; i < 4; ++i)
        {
            list.samples[i].mask = static_cast<std::uint32_t>(list.files.size());
            list.files.push_back(Support::write_raw_image(
                scratch, names[i],
                Support::uniform_image(sizes[i], 1, static_cast<std::uint8_t>(i))));
        }
    }

    /** @brief Returns the training pipeline of K_SIDE and K_SEED. */
    std::unique_ptr<AugmentPipeline> augment(std::int32_t mosaic_epochs)
    {
        AugmentOptions options;
        options.image_size = K_SIDE;
        options.seed = K_SEED;
        options.mosaic_epochs = mosaic_epochs;

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
            EXPECT_NEAR(a.objects.boxes()[i].x1, b.objects.boxes()[i].x1, K_REPLAY_TOLERANCE);
            EXPECT_NEAR(a.objects.boxes()[i].y2, b.objects.boxes()[i].y2, K_REPLAY_TOLERANCE);
        }
    }
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

/** @brief Tests that the fit gives the input size, the objects and the same result per epoch. */
TEST(Unit_DatasetsFitPipeline, FitsEachSampleTheSameInEveryEpoch)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_pipeline_fit");
    const SampleList list = four_images(scratch);

    SampleFitOptions fit;
    fit.size = ImageSize{8, 8};
    fit.fit = SampleFit::Letterbox;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<FitPipeline> pipeline,
                            FitPipeline::make(fit, 3));
    EXPECT_EQ(pipeline->name(), "none");

    const std::vector<FittedSample> first = produce(*pipeline, list, {3, 0}, 0);
    const std::vector<FittedSample> later = produce(*pipeline, list, {3, 0}, 7);

    ASSERT_EQ(first.size(), 2u);
    for (std::size_t row = 0; row < 2; ++row)
    {
        EXPECT_EQ(first[row].image.size(), (ImageSize{8, 8}));
        EXPECT_EQ(first[row].image.channels(), 3);
        ASSERT_EQ(first[row].objects.size(), 1u);
        expect_same(first[row], later[row]);
    }
    EXPECT_EQ(first[0].objects.classes()[0], 3);
    EXPECT_EQ(first[1].objects.classes()[0], 0);
}

/** @brief Tests that the fit refuses a size outside the bounds and counts that differ. */
TEST(Unit_DatasetsFitPipeline, RefusesWhatItCannotFit)
{
    SampleFitOptions fit;
    fit.size = ImageSize{0, 8};
    SA_EXPECT_FAILURE(FitPipeline::make(fit, 3), StatusCode::InvalidDescription, "FitPipeline",
                      "0 by 8");

    fit.size = ImageSize{8, 8};
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<FitPipeline> pipeline,
                            FitPipeline::make(fit, 3));

    const SampleList list;
    const std::size_t samples[] = {0, 1};
    FittedSample fitted;
    SA_EXPECT_FAILURE(pipeline->produce(list, SushiRuntime::span<const std::size_t>(samples, 2),
                                        0, SushiRuntime::span<FittedSample>(&fitted, 1)),
                      StatusCode::InvalidDescription, "2 samples", "1 fitted");
}

/** @brief Tests that the training samples have the training size and three channels. */
TEST(Unit_DatasetsAugmentPipeline, MakesSquareSamplesOfThreeChannels)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_pipeline_shape");
    const SampleList list = four_images(scratch);
    const std::unique_ptr<AugmentPipeline> pipeline = augment(1000);
    ASSERT_NE(pipeline, nullptr);
    EXPECT_EQ(pipeline->name(), "default");

    for (const FittedSample& sample : produce(*pipeline, list, {0, 1, 2, 3}, 0))
    {
        EXPECT_EQ(sample.image.size(), (ImageSize{K_SIDE, K_SIDE}));
        EXPECT_EQ(sample.image.channels(), 3);
        EXPECT_EQ(sample.mask.byte_count(), 0u);

        for (const Vision::Imaging::PixelBox& box : sample.objects.boxes())
        {
            EXPECT_GE(box.x1, 0.0f);
            EXPECT_LE(box.x2, static_cast<float>(K_SIDE));
            EXPECT_GE(box.y1, 0.0f);
            EXPECT_LE(box.y2, static_cast<float>(K_SIDE));
        }
    }
}

/** @brief Tests that a sample depends on its index and its epoch, not on its batch. */
TEST(Unit_DatasetsAugmentPipeline, ASampleDependsOnItsIndexAndItsEpochAlone)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_pipeline_replay");
    const SampleList list = four_images(scratch);
    const std::unique_ptr<AugmentPipeline> pipeline = augment(1000);
    ASSERT_NE(pipeline, nullptr);

    const std::vector<FittedSample> whole = produce(*pipeline, list, {0, 1, 2, 3}, 2);
    const std::vector<FittedSample> again = produce(*pipeline, list, {0, 1, 2, 3}, 2);
    const std::vector<FittedSample> alone = produce(*pipeline, list, {2}, 2);
    const std::vector<FittedSample> reversed = produce(*pipeline, list, {3, 2, 1, 0}, 2);

    for (std::size_t row = 0; row < 4; ++row)
    {
        expect_same(whole[row], again[row]);
        expect_same(whole[row], reversed[3 - row]);
    }
    expect_same(whole[2], alone[0]);

    const std::vector<FittedSample> next = produce(*pipeline, list, {0, 1, 2, 3}, 3);
    std::size_t changed = 0;
    for (std::size_t row = 0; row < 4; ++row)
        changed += same_pixels(whole[row].image, next[row].image) ? 0 : 1;

    EXPECT_GT(changed, 0u);
}

/** @brief Tests that past the mosaic epochs a sample holds the objects of its own image. */
TEST(Unit_DatasetsAugmentPipeline, PastTheMosaicEpochsASampleHoldsItsOwnObjects)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_pipeline_closed");
    const SampleList list = four_images(scratch);
    const std::unique_ptr<AugmentPipeline> pipeline = augment(0);
    ASSERT_NE(pipeline, nullptr);

    for (std::int64_t epoch = 0; epoch < 3; ++epoch)
    {
        const std::vector<FittedSample> made = produce(*pipeline, list, {0, 1, 2, 3}, epoch);
        for (std::size_t row = 0; row < 4; ++row)
        {
            for (std::int32_t class_index : made[row].objects.classes())
                EXPECT_EQ(class_index, static_cast<std::int32_t>(row)) << "epoch " << epoch;
        }
    }
}

/** @brief Tests that a mask comes out at the training size holding classes or the void byte. */
TEST(Unit_DatasetsAugmentPipeline, AMaskTakesTheSizeOfItsSample)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_pipeline_masks");
    SampleList list = four_images(scratch);
    add_masks(scratch, list);
    const std::unique_ptr<AugmentPipeline> pipeline = augment(1000);
    ASSERT_NE(pipeline, nullptr);

    for (const FittedSample& sample : produce(*pipeline, list, {0, 1, 2, 3}, 1))
    {
        ASSERT_EQ(sample.mask.size(), (ImageSize{K_SIDE, K_SIDE}));
        ASSERT_EQ(sample.mask.channels(), 1);

        for (std::size_t i = 0; i < sample.mask.byte_count(); ++i)
        {
            const std::uint8_t value = sample.mask.data()[i];
            EXPECT_TRUE(value < 4 || value == K_MASK_VOID) << "byte " << i << " is "
                                                           << static_cast<int>(value);
        }
    }
}

/** @brief Tests that an image that cannot be read is the failure, with its file. */
TEST(Unit_DatasetsAugmentPipeline, AnImageThatCannotBeReadIsTheFailure)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_pipeline_missing");
    SampleList list = four_images(scratch);
    list.files[1] = scratch.path() / "gone.saim";
    const std::unique_ptr<AugmentPipeline> pipeline = augment(0);
    ASSERT_NE(pipeline, nullptr);

    const std::size_t samples[] = {1};
    FittedSample fitted;
    SA_EXPECT_FAILURE(pipeline->produce(list, SushiRuntime::span<const std::size_t>(samples, 1),
                                        0, SushiRuntime::span<FittedSample>(&fitted, 1)),
                      StatusCode::FileNotFound, "gone.saim");
}

/** @brief Tests each refusal: a size, an epoch count, an epoch and a list out of range. */
TEST(Unit_DatasetsAugmentPipeline, RefusesWhatItCannotMake)
{
    AugmentOptions options;
    options.image_size = 0;
    SA_EXPECT_FAILURE(AugmentPipeline::make(options), StatusCode::InvalidDescription,
                      "AugmentPipeline", "a training size of 0");

    options.image_size = K_SIDE;
    options.mosaic_epochs = -1;
    SA_EXPECT_FAILURE(AugmentPipeline::make(options), StatusCode::InvalidDescription,
                      "-1 epochs of mosaic");

    const std::unique_ptr<AugmentPipeline> pipeline = augment(0);
    ASSERT_NE(pipeline, nullptr);

    const Support::ScratchDirectory scratch("sushiai_test_datasets_pipeline_refusals");
    const SampleList list = four_images(scratch);
    const std::size_t samples[] = {0};
    FittedSample fitted;
    const SushiRuntime::span<const std::size_t> one(samples, 1);

    SA_EXPECT_FAILURE(pipeline->produce(list, one, -1,
                                        SushiRuntime::span<FittedSample>(&fitted, 1)),
                      StatusCode::InvalidDescription, "epoch -1");
    SA_EXPECT_FAILURE(pipeline->produce(SampleList(), one, 0,
                                        SushiRuntime::span<FittedSample>(&fitted, 1)),
                      StatusCode::InvalidDescription, "a list of 0 samples");
}
