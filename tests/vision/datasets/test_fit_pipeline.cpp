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
