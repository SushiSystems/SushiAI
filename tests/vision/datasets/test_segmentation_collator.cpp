/**************************************************************************/
/* test_segmentation_collator.cpp                                         */
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
 * @file test_segmentation_collator.cpp
 * @brief Tests the segmentation collator.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "datasets_test_support.hpp"
#include <SushiAI/vision/imaging/polygon.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/binary_collator.hpp>
#include <SushiAI/vision/datasets/fit_mask.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/multi_label_collator.hpp>
#include <SushiAI/vision/datasets/regression_collator.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/segmentation_collator.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/image.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
using Vision::Imaging::ImageSize;
using Vision::Imaging::PixelBox;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the size of every image and mask of these tests. */
    constexpr ImageSize K_SIZE{1, 2};

    /** @brief Provides an engine, a two-file list and the samples of one batch. */
    class Medium_SegmentationCollator : public SaTest::EngineTest
    {
        protected:
            /** @brief Names the two files and makes one gray sample of K_SIZE per file. */
            void SetUp() override
            {
                SaTest::EngineTest::SetUp();
                list.files = {"first.png", "second.png"};

                samples.resize(2);
                samples[0].image = Support::uniform_image(K_SIZE, 1, 51);
                samples[0].file = 0;
                samples[1].image = Support::uniform_image(K_SIZE, 1, 255);
                samples[1].file = 1;
            }

            /** @brief Returns the samples as the span a collator takes. */
            SushiRuntime::span<const FittedSample> span() const
            {
                return SushiRuntime::span<const FittedSample>(samples.data(), samples.size());
            }

            /** @brief Allocates a batch of two samples of a shape over some roles. */
            Data::Batch allocate(const Shape& sample,
                                 const std::vector<Data::TargetDescription>& targets)
            {
                const Data::InputDescription input{sample, DType::Float32};
                Result<Data::Batch> batch = Data::allocate_batch(*engine, 2, input, targets);
                EXPECT_TRUE(batch.status().is_ok());
                return std::move(batch).value();
            }

            /** @brief Allocates a batch of two gray images of K_SIZE over some roles. */
            Data::Batch allocate_images(const std::vector<Data::TargetDescription>& targets)
            {
                return allocate(Shape{K_SIZE.height, K_SIZE.width, 1}, targets);
            }

            /** @brief Holds the files the samples name. */
            SampleList list;

            /** @brief Holds the fitted samples of the batch. */
            std::vector<FittedSample> samples;
    };

} // namespace

/** @brief Tests that segmentation publishes mask as [H, W] int32 over the classes. */
TEST_F(Medium_SegmentationCollator, SegmentationPublishesTheMask)
{
    SegmentationOptions options;
    options.size = K_SIZE;
    SA_EXPECT_FAILURE(SegmentationCollator::make(options), StatusCode::InvalidDescription,
                      "SegmentationCollator", "0 classes");

    options.class_count = 256;
    SA_EXPECT_FAILURE(SegmentationCollator::make(options), StatusCode::InvalidDescription,
                      "256 classes", "1 to 255");

    options.class_count = 3;
    options.ignore_index = 2;
    SA_EXPECT_FAILURE(SegmentationCollator::make(options), StatusCode::InvalidDescription,
                      "the ignore index 2 is one of the 3 classes");

    options.ignore_index = K_DEFAULT_IGNORE_INDEX;
    options.size = ImageSize{0, 2};
    SA_EXPECT_FAILURE(SegmentationCollator::make(options), StatusCode::InvalidDescription,
                      "0 by 2 pixels");

    options.size = K_SIZE;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SegmentationCollator> collator,
                            SegmentationCollator::make(options));

    EXPECT_EQ(collator->task(), K_TASK_SEGMENTATION);
    ASSERT_EQ(collator->targets().size(), 1u);
    EXPECT_EQ(collator->targets()[0].role, Data::K_ROLE_MASK);
    EXPECT_EQ(collator->targets()[0].sample_shape, (Shape{K_SIZE.height, K_SIZE.width}));
    EXPECT_EQ(collator->targets()[0].dtype, DType::Int32);
    EXPECT_EQ(collator->targets()[0].classes, 3);
}

/** @brief Tests that segmentation writes each class and the ignore index for the void byte. */
TEST_F(Medium_SegmentationCollator, SegmentationWritesClassesAndTheIgnoreIndex)
{
    samples[0].mask = Support::gray_image(K_SIZE, {2, K_MASK_VOID});
    samples[1].mask = Support::gray_image(K_SIZE, {0, 1});

    SegmentationOptions options;
    options.class_count = 3;
    options.size = K_SIZE;

    for (const std::int32_t ignore : {K_DEFAULT_IGNORE_INDEX, -100})
    {
        options.ignore_index = ignore;
        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SegmentationCollator> collator,
                                SegmentationCollator::make(options));
        Data::Batch batch = allocate_images(collator->targets());
        SA_EXPECT_OK(collator->collate(list, span(), batch));

        const std::int32_t* mask = batch.targets[0].values.data_as<std::int32_t>();
        EXPECT_EQ(std::vector<std::int32_t>(mask, mask + 4),
                  (std::vector<std::int32_t>{2, ignore, 0, 1}));
    }
}

/** @brief Tests that segmentation names the file and the pixel of a byte that is no class. */
TEST_F(Medium_SegmentationCollator, SegmentationRefusesAByteThatIsNoClass)
{
    samples[0].mask = Support::gray_image(K_SIZE, {0, 1});
    samples[1].mask = Support::gray_image(K_SIZE, {0, 3});

    SegmentationOptions options;
    options.class_count = 3;
    options.size = K_SIZE;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SegmentationCollator> collator,
                            SegmentationCollator::make(options));
    Data::Batch batch = allocate_images(collator->targets());

    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "second.png", "SegmentationCollator", "holds 3 at row 0, column 1",
                      "below 3 and 255 names none");
}

/** @brief Tests that segmentation names the file of a sample with no mask of the input size. */
TEST_F(Medium_SegmentationCollator, SegmentationRefusesASampleWithoutAMask)
{
    samples[0].mask = Support::gray_image(K_SIZE, {0, 1});

    SegmentationOptions options;
    options.class_count = 3;
    options.size = K_SIZE;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SegmentationCollator> collator,
                            SegmentationCollator::make(options));
    Data::Batch batch = allocate_images(collator->targets());

    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "second.png", "SegmentationCollator", "the sample's mask is 0 by 0");
}

/** @brief Tests that segmentation paints the polygons of a sample that has no mask file. */
TEST_F(Medium_SegmentationCollator, SegmentationPaintsThePolygonsOfASample)
{
    using Vision::Imaging::Polygon;

    // Each object covers one of the two pixels of K_SIZE: the second, then the first.
    samples[0].objects.add(2, PixelBox{1.0f, 0.0f, 2.0f, 1.0f},
                           Polygon{{1.0f, 0.0f}, {2.0f, 0.0f}, {2.0f, 1.0f}, {1.0f, 1.0f}});
    samples[1].objects.add(0, PixelBox{0.0f, 0.0f, 1.0f, 1.0f},
                           Polygon{{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}});

    SegmentationOptions options;
    options.class_count = 3;
    options.size = K_SIZE;
    options.ignore_index = -1;
    options.origin = MaskOrigin::Polygons;

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SegmentationCollator> ignoring,
                            SegmentationCollator::make(options));
    Data::Batch batch = allocate_images(ignoring->targets());
    SA_EXPECT_OK(ignoring->collate(list, span(), batch));

    const std::int32_t* mask = batch.targets[0].values.data_as<std::int32_t>();
    EXPECT_EQ(std::vector<std::int32_t>(mask, mask + 4), (std::vector<std::int32_t>{-1, 2, 0, -1}));

    options.background = 1;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SegmentationCollator> filling,
                            SegmentationCollator::make(options));
    SA_EXPECT_OK(filling->collate(list, span(), batch));

    mask = batch.targets[0].values.data_as<std::int32_t>();
    EXPECT_EQ(std::vector<std::int32_t>(mask, mask + 4), (std::vector<std::int32_t>{1, 2, 0, 1}));
}

/** @brief Tests that an object with no polygon is refused with the file of its sample. */
TEST_F(Medium_SegmentationCollator, SegmentationRefusesAnObjectWithoutAPolygon)
{
    samples[1].objects.add(0, PixelBox{0.0f, 0.0f, 1.0f, 1.0f});

    SegmentationOptions options;
    options.class_count = 3;
    options.size = K_SIZE;
    options.origin = MaskOrigin::Polygons;

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SegmentationCollator> collator,
                            SegmentationCollator::make(options));
    Data::Batch batch = allocate_images(collator->targets());
    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "second.png", "paint_polygon_mask", "no polygon");
}

/** @brief Tests that a background must be a class and is stated with polygons alone. */
TEST_F(Medium_SegmentationCollator, SegmentationRefusesABackgroundItCannotWrite)
{
    SegmentationOptions options;
    options.class_count = 3;
    options.size = K_SIZE;
    options.background = 1;
    SA_EXPECT_FAILURE(SegmentationCollator::make(options), StatusCode::InvalidDescription,
                      "a background of 1", "masks come from files");

    options.origin = MaskOrigin::Polygons;
    options.background = 3;
    SA_EXPECT_FAILURE(SegmentationCollator::make(options), StatusCode::InvalidDescription,
                      "the background 3 is none of the 3 classes");
}
