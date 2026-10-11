/**************************************************************************/
/* test_datasets_task_collators.cpp                                       */
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
 * @file test_datasets_task_collators.cpp
 * @brief Tests the binary, multi-label, regression and segmentation collators.
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
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/vision/datasets/binary_collator.hpp>
#include <SushiAI/vision/datasets/fit_mask.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/multi_label_collator.hpp>
#include <SushiAI/vision/datasets/regression_collator.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/segmentation_collator.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

#include "datasets_test_support.hpp"

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
    class Unit_DatasetsTaskCollators : public SaTest::EngineTest
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

/** @brief Tests that binary publishes labels as one float32 per sample that is no index. */
TEST_F(Unit_DatasetsTaskCollators, BinaryPublishesLabels)
{
    SA_EXPECT_FAILURE(BinaryCollator::make(2), StatusCode::InvalidDescription, "BinaryCollator",
                      "the positive class is 2");
    SA_EXPECT_FAILURE(BinaryCollator::make(-1), StatusCode::InvalidDescription,
                      "the positive class is -1");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<BinaryCollator> collator,
                            BinaryCollator::make(1));

    EXPECT_EQ(collator->task(), K_TASK_BINARY);
    ASSERT_EQ(collator->targets().size(), 1u);
    EXPECT_EQ(collator->targets()[0].role, Data::K_ROLE_LABELS);
    EXPECT_EQ(collator->targets()[0].sample_shape, Shape{});
    EXPECT_EQ(collator->targets()[0].dtype, DType::Float32);
    EXPECT_EQ(collator->targets()[0].classes, 0);
}

/** @brief Tests that binary writes 1 for the positive class, whichever of the two it is. */
TEST_F(Unit_DatasetsTaskCollators, BinaryWritesOneForThePositiveClass)
{
    samples[0].class_index = 1;
    samples[1].class_index = 0;

    for (const std::int32_t positive : {1, 0})
    {
        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<BinaryCollator> collator,
                                BinaryCollator::make(positive));
        Data::Batch batch = allocate_images(collator->targets());
        SA_EXPECT_OK(collator->collate(list, span(), batch));

        // The labels are the constants 0 and 1, so they compare exactly.
        const float* labels = batch.targets[0].values.data_as<float>();
        EXPECT_EQ(labels[0], positive == 1 ? 1.0f : 0.0f);
        EXPECT_EQ(labels[1], positive == 1 ? 0.0f : 1.0f);
    }
}

/** @brief Tests that binary names the file of a sample with no class or a third class. */
TEST_F(Unit_DatasetsTaskCollators, BinaryNamesTheFileOfASampleItCannotLabel)
{
    samples[0].class_index = 1;

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<BinaryCollator> collator,
                            BinaryCollator::make(1));
    Data::Batch batch = allocate_images(collator->targets());
    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "second.png", "BinaryCollator", "no class index");

    samples[1].class_index = 2;
    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "second.png", "class index 2", "[0, 2)");
}

/** @brief Tests that multi-label writes one flag per class a sample's objects name. */
TEST_F(Unit_DatasetsTaskCollators, MultiLabelWritesTheSetOfClassesPresent)
{
    SA_EXPECT_FAILURE(MultiLabelCollator::make(0), StatusCode::InvalidDescription,
                      "MultiLabelCollator", "0 classes");

    samples[0].objects.add(2, PixelBox{0.0f, 0.0f, 1.0f, 1.0f});
    samples[0].objects.add(0, PixelBox{0.0f, 0.0f, 1.0f, 1.0f});
    samples[0].objects.add(2, PixelBox{1.0f, 0.0f, 2.0f, 1.0f});
    samples[0].class_index = 3;

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MultiLabelCollator> collator,
                            MultiLabelCollator::make(4));

    EXPECT_EQ(collator->task(), K_TASK_MULTI_LABEL);
    ASSERT_EQ(collator->targets().size(), 1u);
    EXPECT_EQ(collator->targets()[0].role, Data::K_ROLE_LABELS);
    EXPECT_EQ(collator->targets()[0].sample_shape, Shape{4});
    EXPECT_EQ(collator->targets()[0].dtype, DType::Float32);

    Data::Batch batch = allocate_images(collator->targets());
    std::fill_n(batch.targets[0].values.data_as<float>(), 8, 9.0f);
    SA_EXPECT_OK(collator->collate(list, span(), batch));

    // The second sample has no object and no class, so its row is zeros and not stale.
    const float* labels = batch.targets[0].values.data_as<float>();
    const std::vector<float> expected = {1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    EXPECT_EQ(std::vector<float>(labels, labels + 8), expected);
}

/** @brief Tests that multi-label names the file of a sample with a class outside the set. */
TEST_F(Unit_DatasetsTaskCollators, MultiLabelRefusesAClassOutsideTheSet)
{
    samples[1].objects.add(4, PixelBox{0.0f, 0.0f, 1.0f, 1.0f});

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MultiLabelCollator> collator,
                            MultiLabelCollator::make(4));
    Data::Batch batch = allocate_images(collator->targets());
    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "second.png", "MultiLabelCollator", "the class 4", "[0, 4)");
}

/** @brief Tests that regression copies the features into the inputs and the targets beside. */
TEST_F(Unit_DatasetsTaskCollators, RegressionCopiesFeaturesAndTargets)
{
    SA_EXPECT_FAILURE(RegressionCollator::make(0, 1), StatusCode::InvalidDescription,
                      "RegressionCollator", "0 features");
    SA_EXPECT_FAILURE(RegressionCollator::make(3, 0), StatusCode::InvalidDescription,
                      "RegressionCollator", "0 targets");

    const std::vector<float> rows = {1.0f, 2.0f, 3.0f, 10.0f, 20.0f,
                                     4.0f, 5.0f, 6.0f, 40.0f, 50.0f};
    for (std::size_t i = 0; i < samples.size(); ++i)
    {
        samples[i].image = Vision::Imaging::Image();
        samples[i].features = SushiRuntime::span<const float>(rows.data() + i * 5, 3);
        samples[i].targets = SushiRuntime::span<const float>(rows.data() + i * 5 + 3, 2);
    }

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<RegressionCollator> collator,
                            RegressionCollator::make(3, 2));

    EXPECT_EQ(collator->task(), K_TASK_REGRESSION);
    ASSERT_EQ(collator->targets().size(), 1u);
    EXPECT_EQ(collator->targets()[0].role, Data::K_ROLE_VALUES);
    EXPECT_EQ(collator->targets()[0].sample_shape, Shape{2});
    EXPECT_EQ(collator->targets()[0].dtype, DType::Float32);

    Data::Batch batch = allocate(Shape{3}, collator->targets());
    SA_EXPECT_OK(collator->collate(list, span(), batch));

    // The values are copied and never computed, so they compare exactly.
    const float* inputs = batch.inputs.data_as<float>();
    const float* values = batch.targets[0].values.data_as<float>();
    EXPECT_EQ(std::vector<float>(inputs, inputs + 6),
              (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
    EXPECT_EQ(std::vector<float>(values, values + 4),
              (std::vector<float>{10.0f, 20.0f, 40.0f, 50.0f}));
}

/** @brief Tests that regression refuses a sample of images and inputs of another width. */
TEST_F(Unit_DatasetsTaskCollators, RegressionRefusesWhatHoldsNoRow)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<RegressionCollator> collator,
                            RegressionCollator::make(3, 2));

    Data::Batch batch = allocate(Shape{3}, collator->targets());
    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "first.png", "RegressionCollator", "0 features and 0 targets");

    Data::Batch narrow = allocate(Shape{2}, collator->targets());
    SA_EXPECT_FAILURE(collator->collate(list, span(), narrow), StatusCode::Incompatible,
                      "RegressionCollator", "float32 of [2, 3]");
}

/** @brief Tests that segmentation publishes mask as [H, W] int32 over the classes. */
TEST_F(Unit_DatasetsTaskCollators, SegmentationPublishesTheMask)
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
TEST_F(Unit_DatasetsTaskCollators, SegmentationWritesClassesAndTheIgnoreIndex)
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
TEST_F(Unit_DatasetsTaskCollators, SegmentationRefusesAByteThatIsNoClass)
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
TEST_F(Unit_DatasetsTaskCollators, SegmentationRefusesASampleWithoutAMask)
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
TEST_F(Unit_DatasetsTaskCollators, SegmentationPaintsThePolygonsOfASample)
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
TEST_F(Unit_DatasetsTaskCollators, SegmentationRefusesAnObjectWithoutAPolygon)
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
TEST_F(Unit_DatasetsTaskCollators, SegmentationRefusesABackgroundItCannotWrite)
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
