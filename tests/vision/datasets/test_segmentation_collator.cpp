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
 * @file test_segmentation_collator.cpp
 * @brief Tests the segmentation collator: its refusals, its mask role and the masks it writes.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/data/roles.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/vision/datasets/fit_mask.hpp>
#include <SushiAI/vision/datasets/segmentation_collator.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <support/test_names.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
using Vision::Imaging::ImageSize;
using Vision::Imaging::PixelBox;
using Vision::Imaging::Polygon;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the size of every mask of these tests. */
    constexpr ImageSize K_SIZE = Support::K_BENCH_SIZE;

    /** @brief Provides a host engine, a two-file list and two gray samples. */
    class Small_SegmentationCollator : public Support::CollatorBench
    {
    };

    /** @brief Holds the options a make refuses and the text of the refusal. */
    struct OptionsRefusal
    {
        /** @brief Names the row. */
        const char* name;

        /** @brief Holds the options. */
        SegmentationOptions options;

        /** @brief Holds a text the refusal carries. */
        const char* message;

        /** @brief Holds another text the refusal carries, or none. */
        const char* detail;
    };

    /** @brief Holds one refused make per row. */
    class Small_SegmentationCollatorMakeRefusal : public ::testing::TestWithParam<OptionsRefusal>
    {
    };

    /** @brief Holds the ignore index of a row. */
    class Small_SegmentationCollatorIgnore
        : public Support::CollatorBench,
          public ::testing::WithParamInterface<std::int32_t>
    {
    };

    /** @brief Returns options of 3 classes and the size K_SIZE, with a change made by a caller. */
    SegmentationOptions three_classes()
    {
        SegmentationOptions options;
        options.class_count = 3;
        options.size = K_SIZE;
        return options;
    }

    /** @brief Returns options of the given class count, ignore index and size. */
    SegmentationOptions with(std::int32_t classes, std::int32_t ignore, ImageSize size)
    {
        SegmentationOptions options;
        options.class_count = classes;
        options.ignore_index = ignore;
        options.size = size;
        return options;
    }

    /** @brief Returns options of polygon masks with a background class. */
    SegmentationOptions polygons(std::optional<std::int32_t> background)
    {
        SegmentationOptions options = three_classes();
        options.ignore_index = -1;
        options.origin = MaskOrigin::Polygons;
        options.background = background;
        return options;
    }

    /** @brief Returns options of a file mask with a background class, which they cannot write. */
    SegmentationOptions file_masks_with_background()
    {
        SegmentationOptions options = three_classes();
        options.background = 1;
        return options;
    }

    /** @brief Returns options of polygon masks whose background is none of the classes. */
    SegmentationOptions background_outside_the_classes()
    {
        SegmentationOptions options = three_classes();
        options.origin = MaskOrigin::Polygons;
        options.background = 3;
        return options;
    }

    /** @brief Gives the samples one polygon each, on the second and then the first pixel. */
    void add_polygons(std::vector<FittedSample>& samples)
    {
        samples[0].objects.add(2, PixelBox{1.0f, 0.0f, 2.0f, 1.0f},
                               Polygon{{1.0f, 0.0f}, {2.0f, 0.0f}, {2.0f, 1.0f}, {1.0f, 1.0f}});
        samples[1].objects.add(0, PixelBox{0.0f, 0.0f, 1.0f, 1.0f},
                               Polygon{{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}});
    }
} // namespace

/** @brief Tests that make refuses options it cannot serve. */
TEST_P(Small_SegmentationCollatorMakeRefusal, MakeRefusesOptionsItCannotServe)
{
    const OptionsRefusal& row = GetParam();

    SA_EXPECT_FAILURE(SegmentationCollator::make(row.options), StatusCode::InvalidDescription,
                      row.message, row.detail);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_SegmentationCollatorMakeRefusal,
    ::testing::Values(
        OptionsRefusal{"ZeroClasses", with(0, K_DEFAULT_IGNORE_INDEX, K_SIZE),
                       "SegmentationCollator", "0 classes"},
        OptionsRefusal{"TooManyClasses", with(256, K_DEFAULT_IGNORE_INDEX, K_SIZE),
                       "256 classes", "1 to 255"},
        OptionsRefusal{"AnIgnoreIndexThatIsAClass", with(3, 2, K_SIZE),
                       "the ignore index 2 is one of the 3 classes", ""},
        OptionsRefusal{"ASizeOfNoPixel", with(3, K_DEFAULT_IGNORE_INDEX, ImageSize{0, 2}),
                       "0 by 2 pixels", ""},
        OptionsRefusal{"ABackgroundOfAFileMask", file_masks_with_background(),
                       "a background of 1", "masks come from files"},
        OptionsRefusal{"ABackgroundThatIsNoClass", background_outside_the_classes(),
                       "the background 3 is none of the 3 classes", ""}),
    [](const ::testing::TestParamInfo<OptionsRefusal>& info) { return ::SushiAI::Test::parameter_name(info.param.name); });

/** @brief Tests that segmentation publishes mask as [H, W] int32 over the classes. */
TEST(Small_SegmentationCollatorTargets, SegmentationPublishesTheMask)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SegmentationCollator> collator,
                            SegmentationCollator::make(three_classes()));

    EXPECT_EQ(collator->task(), K_TASK_SEGMENTATION);
    ASSERT_EQ(collator->targets().size(), 1u);
    EXPECT_EQ(collator->targets()[0].role, Data::K_ROLE_MASK);
    EXPECT_EQ(collator->targets()[0].sample_shape, (Shape{K_SIZE.height, K_SIZE.width}));
    EXPECT_EQ(collator->targets()[0].dtype, DType::Int32);
    EXPECT_EQ(collator->targets()[0].classes, 3);
}

/** @brief Tests that segmentation writes each class and the ignore index for the void byte. */
TEST_P(Small_SegmentationCollatorIgnore, SegmentationWritesClassesAndTheIgnoreIndex)
{
    const std::int32_t ignore = GetParam();
    samples[0].mask = Support::gray_image(K_SIZE, {2, K_MASK_VOID});
    samples[1].mask = Support::gray_image(K_SIZE, {0, 1});
    SegmentationOptions options = three_classes();
    options.ignore_index = ignore;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SegmentationCollator> collator,
                            SegmentationCollator::make(options));
    Data::Batch batch = allocate_images(collator->targets());

    SA_EXPECT_OK(collator->collate(list, span(), batch));

    EXPECT_EQ(SaTest::read_tensor(batch.targets[0].values),
              (std::vector<double>{2.0, static_cast<double>(ignore), 0.0, 1.0}));
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_SegmentationCollatorIgnore,
                         ::testing::Values(K_DEFAULT_IGNORE_INDEX, -100),
                         [](const ::testing::TestParamInfo<std::int32_t>& info)
                         {
                             return info.param < 0 ? "Minus" + std::to_string(-info.param)
                                                   : "Index" + std::to_string(info.param);
                         });

/** @brief Tests that segmentation names the file and the pixel of a byte that is no class. */
TEST_F(Small_SegmentationCollator, SegmentationRefusesAByteThatIsNoClass)
{
    samples[0].mask = Support::gray_image(K_SIZE, {0, 1});
    samples[1].mask = Support::gray_image(K_SIZE, {0, 3});
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SegmentationCollator> collator,
                            SegmentationCollator::make(three_classes()));
    Data::Batch batch = allocate_images(collator->targets());

    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "second.png", "SegmentationCollator", "holds 3 at row 0, column 1",
                      "below 3 and 255 names none");
}

/** @brief Tests that segmentation names the file of a sample with no mask of the input size. */
TEST_F(Small_SegmentationCollator, SegmentationRefusesASampleWithoutAMask)
{
    samples[0].mask = Support::gray_image(K_SIZE, {0, 1});
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SegmentationCollator> collator,
                            SegmentationCollator::make(three_classes()));
    Data::Batch batch = allocate_images(collator->targets());

    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "second.png", "SegmentationCollator", "the sample's mask is 0 by 0");
}

/** @brief Tests that segmentation paints the polygons of a sample and ignores the rest. */
TEST_F(Small_SegmentationCollator, SegmentationPaintsThePolygonsOfASample)
{
    add_polygons(samples);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SegmentationCollator> collator,
                            SegmentationCollator::make(polygons(std::nullopt)));
    Data::Batch batch = allocate_images(collator->targets());

    SA_EXPECT_OK(collator->collate(list, span(), batch));

    EXPECT_EQ(SaTest::read_tensor(batch.targets[0].values),
              (std::vector<double>{-1.0, 2.0, 0.0, -1.0}));
}

/** @brief Tests that segmentation fills what no polygon covers with the background class. */
TEST_F(Small_SegmentationCollator, SegmentationPaintsTheBackgroundBehindThePolygons)
{
    add_polygons(samples);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SegmentationCollator> collator,
                            SegmentationCollator::make(polygons(1)));
    Data::Batch batch = allocate_images(collator->targets());

    SA_EXPECT_OK(collator->collate(list, span(), batch));

    EXPECT_EQ(SaTest::read_tensor(batch.targets[0].values),
              (std::vector<double>{1.0, 2.0, 0.0, 1.0}));
}

/** @brief Tests that an object with no polygon is refused with the file of its sample. */
TEST_F(Small_SegmentationCollator, SegmentationRefusesAnObjectWithoutAPolygon)
{
    samples[1].objects.add(0, PixelBox{0.0f, 0.0f, 1.0f, 1.0f});
    SegmentationOptions options = three_classes();
    options.origin = MaskOrigin::Polygons;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SegmentationCollator> collator,
                            SegmentationCollator::make(options));
    Data::Batch batch = allocate_images(collator->targets());

    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "second.png", "paint_polygon_mask", "no polygon");
}
