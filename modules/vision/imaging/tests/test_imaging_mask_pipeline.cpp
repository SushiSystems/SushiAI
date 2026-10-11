/**************************************************************************/
/* test_imaging_mask_pipeline.cpp                                         */
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
 * @file test_imaging_mask_pipeline.cpp
 * @brief Tests the nearest affine warp and the pipeline that replays a training trace on a mask.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <SushiAI/vision/imaging/affine_nearest.hpp>
#include <SushiAI/vision/imaging/dataset_partners.hpp>
#include <SushiAI/vision/imaging/mask_pipeline.hpp>
#include <SushiAI/vision/imaging/pipeline.hpp>

#include <common/test_status.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Imaging;

namespace
{
    /** @brief Holds the byte the tests give a pixel no mask covers. */
    constexpr std::uint8_t K_VOID = 255;

    /** @brief Holds the seed of every pipeline below. */
    constexpr std::uint64_t K_SEED = 20261011;

    /** @brief Holds how far inside a box, in pixels, a mask pixel is compared with it. */
    constexpr float K_BOX_MARGIN = 2.0f;

    /** @brief Returns a plan that scales by one factor and translates, with a border value. */
    AffinePlan make_plan(ImageSize source, ImageSize output, float scale, float shift_x,
                         float shift_y)
    {
        AffinePlan plan;
        plan.source = source;
        plan.output = output;
        plan.scale = scale;
        plan.matrix = {scale, 0.0f, shift_x, 0.0f, scale, shift_y};
        plan.border_value = K_VOID;
        return plan;
    }

    /** @brief Returns a one-channel image whose pixel (x, y) holds (y * width + x) mod 200. */
    Image counting_mask(ImageSize size)
    {
        Image mask(size, 1);
        for (std::size_t i = 0; i < mask.byte_count(); ++i)
            mask.data()[i] = static_cast<std::uint8_t>(i % 200);

        return mask;
    }

    /** @brief Returns an image whose every byte is one value. */
    Image uniform(ImageSize size, std::int32_t channels, std::uint8_t value)
    {
        Image image(size, channels);
        image.fill(value);
        return image;
    }

    /** @brief Supplies grey RGB images, each with one box of its own index over all of it. */
    class WholeImageSource final : public SampleSource
    {
        public:
            /** @brief Appends an image of a size; its class is its index. */
            void add(ImageSize size) { sizes_.push_back(size); }

            [[nodiscard]] std::size_t size() const noexcept override { return sizes_.size(); }

            void load(std::size_t index, Image& image, FileLabels& labels) const override
            {
                image = uniform(sizes_[index], K_RGB_CHANNELS, 90);
                labels = FileLabels();
                labels.add(static_cast<std::int32_t>(index),
                           NormalizedBox{0.5f, 0.5f, 1.0f, 1.0f});
            }

        private:
            /// @brief Holds the size of each image.
            std::vector<ImageSize> sizes_;
    };

    /** @brief Supplies masks held in memory, or one failure for every index. */
    class MemoryMasks final : public MaskSource
    {
        public:
            /** @brief Appends one mask. */
            void add(Image mask) { masks_.push_back(std::move(mask)); }

            /** @brief Makes every load return a failure. */
            void fail() noexcept { is_failing_ = true; }

            [[nodiscard]] Status load(std::size_t index, Image& mask) const override
            {
                SA_FAIL_IF(is_failing_, StatusCode::InvalidFile, "mask {} is unreadable", index);

                mask = masks_[index];
                return Status();
            }

        private:
            /// @brief Holds the masks, index for index with the images.
            std::vector<Image> masks_;

            /// @brief Tells load() to fail.
            bool is_failing_ = false;
    };

    /** @brief Returns the settings of a square sample with no mosaic and an identity warp. */
    TrainingSettings identity_settings(std::int32_t side)
    {
        TrainingSettings settings;
        settings.image_size = side;
        settings.seed = K_SEED;
        settings.mosaic.probability = 0.0f;
        settings.affine.scale = 0.0f;
        settings.affine.translate = 0.0f;
        settings.flip.probability = 0.0f;
        return settings;
    }
} // namespace

/** @brief Tests that a scale of 1 with no translation copies the source. */
TEST(Unit_ImagingAffineNearest, AnIdentityPlanCopiesTheSource)
{
    const ImageSize size{3, 4};
    const Image source = counting_mask(size);

    Image warped;
    SA_EXPECT_OK(apply_affine_nearest(source.view(), make_plan(size, size, 1.0f, 0.0f, 0.0f),
                                      warped));

    ASSERT_EQ(warped.size(), size);
    ASSERT_EQ(warped.channels(), 1);
    for (std::size_t i = 0; i < source.byte_count(); ++i)
        EXPECT_EQ(warped.data()[i], source.data()[i]) << "byte " << i;
}

/** @brief Tests that a translation moves the pixels and fills what it uncovers. */
TEST(Unit_ImagingAffineNearest, ATranslationFillsWhatItUncoversWithTheBorderValue)
{
    const ImageSize size{2, 4};
    const Image source = counting_mask(size);

    Image warped;
    SA_EXPECT_OK(apply_affine_nearest(source.view(), make_plan(size, size, 1.0f, 2.0f, 1.0f),
                                      warped));

    for (std::int32_t x = 0; x < size.width; ++x)
    {
        EXPECT_EQ(warped.row(0)[x], K_VOID) << "column " << x;
        EXPECT_EQ(warped.row(1)[x], x < 2 ? K_VOID : source.row(0)[x - 2]) << "column " << x;
    }
}

/** @brief Tests that a scale of 2 repeats pixels, rounding a half up, and blends none. */
TEST(Unit_ImagingAffineNearest, AScaleRepeatsPixelsAndBlendsNone)
{
    Image source(ImageSize{1, 4}, 1);
    const std::uint8_t values[] = {3, 50, 7, 120};
    for (std::size_t i = 0; i < 4; ++i)
        source.data()[i] = values[i];

    Image warped;
    SA_EXPECT_OK(apply_affine_nearest(
        source.view(), make_plan(ImageSize{1, 4}, ImageSize{1, 8}, 2.0f, 0.0f, 0.0f), warped));

    const std::uint8_t expected[] = {3, 50, 50, 7, 7, 120, 120, K_VOID};
    for (std::size_t x = 0; x < 8; ++x)
        EXPECT_EQ(warped.data()[x], expected[x]) << "column " << x;
}

/** @brief Tests that every channel of a pixel moves together. */
TEST(Unit_ImagingAffineNearest, EveryChannelOfAPixelMovesTogether)
{
    Image source(ImageSize{1, 2}, K_RGB_CHANNELS);
    const std::uint8_t values[] = {1, 2, 3, 4, 5, 6};
    for (std::size_t i = 0; i < 6; ++i)
        source.data()[i] = values[i];

    Image warped;
    SA_EXPECT_OK(apply_affine_nearest(
        source.view(), make_plan(ImageSize{1, 2}, ImageSize{1, 2}, 1.0f, 1.0f, 0.0f), warped));

    const std::uint8_t expected[] = {K_VOID, K_VOID, K_VOID, 1, 2, 3};
    for (std::size_t i = 0; i < 6; ++i)
        EXPECT_EQ(warped.data()[i], expected[i]) << "byte " << i;
}

/** @brief Tests each refusal: no pixels, another size, shared pixels and a sheared matrix. */
TEST(Unit_ImagingAffineNearest, RefusesWhatItCannotWarp)
{
    const ImageSize size{2, 2};
    Image source = counting_mask(size);
    Image warped;

    SA_EXPECT_FAILURE(apply_affine_nearest(ImageView(), make_plan(size, size, 1.0f, 0.0f, 0.0f),
                                           warped),
                      StatusCode::InvalidDescription, "holds no image");
    SA_EXPECT_FAILURE(apply_affine_nearest(source.view(),
                                           make_plan(ImageSize{3, 3}, size, 1.0f, 0.0f, 0.0f),
                                           warped),
                      StatusCode::Incompatible, "3x3", "2x2");
    SA_EXPECT_FAILURE(apply_affine_nearest(source.view(),
                                           make_plan(size, size, 1.0f, 0.0f, 0.0f), source),
                      StatusCode::InvalidDescription, "share pixels");

    AffinePlan sheared = make_plan(size, size, 1.0f, 0.0f, 0.0f);
    sheared.matrix[1] = 0.5f;
    SA_EXPECT_FAILURE(apply_affine_nearest(source.view(), sheared, warped),
                      StatusCode::InvalidDescription, "off-diagonal");
}

/** @brief Tests that a flipped sample's mask is its mask mirrored, with nothing resampled. */
TEST(Unit_ImagingMaskPipeline, AFlippedSampleHasItsMaskMirrored)
{
    constexpr std::int32_t K_SIDE = 8;
    const ImageSize size{K_SIDE, K_SIDE};

    WholeImageSource images;
    images.add(size);
    MemoryMasks masks;
    masks.add(counting_mask(size));

    TrainingSettings settings = identity_settings(K_SIDE);
    settings.flip.probability = 1.0f;

    TrainingPipeline pipeline(images, settings, std::make_unique<DatasetPartners>(1));
    Sample sample;
    const TrainingTrace trace = pipeline.make_sample(0, 0, sample);
    ASSERT_TRUE(trace.flip.is_flipped);
    ASSERT_FALSE(trace.mosaic.is_enabled);

    TrainingMaskPipeline mask_pipeline(masks, settings, K_VOID);
    Image mask;
    SA_EXPECT_OK(mask_pipeline.make_mask(0, trace, sample.loaded_size, mask));

    const Image original = counting_mask(size);
    ASSERT_EQ(mask.size(), size);
    for (std::int32_t y = 0; y < K_SIDE; ++y)
    {
        for (std::int32_t x = 0; x < K_SIDE; ++x)
            EXPECT_EQ(mask.row(y)[x], original.row(y)[K_SIDE - 1 - x]) << y << ", " << x;
    }
}

/** @brief Tests that inside every box a mosaic sample keeps, the mask holds the box's class. */
TEST(Unit_ImagingMaskPipeline, AMosaicMaskFollowsTheBoxesOfItsSample)
{
    constexpr std::int32_t K_SIDE = 32;
    const ImageSize sizes[] = {{40, 30}, {30, 40}, {32, 32}, {20, 40}};

    WholeImageSource images;
    MemoryMasks masks;
    for (std::size_t i = 0; i < 4; ++i)
    {
        images.add(sizes[i]);
        masks.add(uniform(sizes[i], 1, static_cast<std::uint8_t>(i)));
    }

    TrainingSettings settings;
    settings.image_size = K_SIDE;
    settings.seed = K_SEED;

    TrainingPipeline pipeline(images, settings, std::make_unique<DatasetPartners>(4));
    TrainingMaskPipeline mask_pipeline(masks, settings, K_VOID);

    std::size_t compared = 0;
    for (std::int32_t epoch = 0; epoch < 3; ++epoch)
    {
        for (std::size_t index = 0; index < 4; ++index)
        {
            Sample sample;
            const TrainingTrace trace = pipeline.make_sample(index, epoch, sample);
            ASSERT_TRUE(trace.mosaic.is_enabled);

            Image mask;
            SA_EXPECT_OK(mask_pipeline.make_mask(index, trace, sample.loaded_size, mask));
            ASSERT_EQ(mask.size(), (ImageSize{K_SIDE, K_SIDE}));
            ASSERT_EQ(mask.channels(), 1);

            for (std::size_t object = 0; object < sample.labels.size(); ++object)
            {
                const PixelBox& box = sample.labels.boxes()[object];
                const std::int32_t first_x =
                    static_cast<std::int32_t>(std::ceil(box.x1 + K_BOX_MARGIN));
                const std::int32_t last_x =
                    static_cast<std::int32_t>(std::floor(box.x2 - K_BOX_MARGIN));
                const std::int32_t first_y =
                    static_cast<std::int32_t>(std::ceil(box.y1 + K_BOX_MARGIN));
                const std::int32_t last_y =
                    static_cast<std::int32_t>(std::floor(box.y2 - K_BOX_MARGIN));

                for (std::int32_t y = first_y; y <= last_y && y < K_SIDE; ++y)
                {
                    for (std::int32_t x = first_x; x <= last_x && x < K_SIDE; ++x, ++compared)
                        EXPECT_EQ(mask.row(y)[x], sample.labels.classes()[object])
                            << "epoch " << epoch << ", index " << index << ", pixel " << y
                            << ", " << x;
                }
            }
        }
    }

    EXPECT_GT(compared, 0u);
}

/** @brief Tests each refusal: a failed load, several channels and a mask not of its image. */
TEST(Unit_ImagingMaskPipeline, RefusesAMaskItCannotReplay)
{
    constexpr std::int32_t K_SIDE = 8;
    const ImageSize size{K_SIDE, K_SIDE};
    const TrainingSettings settings = identity_settings(K_SIDE);

    WholeImageSource images;
    images.add(size);
    TrainingPipeline pipeline(images, settings, std::make_unique<DatasetPartners>(1));
    Sample sample;
    const TrainingTrace trace = pipeline.make_sample(0, 0, sample);
    Image mask;

    MemoryMasks colour;
    colour.add(uniform(size, K_RGB_CHANNELS, 0));
    TrainingMaskPipeline colour_pipeline(colour, settings, K_VOID);
    SA_EXPECT_FAILURE(colour_pipeline.make_mask(0, trace, sample.loaded_size, mask),
                      StatusCode::Incompatible, "3 channels");

    MemoryMasks narrow;
    narrow.add(uniform(ImageSize{K_SIDE, 4}, 1, 0));
    TrainingMaskPipeline narrow_pipeline(narrow, settings, K_VOID);
    SA_EXPECT_FAILURE(narrow_pipeline.make_mask(0, trace, sample.loaded_size, mask),
                      StatusCode::Incompatible, "8x4", "8x8");

    MemoryMasks failing;
    failing.fail();
    TrainingMaskPipeline failing_pipeline(failing, settings, K_VOID);
    SA_EXPECT_FAILURE(failing_pipeline.make_mask(0, trace, sample.loaded_size, mask),
                      StatusCode::InvalidFile, "mask 0 is unreadable");
}
