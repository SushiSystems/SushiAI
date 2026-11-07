/**************************************************************************/
/* test_mask_pipeline.cpp                                                 */
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
 * @file test_mask_pipeline.cpp
 * @brief Tests the nearest affine warp and the pipeline that replays a training trace on a mask.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/affine_nearest.hpp>
#include <SushiAI/vision/imaging/dataset_partners.hpp>
#include <SushiAI/vision/imaging/in_memory_mask_source.hpp>
#include <SushiAI/vision/imaging/in_memory_sample_source.hpp>
#include <SushiAI/vision/imaging/mask_pipeline.hpp>
#include <SushiAI/vision/imaging/pipeline.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Imaging;

namespace
{
    using SushiAI::Test::require_value;

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

    /** @brief Returns an image holding exactly the listed bytes, in order. */
    Image make_image_of(ImageSize size, std::int32_t channels,
                        std::initializer_list<std::uint8_t> bytes)
    {
        Image image(size, channels);
        std::size_t offset = 0;
        for (const std::uint8_t value : bytes)
        {
            image.data()[offset] = value;
            ++offset;
        }

        return image;
    }

    /** @brief Returns the image with the order of the pixels in each row reversed. */
    Image mirrored(const Image& image)
    {
        Image result(image.size(), image.channels());
        for (std::int32_t y = 0; y < image.height(); ++y)
            for (std::int32_t x = 0; x < image.width(); ++x)
                for (std::int32_t c = 0; c < image.channels(); ++c)
                    result.row(y)[x * image.channels() + c] =
                        image.row(y)[(image.width() - 1 - x) * image.channels() + c];

        return result;
    }

    /** @brief Checks that two images have the same size, channels and bytes. */
    void expect_same_image(const Image& actual, const Image& expected)
    {
        ASSERT_EQ(actual.size(), expected.size());
        ASSERT_EQ(actual.channels(), expected.channels());
        for (std::size_t i = 0; i < expected.byte_count(); ++i)
            ASSERT_EQ(actual.data()[i], expected.data()[i]) << "byte " << i;
    }

    /** @brief Checks that an image holds exactly the listed bytes, in order. */
    void expect_bytes(const Image& image, std::initializer_list<std::uint8_t> expected)
    {
        ASSERT_EQ(image.byte_count(), expected.size());
        std::size_t offset = 0;
        for (const std::uint8_t value : expected)
        {
            EXPECT_EQ(image.data()[offset], value) << "byte " << offset;
            ++offset;
        }
    }

    /**
     * @brief Checks that every pixel well inside a box of the sample holds the box's class.
     * @return The number of pixels compared.
     */
    std::size_t expect_box_interiors_hold_class(const Image& mask, const Sample& sample,
                                                std::int32_t side)
    {
        std::size_t compared = 0;
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

            for (std::int32_t y = first_y; y <= last_y && y < side; ++y)
                for (std::int32_t x = first_x; x <= last_x && x < side; ++x, ++compared)
                    EXPECT_EQ(mask.row(y)[x], sample.labels.classes()[object])
                        << "pixel " << y << ", " << x;
        }

        return compared;
    }

    /** @brief Adds an RGB image of a size, with one box of its own index over all of it. */
    void add_whole_image_sample(InMemorySampleSource& images, ImageSize size)
    {
        FileLabels labels;
        labels.add(static_cast<std::int32_t>(images.size()),
                   NormalizedBox{0.5f, 0.5f, 1.0f, 1.0f});
        images.add(uniform(size, K_RGB_CHANNELS, 90), labels);
    }

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

    /** @brief Holds a sample of an identity pipeline and the trace that made it. */
    struct TracedSample
    {
        /** @brief Holds the sample. */
        Sample sample;

        /** @brief Holds the trace of the draw. */
        TrainingTrace trace;
    };

    /** @brief Draws sample 0 of a one-image set of a side under settings, and keeps its trace. */
    TracedSample make_traced_sample(std::int32_t side, const TrainingSettings& settings)
    {
        InMemorySampleSource images;
        add_whole_image_sample(images, ImageSize{side, side});
        TrainingPipeline pipeline = require_value(
            TrainingPipeline::make(images, settings, require_value(DatasetPartners::make(1))));

        TracedSample traced;
        traced.trace = require_value(pipeline.make_sample(0, 0, traced.sample));
        return traced;
    }
} // namespace

/** @brief Tests that a scale of 2 repeats pixels, rounding a half up, and blends none. */
TEST(Small_ImagingAffineNearest, AScaleRepeatsPixelsAndBlendsNone)
{
    const Image source = make_image_of(ImageSize{1, 4}, 1, {3, 50, 7, 120});

    Image warped;
    SA_EXPECT_OK(apply_affine_nearest(
        source.view(), make_plan(ImageSize{1, 4}, ImageSize{1, 8}, 2.0f, 0.0f, 0.0f), warped));

    expect_bytes(warped, {3, 50, 50, 7, 7, 120, 120, K_VOID});
}

/** @brief Tests that every channel of a pixel moves together. */
TEST(Small_ImagingAffineNearest, EveryChannelOfAPixelMovesTogether)
{
    const Image source = make_image_of(ImageSize{1, 2}, K_RGB_CHANNELS, {1, 2, 3, 4, 5, 6});

    Image warped;
    SA_EXPECT_OK(apply_affine_nearest(
        source.view(), make_plan(ImageSize{1, 2}, ImageSize{1, 2}, 1.0f, 1.0f, 0.0f), warped));

    expect_bytes(warped, {K_VOID, K_VOID, K_VOID, 1, 2, 3});
}

/** @brief Tests that a view without pixels and a plan of another source size are refused. */
TEST(Small_ImagingAffineNearest, RefusesAViewWithoutPixelsAndAPlanOfAnotherSize)
{
    const ImageSize size{2, 2};
    const Image source = counting_mask(size);
    Image warped;

    SA_EXPECT_FAILURE(apply_affine_nearest(ImageView(), make_plan(size, size, 1.0f, 0.0f, 0.0f),
                                           warped),
                      StatusCode::InvalidDescription, "holds no image");
    SA_EXPECT_FAILURE(apply_affine_nearest(source.view(),
                                           make_plan(ImageSize{3, 3}, size, 1.0f, 0.0f, 0.0f),
                                           warped),
                      StatusCode::Incompatible, "3x3", "2x2");
}

/** @brief Tests that a destination that shares the source's pixels is refused. */
TEST(Small_ImagingAffineNearest, RefusesADestinationThatSharesThePixelsOfTheSource)
{
    const ImageSize size{2, 2};
    Image source = counting_mask(size);

    SA_EXPECT_FAILURE(apply_affine_nearest(source.view(), make_plan(size, size, 1.0f, 0.0f, 0.0f),
                                           source),
                      StatusCode::InvalidDescription, "share pixels");
}

/** @brief Tests that a matrix with an off-diagonal term is refused. */
TEST(Small_ImagingAffineNearest, RefusesAShearedMatrix)
{
    const ImageSize size{2, 2};
    const Image source = counting_mask(size);
    Image warped;
    AffinePlan sheared = make_plan(size, size, 1.0f, 0.0f, 0.0f);
    sheared.matrix[1] = 0.5f;

    SA_EXPECT_FAILURE(apply_affine_nearest(source.view(), sheared, warped),
                      StatusCode::InvalidDescription, "off-diagonal");
}

/** @brief Tests that a flipped sample's mask is its mask mirrored, with nothing resampled. */
TEST(Small_ImagingMaskPipeline, AFlippedSampleHasItsMaskMirrored)
{
    constexpr std::int32_t K_SIDE = 8;
    const ImageSize size{K_SIDE, K_SIDE};
    InMemoryMaskSource masks;
    masks.add(counting_mask(size));
    TrainingSettings settings = identity_settings(K_SIDE);
    settings.flip.probability = 1.0f;
    const TracedSample traced = make_traced_sample(K_SIDE, settings);
    ASSERT_TRUE(traced.trace.flip.is_flipped);
    ASSERT_FALSE(traced.trace.mosaic.is_enabled);

    TrainingMaskPipeline mask_pipeline(masks, settings, K_VOID);
    Image mask;
    SA_EXPECT_OK(mask_pipeline.make_mask(0, traced.trace, traced.sample.loaded_size, mask));

    expect_same_image(mask, mirrored(counting_mask(size)));
}

/** @brief Tests that inside every box a mosaic sample keeps, the mask holds the box's class. */
TEST(Small_ImagingMaskPipeline, AMosaicMaskFollowsTheBoxesOfItsSample)
{
    constexpr std::int32_t K_SIDE = 32;
    const ImageSize sizes[] = {{40, 30}, {30, 40}, {32, 32}, {20, 40}};

    InMemorySampleSource images;
    InMemoryMaskSource masks;
    for (std::size_t i = 0; i < 4; ++i)
    {
        add_whole_image_sample(images, sizes[i]);
        masks.add(uniform(sizes[i], 1, static_cast<std::uint8_t>(i)));
    }

    TrainingSettings settings;
    settings.image_size = K_SIDE;
    settings.seed = K_SEED;

    TrainingPipeline pipeline =
        require_value(TrainingPipeline::make(images, settings,
                                             require_value(DatasetPartners::make(4))));
    TrainingMaskPipeline mask_pipeline(masks, settings, K_VOID);

    std::size_t compared = 0;
    for (std::int32_t epoch = 0; epoch < 3; ++epoch)
    {
        for (std::size_t index = 0; index < 4; ++index)
        {
            Sample sample;
            const TrainingTrace trace = require_value(pipeline.make_sample(index, epoch, sample));
            ASSERT_TRUE(trace.mosaic.is_enabled);

            Image mask;
            SA_EXPECT_OK(mask_pipeline.make_mask(index, trace, sample.loaded_size, mask));
            ASSERT_EQ(mask.size(), (ImageSize{K_SIDE, K_SIDE}));
            ASSERT_EQ(mask.channels(), 1);

            compared += expect_box_interiors_hold_class(mask, sample, K_SIDE);
        }
    }

    EXPECT_GT(compared, 0u);
}

/** @brief Tests that a mask with several channels is refused. */
TEST(Small_ImagingMaskPipeline, RefusesAMaskOfSeveralChannels)
{
    constexpr std::int32_t K_SIDE = 8;
    const TrainingSettings settings = identity_settings(K_SIDE);
    const TracedSample traced = make_traced_sample(K_SIDE, settings);
    InMemoryMaskSource colour;
    colour.add(uniform(ImageSize{K_SIDE, K_SIDE}, K_RGB_CHANNELS, 0));
    TrainingMaskPipeline mask_pipeline(colour, settings, K_VOID);

    Image mask;
    SA_EXPECT_FAILURE(mask_pipeline.make_mask(0, traced.trace, traced.sample.loaded_size, mask),
                      StatusCode::Incompatible, "3 channels");
}

/** @brief Tests that a mask of another size than its image is refused, naming both sizes. */
TEST(Small_ImagingMaskPipeline, RefusesAMaskOfAnotherSizeThanItsImage)
{
    constexpr std::int32_t K_SIDE = 8;
    const TrainingSettings settings = identity_settings(K_SIDE);
    const TracedSample traced = make_traced_sample(K_SIDE, settings);
    InMemoryMaskSource narrow;
    narrow.add(uniform(ImageSize{K_SIDE, 4}, 1, 0));
    TrainingMaskPipeline mask_pipeline(narrow, settings, K_VOID);

    Image mask;
    SA_EXPECT_FAILURE(mask_pipeline.make_mask(0, traced.trace, traced.sample.loaded_size, mask),
                      StatusCode::Incompatible, "8x4", "8x8");
}

/** @brief Tests that a mask that fails to load returns the failure of its source. */
TEST(Small_ImagingMaskPipeline, ReturnsTheFailureOfAMaskThatCannotBeLoaded)
{
    constexpr std::int32_t K_SIDE = 8;
    const TrainingSettings settings = identity_settings(K_SIDE);
    const TracedSample traced = make_traced_sample(K_SIDE, settings);
    InMemoryMaskSource failing;
    failing.add(uniform(ImageSize{K_SIDE, K_SIDE}, 1, 0));
    failing.fail_load_of(0, Status::from_message(StatusCode::InvalidFile, []()
    {
        return std::string("mask 0 is unreadable");
    }));
    TrainingMaskPipeline mask_pipeline(failing, settings, K_VOID);

    Image mask;
    SA_EXPECT_FAILURE(mask_pipeline.make_mask(0, traced.trace, traced.sample.loaded_size, mask),
                      StatusCode::InvalidFile, "mask 0 is unreadable");
}
