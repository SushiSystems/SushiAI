/**************************************************************************/
/* test_pipeline.cpp                                                      */
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
 * @file test_pipeline.cpp
 * @brief Tests the order, the output size and the reproducibility of the two data pipelines.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/dataset_partners.hpp>
#include <SushiAI/vision/imaging/gray.hpp>
#include <SushiAI/vision/imaging/in_memory_partner_source.hpp>
#include <SushiAI/vision/imaging/in_memory_sample_source.hpp>
#include <SushiAI/vision/imaging/pipeline.hpp>
#include <SushiAI/vision/imaging/resize.hpp>
#include <support/imaging_images.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Imaging;

namespace
{
    using SushiAI::Test::require_value;
    using SushiAI::Test::make_constant;
    using SushiAI::Test::make_flat_image;

    /** @brief Holds the training size of the tests; a mosaic canvas is 64 by 64. */
    constexpr std::int32_t K_SIDE = 32;

    /**
     * @brief Holds one step of the 24-bit uniform over the default scale range of width 1: 2^-24.
     */
    constexpr float K_MIN_SCALE_GAP = 1.0f / 16777216.0f;

    /** @brief Returns a three-channel image whose every byte depends on its row and column. */
    Image make_textured(ImageSize size, std::int32_t salt)
    {
        Image image(size, 3);
        for (std::int32_t y = 0; y < size.height; ++y)
        {
            std::uint8_t* row = image.row(y);
            for (std::int32_t x = 0; x < size.width; ++x)
            {
                for (std::int32_t c = 0; c < 3; ++c)
                    row[3 * x + c] = static_cast<std::uint8_t>(13 * y + 29 * x + 71 * c + salt);
            }
        }

        return image;
    }

    /** @brief Returns the labels of one object of the given class. */
    FileLabels one_object(std::int32_t class_id, NormalizedBox box)
    {
        return FileLabels({class_id}, {box});
    }

    /** @brief Fills a source with five textured images whose sides are powers of two. */
    template <typename Source>
    void fill_source(Source& source)
    {
        source.add(make_textured(ImageSize{32, 64}, 3),
                   one_object(0, NormalizedBox{0.5f, 0.5f, 0.5f, 0.5f}));
        source.add(make_textured(ImageSize{64, 64}, 40),
                   one_object(1, NormalizedBox{0.25f, 0.25f, 0.25f, 0.5f}));
        source.add(make_textured(ImageSize{64, 32}, 90),
                   one_object(2, NormalizedBox{0.5f, 0.75f, 0.75f, 0.25f}));
        source.add(make_textured(ImageSize{16, 32}, 150),
                   one_object(3, NormalizedBox{0.75f, 0.5f, 0.25f, 0.75f}));
        source.add(make_textured(ImageSize{32, 32}, 210), FileLabels{});
    }

    /** @brief Returns the default training settings at the small test size. */
    TrainingSettings make_settings()
    {
        TrainingSettings settings;
        settings.image_size = K_SIDE;
        return settings;
    }

    /** @brief Returns settings whose warp, jitter and flip change nothing, in a closed epoch 0. */
    TrainingSettings make_still_settings()
    {
        TrainingSettings settings = make_settings();
        settings.epochs = 1;
        settings.close_mosaic = 1;
        settings.affine.scale = 0.0f;
        settings.affine.translate = 0.0f;
        settings.hsv.hue_gain = 0.0f;
        settings.hsv.saturation_gain = 0.0f;
        settings.hsv.value_gain = 0.0f;
        settings.flip.probability = 0.0f;
        return settings;
    }

    /** @brief Returns a partner source that draws from every image of a source. */
    std::unique_ptr<PartnerSource> whole_dataset(const SampleSource& source)
    {
        return require_value(DatasetPartners::make(source.size()));
    }

    /** @brief Returns a partner source that gives the same three partners for every sample. */
    std::unique_ptr<PartnerSource> fixed_partners(MosaicPartners partners)
    {
        return std::make_unique<InMemoryPartnerSource>(std::vector<MosaicPartners>{partners});
    }

    /** @brief Reports whether two images have the same size, channels and bytes. */
    ::testing::AssertionResult images_match(const Image& actual, const Image& expected)
    {
        if (actual.size() != expected.size() || actual.channels() != expected.channels())
            return ::testing::AssertionFailure()
                   << "the image is " << actual.height() << " by " << actual.width() << " by "
                   << actual.channels() << ", expected " << expected.height() << " by "
                   << expected.width() << " by " << expected.channels();

        if (std::memcmp(actual.data(), expected.data(), expected.byte_count()) != 0)
            return ::testing::AssertionFailure() << "the pixels differ";

        return ::testing::AssertionSuccess();
    }

    /** @brief Checks that two label sets hold the same classes, box edges and polygons. */
    void expect_same_labels(const Labels& actual, const Labels& expected)
    {
        ASSERT_EQ(actual.classes(), expected.classes());
        ASSERT_EQ(actual.boxes().size(), expected.boxes().size());
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            SCOPED_TRACE(::testing::Message() << "object " << i);
            EXPECT_EQ(actual.boxes()[i].x1, expected.boxes()[i].x1);
            EXPECT_EQ(actual.boxes()[i].y1, expected.boxes()[i].y1);
            EXPECT_EQ(actual.boxes()[i].x2, expected.boxes()[i].x2);
            EXPECT_EQ(actual.boxes()[i].y2, expected.boxes()[i].y2);
            ASSERT_EQ(actual.polygons()[i].size(), expected.polygons()[i].size());
            for (std::size_t j = 0; j < expected.polygons()[i].size(); ++j)
            {
                EXPECT_EQ(actual.polygons()[i][j].x, expected.polygons()[i][j].x);
                EXPECT_EQ(actual.polygons()[i][j].y, expected.polygons()[i][j].y);
            }
        }
    }

    /** @brief Checks that a box has the given edges, which are small integers, exactly. */
    void expect_box(const PixelBox& actual, float x1, float y1, float x2, float y2)
    {
        EXPECT_EQ(actual.x1, x1);
        EXPECT_EQ(actual.y1, y1);
        EXPECT_EQ(actual.x2, x2);
        EXPECT_EQ(actual.y2, y2);
    }

    /** @brief Checks that an affine plan holds the sizes and the three draws of another. */
    void expect_same_affine(const AffinePlan& actual, const AffinePlan& expected)
    {
        EXPECT_EQ(actual.source, expected.source);
        EXPECT_EQ(actual.output, expected.output);
        EXPECT_EQ(actual.scale, expected.scale);
        EXPECT_EQ(actual.translate_x, expected.translate_x);
        EXPECT_EQ(actual.translate_y, expected.translate_y);
    }

    /** @brief Checks that a colour jitter plan holds the three draws of another. */
    void expect_same_hsv(const HSVPlan& actual, const HSVPlan& expected)
    {
        EXPECT_EQ(actual.hue, expected.hue);
        EXPECT_EQ(actual.saturation, expected.saturation);
        EXPECT_EQ(actual.value, expected.value);
    }

    /** @brief Checks that every partner is one of the listed indices. */
    void expect_partners_among(const MosaicPartners& partners,
                               std::initializer_list<std::size_t> allowed)
    {
        for (const std::size_t partner : partners)
            EXPECT_NE(std::find(allowed.begin(), allowed.end(), partner), allowed.end())
                << "partner " << partner << " was never loaded";
    }

    /** @brief Counts the pixels of @p flipped that are not the mirror of @p straight. */
    std::size_t count_unmirrored_pixels(const Image& flipped, const Image& straight)
    {
        std::size_t wrong = 0;
        for (std::int32_t y = 0; y < K_SIDE; ++y)
        {
            for (std::int32_t x = 0; x < K_SIDE; ++x)
            {
                if (std::memcmp(flipped.row(y) + 3 * x,
                                straight.row(y) + 3 * (K_SIDE - 1 - x), 3) != 0)
                    ++wrong;
            }
        }

        return wrong;
    }

    /** @brief Counts the pixels of a letterboxed image outside its colour and padding bands. */
    std::size_t count_pixels_outside_the_bands(const Image& image,
                                               const std::array<std::uint8_t, 3>& colour)
    {
        std::size_t wrong = 0;
        for (std::int32_t y = 0; y < K_SIDE; ++y)
        {
            const bool is_padding = y < 8 || y >= 24;
            const std::array<std::uint8_t, 3> expected =
                is_padding ? std::array<std::uint8_t, 3>{114, 114, 114} : colour;
            for (std::int32_t x = 0; x < K_SIDE; ++x)
            {
                if (std::memcmp(image.row(y) + 3 * x, expected.data(), 3) != 0)
                    ++wrong;
            }
        }

        return wrong;
    }

    /** @brief Counts the boxes that equal @p target, and checks every box is inside the canvas. */
    std::size_t count_boxes_equal_to(const Labels& labels, const PixelBox& target)
    {
        const float side = static_cast<float>(K_SIDE);
        std::size_t matches = 0;
        for (const PixelBox& box : labels.boxes())
        {
            EXPECT_GE(box.x1, 0.0f);
            EXPECT_GE(box.y1, 0.0f);
            EXPECT_LE(box.x2, side);
            EXPECT_LE(box.y2, side);
            EXPECT_LT(box.x1, box.x2);
            EXPECT_LT(box.y1, box.y2);
            if (box.x1 == target.x1 && box.y1 == target.y1 && box.x2 == target.x2 &&
                box.y2 == target.y2)
                ++matches;
        }

        return matches;
    }

    /** @brief Loads one image the way a pipeline does and returns its pixel labels. */
    Labels load_patch(const SampleSource& source, std::size_t index, Image& patch)
    {
        Image decoded;
        FileLabels file_labels;
        SA_EXPECT_OK(source.load(index, decoded, file_labels));

        if (decoded.channels() == K_GRAY_CHANNELS)
        {
            Image gray;
            SA_EXPECT_OK(resize_long_side(decoded.view(), K_SIDE, gray));
            SA_EXPECT_OK(expand_gray(gray.view(), patch));
        }
        else
        {
            SA_EXPECT_OK(resize_long_side(decoded.view(), K_SIDE, patch));
        }

        return to_pixel_labels(resample_polygons(file_labels), patch.size());
    }

    /**
     * @brief Builds a sample from a trace with the transforms called in the reference order.
     *
     * Mosaic, affine, colour jitter, flip (data/augment.py:2870-2912), each applied with
     * the plan the trace holds.
     */
    Sample compose_from_trace(const SampleSource& source, const TrainingSettings& settings,
                              std::size_t index, const TrainingTrace& trace)
    {
        std::array<Image, K_MOSAIC_PATCHES> patches;
        std::array<Labels, K_MOSAIC_PATCHES> patch_labels;
        patch_labels[0] = load_patch(source, index, patches[0]);

        Image canvas = patches[0];
        Labels labels = patch_labels[0];
        if (trace.mosaic.is_enabled)
        {
            for (std::size_t i = 1; i < K_MOSAIC_PATCHES; ++i)
                patch_labels[i] = load_patch(source, trace.mosaic.partners[i - 1], patches[i]);

            const MosaicPlan plan = require_value(plan_mosaic(
                settings.mosaic, trace.mosaic, settings.image_size,
                {patches[0].size(), patches[1].size(), patches[2].size(), patches[3].size()}));
            SA_EXPECT_OK(apply_mosaic({patches[0].view(), patches[1].view(), patches[2].view(),
                                       patches[3].view()}, plan, canvas));
            labels = apply_mosaic(patch_labels, plan);
        }

        Image warped;
        apply_affine(canvas.view(), trace.affine, warped);
        labels = apply_affine(labels, trace.affine);

        Image jittered;
        apply_hsv(warped.view(), trace.hsv, jittered);

        Sample sample;
        apply_flip(jittered.view(), trace.flip, sample.image);
        sample.labels = apply_flip(labels, trace.flip);
        return sample;
    }

    /** @brief Describes an epoch and a sample index. */
    struct EpochIndexRow
    {
        /** @brief Names the row in the test report. */
        std::string name;

        /** @brief Holds the epoch. */
        std::int32_t epoch;

        /** @brief Holds the sample index. */
        std::size_t index;
    };

    /** @brief Returns one row for each of epochs 0 and 1 and each of the five indices. */
    std::vector<EpochIndexRow> epoch_index_rows()
    {
        std::vector<EpochIndexRow> rows;
        for (std::int32_t epoch = 0; epoch < 2; ++epoch)
            for (std::size_t index = 0; index < 5; ++index)
                rows.push_back(EpochIndexRow{"Epoch" + std::to_string(epoch) + "Index" +
                                                 std::to_string(index),
                                             epoch, index});

        return rows;
    }

    /** @brief Runs each epoch and index. */
    class Small_ImagingPipelineReferenceOrder : public ::testing::TestWithParam<EpochIndexRow>
    {
    };

    /** @brief Describes a seed of a mosaic sample whose largest quadrant is checked. */
    struct SeedRow
    {
        /** @brief Names the row in the test report. */
        std::string name;

        /** @brief Holds the seed. */
        std::uint64_t seed;
    };

    /** @brief Returns one row for each seed in [30, 38). */
    std::vector<SeedRow> quadrant_seed_rows()
    {
        std::vector<SeedRow> rows;
        for (std::uint64_t seed = 30; seed < 38; ++seed)
            rows.push_back(SeedRow{"Seed" + std::to_string(seed), seed});

        return rows;
    }

    /** @brief Runs each seed of the largest-quadrant case. */
    class Small_ImagingPipelineQuadrant : public ::testing::TestWithParam<SeedRow>
    {
    };

    /** @brief Describes one setting out of range and the substrings its refusal names. */
    struct SettingRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Puts the setting out of range. */
        void (*apply)(TrainingSettings&);

        /** @brief Holds the first substring the message names. */
        const char* first;

        /** @brief Holds the second substring the message names; empty when unused. */
        const char* second;

        /** @brief Holds the third substring the message names; empty when unused. */
        const char* third;
    };

    /** @brief Runs each training setting out of range. */
    class Small_ImagingPipelineSetting : public ::testing::TestWithParam<SettingRow>
    {
    };

    /** @brief Describes an index and an epoch outside their ranges and what the refusal names. */
    struct RangeRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Holds the sample index. */
        std::size_t index;

        /** @brief Holds the epoch. */
        std::int32_t epoch;

        /** @brief Holds the substring the message names. */
        const char* needle;
    };

    /** @brief Runs each index or epoch out of range; each starts a child process. */
    class Medium_ImagingPipelineRange : public ::testing::TestWithParam<RangeRow>
    {
    };
} // namespace

/** @brief Tests that the mosaic closes exactly when the epoch reaches epochs minus close_mosaic. */
TEST(Small_ImagingPipeline, MosaicClosesForTheLastTenEpochsOfAHundred)
{
    EXPECT_FALSE(is_mosaic_closed(0, 100, 10));
    EXPECT_FALSE(is_mosaic_closed(89, 100, 10));
    EXPECT_TRUE(is_mosaic_closed(90, 100, 10));
    EXPECT_TRUE(is_mosaic_closed(99, 100, 10));
    EXPECT_FALSE(is_mosaic_closed(99, 100, 0));
    EXPECT_TRUE(is_mosaic_closed(0, 5, 10));
}

/** @brief Tests that a sample's generator is the seed derived by the epoch, then by the index. */
TEST(Small_ImagingPipeline, SampleRandomDerivesFromTheSeedTheEpochAndTheIndex)
{
    const Random random = make_sample_random(2026u, 3, 17);

    EXPECT_TRUE(random == Random(2026u).derive(3u).derive(17u));
    EXPECT_TRUE(random != make_sample_random(2027u, 3, 17));
    EXPECT_TRUE(random != make_sample_random(2026u, 4, 17));
    EXPECT_TRUE(random != make_sample_random(2026u, 3, 18));
    EXPECT_TRUE(random != make_sample_random(2026u, 17, 3));
}

/** @brief Tests that a sample draws partners, gate and centre, then affine, jitter and flip. */
TEST(Small_ImagingPipeline, TrainingDrawsMosaicThenAffineThenHSVThenFlip)
{
    InMemorySampleSource source;
    fill_source(source);
    TrainingSettings settings = make_settings();
    settings.seed = 41;
    TrainingPipeline pipeline =
        require_value(TrainingPipeline::make(source, settings, whole_dataset(source)));
    Sample sample;

    const TrainingTrace trace = require_value(pipeline.make_sample(2, 5, sample));

    Random replica = make_sample_random(41u, 5, 2);
    const std::unique_ptr<DatasetPartners> partners_owner =
        require_value(DatasetPartners::make(source.size()));
    DatasetPartners& partners = *partners_owner;
    const MosaicDraw mosaic =
        require_value(draw_mosaic(settings.mosaic, K_SIDE, 2, partners, replica));
    const AffinePlan affine =
        require_value(plan_affine(settings.affine, ImageSize{2 * K_SIDE, 2 * K_SIDE},
                                  ImageSize{K_SIDE, K_SIDE}, replica));
    const HSVPlan hsv = require_value(plan_hsv(settings.hsv, replica));
    const FlipPlan flip =
        require_value(plan_flip(settings.flip, ImageSize{K_SIDE, K_SIDE}, replica));

    EXPECT_FALSE(trace.is_mosaic_closed);
    ASSERT_TRUE(trace.mosaic.is_enabled);
    EXPECT_EQ(trace.mosaic.partners, mosaic.partners);
    EXPECT_EQ(trace.mosaic.center_x, mosaic.center_x);
    EXPECT_EQ(trace.mosaic.center_y, mosaic.center_y);
    expect_same_affine(trace.affine, affine);
    expect_same_hsv(trace.hsv, hsv);
    EXPECT_EQ(trace.flip.is_flipped, flip.is_flipped);
    EXPECT_EQ(trace.flip.source, (ImageSize{K_SIDE, K_SIDE}));
}

/** @brief Tests that the sample's image is loaded first, then its partners in draw order. */
TEST(Small_ImagingPipeline, TrainingLoadsTheSampleThenItsPartnersInDrawOrder)
{
    InMemorySampleSource source;
    fill_source(source);
    TrainingSettings settings = make_settings();
    settings.seed = 8;
    TrainingPipeline pipeline =
        require_value(TrainingPipeline::make(source, settings, whole_dataset(source)));
    Sample sample;

    const TrainingTrace trace = require_value(pipeline.make_sample(3, 0, sample));

    const std::vector<std::size_t> expected =
        {3, trace.mosaic.partners[0], trace.mosaic.partners[1], trace.mosaic.partners[2]};
    EXPECT_EQ(source.loads(), expected);
}

/** @brief Tests that the sample equals the transforms applied by hand in the reference order. */
TEST_P(Small_ImagingPipelineReferenceOrder, TrainingAppliesItsPlansInTheReferenceOrder)
{
    const EpochIndexRow& row = GetParam();
    InMemorySampleSource source;
    fill_source(source);
    const TrainingSettings settings = make_settings();
    TrainingPipeline pipeline =
        require_value(TrainingPipeline::make(source, settings, whole_dataset(source)));
    Sample sample;

    const TrainingTrace trace = require_value(pipeline.make_sample(row.index, row.epoch, sample));
    const Sample expected = compose_from_trace(source, settings, row.index, trace);

    EXPECT_TRUE(images_match(sample.image, expected.image));
    expect_same_labels(sample.labels, expected.labels);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_ImagingPipelineReferenceOrder,
                         ::testing::ValuesIn(epoch_index_rows()),
                         [](const ::testing::TestParamInfo<EpochIndexRow>& info)
                         {
                             return info.param.name;
                         });

/** @brief Tests that a flip probability of one mirrors the sample a probability of zero gives. */
TEST(Small_ImagingPipeline, TrainingAppliesAForcedFlipAndSkipsAForbiddenOne)
{
    InMemorySampleSource source;
    fill_source(source);
    TrainingSettings always = make_settings();
    always.seed = 77;
    always.flip.probability = 1.0f;
    TrainingSettings never = make_settings();
    never.seed = 77;
    never.flip.probability = 0.0f;
    TrainingPipeline flipping =
        require_value(TrainingPipeline::make(source, always, whole_dataset(source)));
    TrainingPipeline plain =
        require_value(TrainingPipeline::make(source, never, whole_dataset(source)));
    Sample flipped;
    Sample straight;

    const TrainingTrace flipped_trace = require_value(flipping.make_sample(0, 0, flipped));
    const TrainingTrace straight_trace = require_value(plain.make_sample(0, 0, straight));

    ASSERT_TRUE(flipped_trace.flip.is_flipped);
    ASSERT_FALSE(straight_trace.flip.is_flipped);
    ASSERT_EQ(flipped.image.size(), straight.image.size());
    EXPECT_EQ(count_unmirrored_pixels(flipped.image, straight.image), 0u)
        << "the flipped sample is not the mirror of the unflipped one";
    EXPECT_TRUE(images_match(
        flipped.image, compose_from_trace(source, always, 0, flipped_trace).image));
}

/** @brief Tests that a sample has the training size, three channels and its own image's sizes. */
TEST(Small_ImagingPipeline, TrainingProducesTheTrainingSizeAndTheSizesOfItsOwnImage)
{
    InMemorySampleSource source;
    fill_source(source);
    TrainingPipeline pipeline = require_value(TrainingPipeline::make(source, make_settings()));
    Sample sample;

    const TrainingTrace trace = require_value(pipeline.make_sample(0, 0, sample));

    EXPECT_EQ(sample.image.size(), (ImageSize{K_SIDE, K_SIDE}));
    EXPECT_EQ(sample.image.channels(), K_RGB_CHANNELS);
    EXPECT_EQ(sample.original_size, (ImageSize{32, 64}));
    EXPECT_EQ(sample.loaded_size, (ImageSize{16, 32}));
    EXPECT_EQ(trace.affine.source, (ImageSize{2 * K_SIDE, 2 * K_SIDE}));
    EXPECT_EQ(trace.affine.output, (ImageSize{K_SIDE, K_SIDE}));
}

/** @brief Tests that the quadrant of at least 16 by 16 pixels keeps its box in the output. */
TEST_P(Small_ImagingPipelineQuadrant, MosaicSampleKeepsItsLargestQuadrantBoxInsideTheTrainingSize)
{
    InMemorySampleSource source;
    for (std::int32_t i = 0; i < 4; ++i)
    {
        source.add(make_textured(ImageSize{K_SIDE, K_SIDE}, 50 * i),
                   one_object(i, NormalizedBox{0.5f, 0.5f, 1.0f, 1.0f}));
    }

    TrainingSettings settings = make_settings();
    settings.affine.scale = 0.0f;
    settings.affine.translate = 0.0f;
    settings.flip.probability = 0.0f;
    settings.seed = GetParam().seed;
    const float side = static_cast<float>(K_SIDE);
    TrainingPipeline pipeline =
        require_value(TrainingPipeline::make(source, settings, whole_dataset(source)));
    Sample sample;

    const TrainingTrace trace = require_value(pipeline.make_sample(0, 0, sample));

    // The warp shows canvas columns and rows 16 to 48, so the centre lands at centre - 16.
    ASSERT_TRUE(trace.mosaic.is_enabled);
    const float split_x = static_cast<float>(trace.mosaic.center_x - K_SIDE / 2);
    const float split_y = static_cast<float>(trace.mosaic.center_y - K_SIDE / 2);
    const bool is_left_wider = trace.mosaic.center_x >= K_SIDE;
    const bool is_top_taller = trace.mosaic.center_y >= K_SIDE;
    const PixelBox largest{is_left_wider ? 0.0f : split_x, is_top_taller ? 0.0f : split_y,
                           is_left_wider ? split_x : side, is_top_taller ? split_y : side};
    ASSERT_FALSE(sample.labels.is_empty());

    EXPECT_EQ(count_boxes_equal_to(sample.labels, largest), 1u)
        << "the quadrant of at least 16 by 16 pixels lost its box";
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_ImagingPipelineQuadrant,
                         ::testing::ValuesIn(quadrant_seed_rows()),
                         [](const ::testing::TestParamInfo<SeedRow>& info)
                         {
                             return info.param.name;
                         });

/** @brief Tests that a closed epoch loads one image and draws affine, jitter and flip only. */
TEST(Small_ImagingPipeline, ClosedEpochSkipsTheMosaicAndItsDraws)
{
    InMemorySampleSource source;
    fill_source(source);
    TrainingSettings settings = make_settings();
    settings.epochs = 10;
    settings.close_mosaic = 3;
    settings.seed = 12;
    TrainingPipeline pipeline = require_value(TrainingPipeline::make(source, settings));
    Sample sample;

    const TrainingTrace trace = require_value(pipeline.make_sample(0, 7, sample));

    Random replica = make_sample_random(12u, 7, 0);
    const AffinePlan affine =
        require_value(plan_affine(settings.affine, ImageSize{16, 32}, ImageSize{K_SIDE, K_SIDE},
                                  replica));
    const HSVPlan hsv = require_value(plan_hsv(settings.hsv, replica));
    const FlipPlan flip =
        require_value(plan_flip(settings.flip, ImageSize{K_SIDE, K_SIDE}, replica));

    EXPECT_TRUE(trace.is_mosaic_closed);
    EXPECT_FALSE(trace.mosaic.is_enabled);
    EXPECT_EQ(source.loads(), std::vector<std::size_t>{0});
    expect_same_affine(trace.affine, affine);
    expect_same_hsv(trace.hsv, hsv);
    EXPECT_EQ(trace.flip.is_flipped, flip.is_flipped);
    EXPECT_EQ(sample.image.size(), (ImageSize{K_SIDE, K_SIDE}));
    const Sample expected = compose_from_trace(source, settings, 0, trace);
    EXPECT_TRUE(images_match(sample.image, expected.image));
    expect_same_labels(sample.labels, expected.labels);
}

/** @brief Tests that a sample without a mosaic carries the box of its own image. */
TEST(Small_ImagingPipeline, SampleWithoutMosaicKeepsTheBoxOfItsOwnImage)
{
    InMemorySampleSource source;
    fill_source(source);
    TrainingSettings settings = make_settings();
    settings.epochs = 10;
    settings.close_mosaic = 3;
    settings.affine.scale = 0.0f;
    settings.affine.translate = 0.0f;
    TrainingPipeline pipeline = require_value(TrainingPipeline::make(source, settings));
    Sample sample;

    const TrainingTrace trace = require_value(pipeline.make_sample(0, 7, sample));

    // The box (8, 4, 24, 12) of the 16 by 32 image moves down by 8 and mirrors onto itself.
    ASSERT_FALSE(trace.mosaic.is_enabled);
    ASSERT_EQ(sample.labels.size(), 1u);
    EXPECT_EQ(sample.labels.classes()[0], 0);
    expect_box(sample.labels.boxes()[0], 8.0f, 12.0f, 24.0f, 20.0f);
}

/** @brief Tests that the last open epoch still composes a mosaic of four loads. */
TEST(Small_ImagingPipeline, EpochBeforeTheClosedOnesStillMakesAMosaic)
{
    InMemorySampleSource source;
    fill_source(source);
    TrainingSettings settings = make_settings();
    settings.epochs = 10;
    settings.close_mosaic = 3;
    TrainingPipeline pipeline = require_value(TrainingPipeline::make(source, settings));
    Sample sample;

    const TrainingTrace trace = require_value(pipeline.make_sample(0, 6, sample));

    EXPECT_FALSE(trace.is_mosaic_closed);
    EXPECT_TRUE(trace.mosaic.is_enabled);
    EXPECT_EQ(source.loads().size(), K_MOSAIC_PATCHES);
}

/** @brief Tests that a gate that stays shut in an open epoch warps the loaded image alone. */
TEST(Small_ImagingPipeline, ClosedGateInAnOpenEpochWarpsTheLoadedImage)
{
    InMemorySampleSource source;
    fill_source(source);
    TrainingSettings settings = make_settings();
    settings.mosaic.probability = 0.0f;
    settings.seed = 3;
    TrainingPipeline pipeline = require_value(TrainingPipeline::make(source, settings));
    Sample sample;

    const TrainingTrace trace = require_value(pipeline.make_sample(2, 0, sample));

    EXPECT_FALSE(trace.is_mosaic_closed);
    EXPECT_FALSE(trace.mosaic.is_enabled);
    EXPECT_EQ(trace.affine.source, (ImageSize{32, 16}));
    EXPECT_EQ(source.loads(), std::vector<std::size_t>{2});
    EXPECT_EQ(sample.image.size(), (ImageSize{K_SIDE, K_SIDE}));
    EXPECT_TRUE(images_match(sample.image,
                             compose_from_trace(source, settings, 2, trace).image));
}

/** @brief Tests that one seed repeats a sample and that another seed, epoch or index changes it. */
TEST(Small_ImagingPipeline, SampleDependsOnTheSeedTheEpochAndTheIndexAlone)
{
    InMemorySampleSource source;
    fill_source(source);
    TrainingSettings settings = make_settings();
    settings.seed = 2026;
    TrainingSettings other_settings = make_settings();
    other_settings.seed = 2027;
    TrainingPipeline pipeline =
        require_value(TrainingPipeline::make(source, settings, whole_dataset(source)));
    TrainingPipeline other_pipeline =
        require_value(TrainingPipeline::make(source, other_settings, whole_dataset(source)));
    Sample first;
    Sample other_seed;
    Sample other_epoch;
    Sample again;

    const TrainingTrace first_trace = require_value(pipeline.make_sample(1, 0, first));
    const TrainingTrace seed_trace = require_value(other_pipeline.make_sample(1, 0, other_seed));
    const TrainingTrace epoch_trace = require_value(pipeline.make_sample(1, 1, other_epoch));
    const TrainingTrace again_trace = require_value(pipeline.make_sample(1, 0, again));

    EXPECT_TRUE(images_match(again.image, first.image));
    expect_same_labels(again.labels, first.labels);
    expect_same_affine(again_trace.affine, first_trace.affine);
    EXPECT_EQ(again_trace.mosaic.partners, first_trace.mosaic.partners);
    EXPECT_FALSE(images_match(other_seed.image, first.image));
    EXPECT_FALSE(images_match(other_epoch.image, first.image));
    EXPECT_GT(std::abs(seed_trace.affine.scale - first_trace.affine.scale), K_MIN_SCALE_GAP);
    EXPECT_GT(std::abs(epoch_trace.affine.scale - first_trace.affine.scale), K_MIN_SCALE_GAP);
}

/** @brief Tests that the samples made before do not change the draws of the next one. */
TEST(Small_ImagingPipeline, EarlierSamplesDoNotShiftTheDrawsOfALaterOne)
{
    InMemorySampleSource source;
    fill_source(source);
    TrainingPipeline used =
        require_value(TrainingPipeline::make(source, make_settings(), whole_dataset(source)));
    TrainingPipeline fresh =
        require_value(TrainingPipeline::make(source, make_settings(), whole_dataset(source)));
    Sample warm;
    Sample from_used;
    Sample from_fresh;

    // A mosaic sample makes six draws more than a sample of a closed epoch.
    SA_EXPECT_OK(used.make_sample(4, 0, warm));
    SA_EXPECT_OK(used.make_sample(2, 95, warm));
    const TrainingTrace used_trace = require_value(used.make_sample(0, 0, from_used));
    const TrainingTrace fresh_trace = require_value(fresh.make_sample(0, 0, from_fresh));

    EXPECT_TRUE(images_match(from_used.image, from_fresh.image));
    expect_same_labels(from_used.labels, from_fresh.labels);
    expect_same_affine(used_trace.affine, fresh_trace.affine);
    expect_same_hsv(used_trace.hsv, fresh_trace.hsv);
    EXPECT_EQ(used_trace.mosaic.partners, fresh_trace.mosaic.partners);
    EXPECT_EQ(used_trace.flip.is_flipped, fresh_trace.flip.is_flipped);
}

/** @brief Tests that a pipeline built without a partner source draws from its recent loads. */
TEST(Small_ImagingPipeline, DefaultPartnersComeFromTheImagesThePipelineLoaded)
{
    InMemorySampleSource source;
    fill_source(source);
    TrainingPipeline pipeline = require_value(TrainingPipeline::make(source, make_settings()));
    Sample sample;

    const TrainingTrace first = require_value(pipeline.make_sample(3, 0, sample));
    const TrainingTrace second = require_value(pipeline.make_sample(1, 0, sample));

    EXPECT_EQ(first.mosaic.partners, (MosaicPartners{3, 3, 3}));
    expect_partners_among(second.mosaic.partners, {3, 1});
}

/** @brief Tests that a pipeline loads the partners the source it was given chooses. */
TEST(Small_ImagingPipeline, GivenPartnerSourceChoosesThePartners)
{
    InMemorySampleSource source;
    fill_source(source);
    TrainingPipeline pipeline = require_value(
        TrainingPipeline::make(source, make_settings(), fixed_partners(MosaicPartners{4, 4, 0})));
    Sample sample;

    const TrainingTrace trace = require_value(pipeline.make_sample(2, 0, sample));

    ASSERT_TRUE(trace.mosaic.is_enabled);
    EXPECT_EQ(trace.mosaic.partners, (MosaicPartners{4, 4, 0}));
    EXPECT_EQ(source.loads(), (std::vector<std::size_t>{2, 4, 4, 0}));
}

/** @brief Tests that each setting out of range is refused when the pipeline is constructed. */
TEST_P(Small_ImagingPipelineSetting, TrainingRejectsASettingOutOfRangeAtConstruction)
{
    const SettingRow& row = GetParam();
    InMemorySampleSource source;
    fill_source(source);
    TrainingSettings settings = make_settings();
    row.apply(settings);

    SA_EXPECT_FAILURE(TrainingPipeline::make(source, settings), StatusCode::InvalidDescription,
                      row.first, row.second, row.third);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ImagingPipelineSetting,
    ::testing::Values(
        SettingRow{"NoImageSize", [](TrainingSettings& s) { s.image_size = 0; },
                   "validate_mosaic_size", "image_size of 0", ""},
        SettingRow{"ImageSizeWhoseCanvasIsTooLarge",
                   [](TrainingSettings& s) { s.image_size = K_MAX_IMAGE_SIDE / 2 + 1; },
                   "validate_mosaic_size", "image_size of 8193", "[1, 8192]"},
        SettingRow{"NoEpochs", [](TrainingSettings& s) { s.epochs = 0; }, "validate_training",
                   "epoch count", ""},
        SettingRow{"NegativeCloseMosaic", [](TrainingSettings& s) { s.close_mosaic = -1; },
                   "validate_training", "close_mosaic of -1", ""},
        SettingRow{"NoBatchSize", [](TrainingSettings& s) { s.batch_size = 0; },
                   "validate_training", "batch_size of 0", ""},
        SettingRow{"MosaicProbabilityAboveOne",
                   [](TrainingSettings& s) { s.mosaic.probability = 1.5f; }, "validate_mosaic",
                   "probability", ""},
        SettingRow{"AffineScaleOfOne", [](TrainingSettings& s) { s.affine.scale = 1.0f; },
                   "validate_affine", "scale", ""},
        SettingRow{"NegativeValueGain", [](TrainingSettings& s) { s.hsv.value_gain = -0.4f; },
                   "validate_hsv", "gain", ""},
        SettingRow{"FlipProbabilityNaN",
                   [](TrainingSettings& s)
                   {
                       s.flip.probability = std::numeric_limits<float>::quiet_NaN();
                   },
                   "validate_flip", "probability", ""}),
    [](const ::testing::TestParamInfo<SettingRow>& info)
    {
        return std::string(info.param.name);
    });

/** @brief Tests that a setting out of range is refused also when a partner source is given. */
TEST(Small_ImagingPipeline, TrainingRejectsASettingOutOfRangeWithAGivenPartnerSource)
{
    InMemorySampleSource source;
    fill_source(source);
    TrainingSettings bad_flip = make_settings();
    bad_flip.flip.probability = std::numeric_limits<float>::quiet_NaN();

    SA_EXPECT_FAILURE(TrainingPipeline::make(source, bad_flip, whole_dataset(source)),
                      StatusCode::InvalidDescription, "validate_flip", "probability");
}

/** @brief Tests that a pipeline refused for its settings loaded no image. */
TEST(Small_ImagingPipeline, TrainingRejectedAtConstructionLoadsNoImage)
{
    InMemorySampleSource source;
    fill_source(source);
    TrainingSettings no_size = make_settings();
    no_size.image_size = 0;

    static_cast<void>(TrainingPipeline::make(source, no_size));

    EXPECT_TRUE(source.loads().empty()) << "a rejected construction loaded an image";
}

/** @brief Tests that validate_training accepts the default settings. */
TEST(Small_ImagingPipeline, ValidateTrainingAcceptsTheDefaultSettings)
{
    SA_EXPECT_OK(validate_training(make_settings()));
}

/** @brief Tests that an empty source is refused at construction, with or without partners. */
TEST(Small_ImagingPipeline, TrainingRejectsAnEmptySource)
{
    InMemorySampleSource source;
    fill_source(source);
    InMemorySampleSource empty;

    SA_EXPECT_FAILURE(TrainingPipeline::make(empty, make_settings()),
                      StatusCode::InvalidDescription, "TrainingPipeline", "no image");
    SA_EXPECT_FAILURE(TrainingPipeline::make(empty, make_settings(), whole_dataset(source)),
                      StatusCode::InvalidDescription, "TrainingPipeline", "no image");
}

/** @brief Tests that a null partner source is refused at construction. */
TEST(Medium_ImagingPipeline, TrainingRejectsANullPartnerSource)
{
    InMemorySampleSource source;
    fill_source(source);

    SA_EXPECT_FATAL(static_cast<void>(TrainingPipeline::make(source, make_settings(), nullptr)),
                    "TrainingPipeline",
                    "partner source is null");
}

/** @brief Tests that an index or an epoch outside its range is refused. */
TEST_P(Medium_ImagingPipelineRange, TrainingRejectsAnIndexOrAnEpochOutsideItsRange)
{
    const RangeRow& row = GetParam();
    InMemorySampleSource source;
    fill_source(source);
    TrainingPipeline pipeline = require_value(TrainingPipeline::make(source, make_settings()));
    Sample sample;

    SA_EXPECT_FATAL(static_cast<void>(require_value(pipeline.make_sample(row.index, row.epoch,
                                                                         sample))),
                    "TrainingPipeline::make_sample", row.needle);
}

INSTANTIATE_TEST_SUITE_P(Rows, Medium_ImagingPipelineRange,
                         ::testing::Values(RangeRow{"IndexAtTheSourceSize", 5, 0, "index 5"},
                                           RangeRow{"NegativeEpoch", 0, -1, "epoch -1"},
                                           RangeRow{"EpochAtTheEpochCount", 0, 100, "epoch 100"}),
                         [](const ::testing::TestParamInfo<RangeRow>& info)
                         {
                             return std::string(info.param.name);
                         });

/** @brief Tests that a one-channel image leaves the training pipeline with three equal channels. */
TEST(Small_ImagingPipeline, TrainingExpandsAGreyImageToRGB)
{
    InMemorySampleSource source;
    source.add(make_flat_image(ImageSize{K_SIDE, K_SIDE}, 1, 77), FileLabels{});
    TrainingPipeline pipeline =
        require_value(TrainingPipeline::make(source, make_still_settings()));
    Sample sample;

    SA_EXPECT_OK(pipeline.make_sample(0, 0, sample));

    ASSERT_EQ(sample.image.size(), (ImageSize{K_SIDE, K_SIDE}));
    ASSERT_EQ(sample.image.channels(), K_RGB_CHANNELS);
    const std::uint8_t* centre = sample.image.row(K_SIDE / 2) + 3 * (K_SIDE / 2);
    EXPECT_EQ(centre[0], 77);
    EXPECT_EQ(centre[1], 77);
    EXPECT_EQ(centre[2], 77);
}

/** @brief Tests that a mosaic of grey and colour images is composed in three channels. */
TEST(Small_ImagingPipeline, TrainingComposesAMosaicOfGreyAndColourImages)
{
    InMemorySampleSource source;
    source.add(make_flat_image(ImageSize{K_SIDE, K_SIDE}, 1, 40), FileLabels{});
    source.add(make_constant(ImageSize{K_SIDE, K_SIDE}, 200, 10, 90), FileLabels{});
    source.add(make_flat_image(ImageSize{16, K_SIDE}, 1, 220), FileLabels{});
    const TrainingSettings settings = make_settings();
    TrainingPipeline pipeline = require_value(
        TrainingPipeline::make(source, settings, fixed_partners(MosaicPartners{1, 2, 0})));
    Sample sample;

    const TrainingTrace trace = require_value(pipeline.make_sample(0, 0, sample));

    ASSERT_TRUE(trace.mosaic.is_enabled);
    EXPECT_EQ(sample.image.channels(), K_RGB_CHANNELS);
    EXPECT_EQ(sample.image.size(), (ImageSize{K_SIDE, K_SIDE}));
    EXPECT_TRUE(images_match(sample.image,
                             compose_from_trace(source, settings, 0, trace).image));
}

/** @brief Tests that an image of neither one nor three channels is refused with its index. */
TEST(Small_ImagingPipeline, TrainingRejectsAnImageOfFourChannels)
{
    InMemorySampleSource source;
    source.add(make_flat_image(ImageSize{K_SIDE, K_SIDE}, 4, 9), FileLabels{});
    TrainingPipeline pipeline =
        require_value(TrainingPipeline::make(source, make_still_settings()));
    Sample sample;

    SA_EXPECT_FAILURE(pipeline.make_sample(0, 0, sample),
                      StatusCode::InvalidFile, "TrainingPipeline::make_sample", "4 channels",
                      "image 0");
}

/** @brief Tests that a polygon label is resampled, carried through and bounds its box. */
TEST(Small_ImagingPipeline, TrainingCarriesAPolygonAndTakesTheBoxFromIt)
{
    InMemorySampleSource source;
    FileLabels labels;
    labels.add(5, NormalizedBox{0.5f, 0.5f, 0.5f, 0.5f},
               Polygon{Point{0.25f, 0.25f}, Point{0.75f, 0.25f}, Point{0.5f, 0.75f}});
    source.add(make_textured(ImageSize{K_SIDE, K_SIDE}, 5), labels);
    TrainingPipeline pipeline =
        require_value(TrainingPipeline::make(source, make_still_settings()));
    Sample sample;

    SA_EXPECT_OK(pipeline.make_sample(0, 0, sample));

    // The still settings map the 32 by 32 image onto itself, so the triangle stays put.
    ASSERT_EQ(sample.labels.size(), 1u);
    EXPECT_EQ(sample.labels.classes()[0], 5);
    expect_box(sample.labels.boxes()[0], 8.0f, 8.0f, 24.0f, 24.0f);
    ASSERT_EQ(sample.labels.polygons()[0].size(), K_POLYGON_POINTS);
    EXPECT_EQ(sample.labels.polygons()[0][0].x, 8.0f);
    EXPECT_EQ(sample.labels.polygons()[0][0].y, 8.0f);
}

/** @brief Tests that validation resizes the long side and then only pads. */
TEST(Small_ImagingPipeline, ValidationResizesTheLongSideThenOnlyPads)
{
    InMemorySampleSource source;
    source.add(make_constant(ImageSize{8, 16}, 10, 200, 30),
               one_object(4, NormalizedBox{0.5f, 0.5f, 0.5f, 0.5f}));
    ValidationSettings settings;
    settings.image_size = K_SIDE;
    ValidationPipeline pipeline = require_value(ValidationPipeline::make(source, settings));
    Sample sample;

    const LetterboxPlan plan =
        require_value(pipeline.make_sample(0, ImageSize{K_SIDE, K_SIDE}, sample));

    EXPECT_EQ(sample.original_size, (ImageSize{8, 16}));
    EXPECT_EQ(sample.loaded_size, (ImageSize{16, 32}));
    EXPECT_EQ(plan.source, (ImageSize{16, 32}));
    EXPECT_EQ(plan.target, (ImageSize{K_SIDE, K_SIDE}));
    EXPECT_EQ(plan.resized, (ImageSize{16, 32}));
    EXPECT_EQ(plan.ratio, 1.0);
    EXPECT_EQ(plan.top, 8);
    EXPECT_EQ(plan.bottom, 8);
    EXPECT_EQ(plan.left, 0);
    EXPECT_EQ(plan.right, 0);

    ASSERT_EQ(sample.image.size(), (ImageSize{K_SIDE, K_SIDE}));
    ASSERT_EQ(sample.image.channels(), 3);
    EXPECT_EQ(count_pixels_outside_the_bands(sample.image, {10, 200, 30}), 0u)
        << "a constant image must come out constant between two bands of 114";
    EXPECT_EQ(source.loads(), std::vector<std::size_t>{0});
}

/** @brief Tests that validation moves a box by the padding and clips nothing. */
TEST(Small_ImagingPipeline, ValidationMovesBoxesByThePaddingWithoutClipping)
{
    InMemorySampleSource source;
    source.add(make_constant(ImageSize{8, 16}, 10, 200, 30),
               one_object(4, NormalizedBox{0.5f, 0.5f, 0.5f, 0.5f}));
    ValidationSettings settings;
    settings.image_size = K_SIDE;
    ValidationPipeline pipeline = require_value(ValidationPipeline::make(source, settings));
    Sample sample;

    SA_EXPECT_OK(pipeline.make_sample(0, ImageSize{K_SIDE, K_SIDE}, sample));

    // On the loaded 16 by 32 image the box is (8, 4, 24, 12); the top padding adds 8 to y.
    ASSERT_EQ(sample.labels.size(), 1u);
    EXPECT_EQ(sample.labels.classes()[0], 4);
    expect_box(sample.labels.boxes()[0], 8.0f, 12.0f, 24.0f, 20.0f);
}

/** @brief Tests that a target of the loaded size adds no padding and copies the image. */
TEST(Small_ImagingPipeline, ValidationIntoARectangularTargetAddsNoPadding)
{
    InMemorySampleSource source;
    source.add(make_textured(ImageSize{16, 32}, 17), FileLabels{});
    ValidationSettings settings;
    settings.image_size = K_SIDE;
    ValidationPipeline pipeline = require_value(ValidationPipeline::make(source, settings));
    Sample sample;

    const LetterboxPlan plan = require_value(pipeline.make_sample(0, ImageSize{16, 32}, sample));

    EXPECT_EQ(plan.top + plan.bottom + plan.left + plan.right, 0);
    EXPECT_TRUE(images_match(sample.image, make_textured(ImageSize{16, 32}, 17)));
    EXPECT_TRUE(sample.labels.is_empty());
}

/** @brief Tests that a validation sample does not depend on the samples made before it. */
TEST(Small_ImagingPipeline, ValidationRepeatsItsSampleAfterAnotherIndex)
{
    InMemorySampleSource source;
    fill_source(source);
    ValidationSettings settings;
    settings.image_size = K_SIDE;
    ValidationPipeline pipeline = require_value(ValidationPipeline::make(source, settings));
    Sample first;
    Sample between;
    Sample again;

    SA_EXPECT_OK(pipeline.make_sample(0, ImageSize{K_SIDE, K_SIDE}, first));
    SA_EXPECT_OK(pipeline.make_sample(2, ImageSize{K_SIDE, K_SIDE}, between));
    SA_EXPECT_OK(pipeline.make_sample(0, ImageSize{K_SIDE, K_SIDE}, again));

    EXPECT_TRUE(images_match(again.image, first.image));
    expect_same_labels(again.labels, first.labels);
    EXPECT_FALSE(images_match(between.image, first.image));
}

/** @brief Tests that validation expands a grey image and pads it in three channels. */
TEST(Small_ImagingPipeline, ValidationExpandsAGreyImageToRGB)
{
    InMemorySampleSource source;
    source.add(make_flat_image(ImageSize{8, 16}, 1, 61), FileLabels{});
    ValidationSettings settings;
    settings.image_size = K_SIDE;
    ValidationPipeline pipeline = require_value(ValidationPipeline::make(source, settings));
    Sample sample;

    SA_EXPECT_OK(pipeline.make_sample(0, ImageSize{K_SIDE, K_SIDE}, sample));

    ASSERT_EQ(sample.image.size(), (ImageSize{K_SIDE, K_SIDE}));
    ASSERT_EQ(sample.image.channels(), K_RGB_CHANNELS);
    const std::uint8_t* content = sample.image.row(16) + 3 * 5;
    const std::uint8_t* padding = sample.image.row(2) + 3 * 5;
    EXPECT_EQ(content[0], 61);
    EXPECT_EQ(content[1], 61);
    EXPECT_EQ(content[2], 61);
    EXPECT_EQ(padding[0], 114);
    EXPECT_EQ(padding[2], 114);
}

/** @brief Tests that validation resamples a polygon and moves it with the letterbox. */
TEST(Small_ImagingPipeline, ValidationCarriesAPolygonThroughTheLetterbox)
{
    InMemorySampleSource source;
    FileLabels labels;
    labels.add(2, NormalizedBox{0.5f, 0.5f, 0.5f, 0.5f},
               Polygon{Point{0.25f, 0.25f}, Point{0.75f, 0.25f}, Point{0.5f, 0.75f}});
    source.add(make_constant(ImageSize{8, 16}, 10, 200, 30), labels);
    ValidationSettings settings;
    settings.image_size = K_SIDE;
    ValidationPipeline pipeline = require_value(ValidationPipeline::make(source, settings));
    Sample sample;

    SA_EXPECT_OK(pipeline.make_sample(0, ImageSize{K_SIDE, K_SIDE}, sample));

    // On the loaded 16 by 32 image the first vertex is (8, 4); the top padding adds 8 to y.
    ASSERT_EQ(sample.labels.size(), 1u);
    ASSERT_EQ(sample.labels.polygons()[0].size(), K_POLYGON_POINTS);
    EXPECT_EQ(sample.labels.polygons()[0][0].x, 8.0f);
    EXPECT_EQ(sample.labels.polygons()[0][0].y, 12.0f);
}

/** @brief Tests that validation checks its settings at construction. */
TEST(Small_ImagingPipeline, ValidationRejectsSettingsOutsideTheirRange)
{
    InMemorySampleSource source;
    fill_source(source);
    ValidationSettings no_size;
    no_size.image_size = 0;
    ValidationSettings settings;
    settings.image_size = K_SIDE;

    SA_EXPECT_FAILURE(ValidationPipeline::make(source, no_size), StatusCode::InvalidDescription,
                      "validate_validation", "image_size of 0");
    SA_EXPECT_OK(validate_validation(settings));
}

/** @brief Tests that validation refuses an empty source at construction. */
TEST(Small_ImagingPipeline, ValidationRejectsAnEmptySource)
{
    InMemorySampleSource empty;
    ValidationSettings settings;
    settings.image_size = K_SIDE;

    SA_EXPECT_FAILURE(ValidationPipeline::make(empty, settings), StatusCode::InvalidDescription,
                      "ValidationPipeline", "no image");
}

/** @brief Tests that validation refuses an index at or above the source size. */
TEST(Medium_ImagingPipeline, ValidationRejectsAnIndexOutsideTheSource)
{
    InMemorySampleSource source;
    fill_source(source);
    ValidationSettings settings;
    settings.image_size = K_SIDE;
    ValidationPipeline pipeline = require_value(ValidationPipeline::make(source, settings));
    Sample sample;

    SA_EXPECT_FATAL(static_cast<void>(require_value(pipeline.make_sample(9,
                                                                         ImageSize{K_SIDE, K_SIDE},
                                                                         sample))),
                    "ValidationPipeline::make_sample", "index 9");
}
