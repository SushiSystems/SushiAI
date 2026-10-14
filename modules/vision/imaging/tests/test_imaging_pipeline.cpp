/**************************************************************************/
/* test_imaging_pipeline.cpp                                              */
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
 * @file test_imaging_pipeline.cpp
 * @brief Tests the order, the output size and the reproducibility of the two data pipelines.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <SushiAI/vision/imaging/dataset_partners.hpp>
#include <SushiAI/vision/imaging/gray.hpp>
#include <SushiAI/vision/imaging/pipeline.hpp>
#include <SushiAI/vision/imaging/resize.hpp>

#include <SushiAI/failure_expectations/throw_expectations.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Imaging;

namespace
{
    /** @brief Holds the training size of the tests; a mosaic canvas is 64 by 64. */
    constexpr std::int32_t K_SIDE = 32;

    /** @brief Holds the tolerance of a replayed draw, which repeats the same arithmetic. */
    constexpr float K_REPLAY_TOLERANCE = 0.0f;

    /** @brief Holds the absolute tolerance, in pixels, of a box edge made of small integers. */
    constexpr float K_BOX_TOLERANCE = 1e-4f;

    /** @brief Holds one step of the 24-bit uniform over the default scale range of width 1. */
    constexpr float K_MIN_SCALE_GAP = 1.0f / 16777216.0f;

    /** @brief Supplies images and labels held in memory and records every load. */
    class MemorySource final : public SampleSource
    {
        public:
            /** @brief Appends one image with its labels. */
            void add(Image image, FileLabels labels)
            {
                images_.push_back(std::move(image));
                labels_.push_back(std::move(labels));
            }

            /** @brief Returns the number of images. */
            [[nodiscard]] std::size_t size() const noexcept override { return images_.size(); }

            /** @brief Copies one image and its labels out and records the index. */
            void load(std::size_t index, Image& image, FileLabels& labels) const override
            {
                loads_.push_back(index);
                image = images_[index];
                labels = labels_[index];
            }

            /** @brief Returns the indices loaded so far, in call order. */
            [[nodiscard]] const std::vector<std::size_t>& loads() const noexcept { return loads_; }

            /** @brief Forgets the recorded loads. */
            void clear_loads() const noexcept { loads_.clear(); }

        private:
            /// @brief Holds the images.
            std::vector<Image> images_;

            /// @brief Holds the labels, index for index with the images.
            std::vector<FileLabels> labels_;

            /// @brief Holds the indices load() was called with.
            mutable std::vector<std::size_t> loads_;
    };

    /** @brief Gives the same three partners for every sample and makes no draw. */
    class FixedPartners final : public PartnerSource
    {
        public:
            /** @brief Constructs a source that always gives the given partners. */
            explicit FixedPartners(MosaicPartners partners) noexcept : partners_(partners)
            {

            }

            /** @brief Returns the partners the source was constructed with. */
            [[nodiscard]] MosaicPartners choose(std::size_t index, Random& random) override
            {
                static_cast<void>(index);
                static_cast<void>(random);
                return partners_;
            }

        private:
            /// @brief Holds the partners every call returns.
            MosaicPartners partners_{};
    };

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

    /** @brief Returns a three-channel image with every pixel the same colour. */
    Image make_constant(ImageSize size, std::uint8_t red, std::uint8_t green, std::uint8_t blue)
    {
        Image image(size, 3);
        for (std::int32_t y = 0; y < size.height; ++y)
        {
            std::uint8_t* row = image.row(y);
            for (std::int32_t x = 0; x < size.width; ++x)
            {
                row[3 * x] = red;
                row[3 * x + 1] = green;
                row[3 * x + 2] = blue;
            }
        }

        return image;
    }

    /** @brief Returns an image of the given channel count with every byte at one level. */
    Image make_flat(ImageSize size, std::int32_t channels, std::uint8_t level)
    {
        Image image(size, channels);
        image.fill(level);
        return image;
    }

    /** @brief Returns the labels of one object of the given class. */
    FileLabels one_object(std::int32_t class_id, NormalizedBox box)
    {
        return FileLabels({class_id}, {box});
    }

    /** @brief Fills a source with five textured images whose sides are powers of two. */
    void fill_source(MemorySource& source)
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
    std::unique_ptr<PartnerSource> whole_dataset(const MemorySource& source)
    {
        return std::make_unique<DatasetPartners>(source.size());
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
            EXPECT_NEAR(actual.boxes()[i].x1, expected.boxes()[i].x1, K_REPLAY_TOLERANCE);
            EXPECT_NEAR(actual.boxes()[i].y1, expected.boxes()[i].y1, K_REPLAY_TOLERANCE);
            EXPECT_NEAR(actual.boxes()[i].x2, expected.boxes()[i].x2, K_REPLAY_TOLERANCE);
            EXPECT_NEAR(actual.boxes()[i].y2, expected.boxes()[i].y2, K_REPLAY_TOLERANCE);
            ASSERT_EQ(actual.polygons()[i].size(), expected.polygons()[i].size());
            for (std::size_t j = 0; j < expected.polygons()[i].size(); ++j)
            {
                EXPECT_NEAR(actual.polygons()[i][j].x, expected.polygons()[i][j].x,
                            K_REPLAY_TOLERANCE);
                EXPECT_NEAR(actual.polygons()[i][j].y, expected.polygons()[i][j].y,
                            K_REPLAY_TOLERANCE);
            }
        }
    }

    /** @brief Checks that an affine plan holds the sizes and the three draws of another. */
    void expect_same_affine(const AffinePlan& actual, const AffinePlan& expected)
    {
        EXPECT_EQ(actual.source, expected.source);
        EXPECT_EQ(actual.output, expected.output);
        EXPECT_NEAR(actual.scale, expected.scale, K_REPLAY_TOLERANCE);
        EXPECT_NEAR(actual.translate_x, expected.translate_x, K_REPLAY_TOLERANCE);
        EXPECT_NEAR(actual.translate_y, expected.translate_y, K_REPLAY_TOLERANCE);
    }

    /** @brief Checks that a colour jitter plan holds the three draws of another. */
    void expect_same_hsv(const HSVPlan& actual, const HSVPlan& expected)
    {
        EXPECT_NEAR(actual.hue, expected.hue, K_REPLAY_TOLERANCE);
        EXPECT_NEAR(actual.saturation, expected.saturation, K_REPLAY_TOLERANCE);
        EXPECT_NEAR(actual.value, expected.value, K_REPLAY_TOLERANCE);
    }

    /** @brief Loads one image the way a pipeline does and returns its pixel labels. */
    Labels load_patch(const MemorySource& source, std::size_t index, Image& patch)
    {
        Image decoded;
        FileLabels file_labels;
        source.load(index, decoded, file_labels);

        if (decoded.channels() == K_GRAY_CHANNELS)
        {
            Image gray;
            resize_long_side(decoded.view(), K_SIDE, gray);
            expand_gray(gray.view(), patch);
        }
        else
        {
            resize_long_side(decoded.view(), K_SIDE, patch);
        }

        return to_pixel_labels(resample_polygons(file_labels), patch.size());
    }

    /**
     * @brief Builds a sample from a trace with the transforms called in the reference order.
     *
     * Mosaic, affine, colour jitter, flip (data/augment.py:2870-2912), each applied with
     * the plan the trace holds.
     */
    Sample compose_from_trace(const MemorySource& source, const TrainingSettings& settings,
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

            const MosaicPlan plan = plan_mosaic(
                settings.mosaic, trace.mosaic, settings.image_size,
                {patches[0].size(), patches[1].size(), patches[2].size(), patches[3].size()});
            apply_mosaic({patches[0].view(), patches[1].view(), patches[2].view(),
                          patches[3].view()},
                         plan, canvas);
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
} // namespace

/** @brief Tests that the settings default to the reference's training run. */
TEST(Unit_ImagingPipeline, SettingsDefaultToTheReferenceRun)
{
    const TrainingSettings settings;

    EXPECT_EQ(settings.image_size, 640);
    EXPECT_EQ(settings.epochs, 100);
    EXPECT_EQ(settings.close_mosaic, 10);
    EXPECT_EQ(settings.batch_size, 16);
    EXPECT_EQ(settings.seed, 0u);
    EXPECT_EQ(ValidationSettings{}.image_size, 640);
    EXPECT_FALSE(ValidationSettings{}.letterbox.scale_up);
}

/** @brief Tests that the mosaic closes exactly when the epoch reaches epochs minus close_mosaic. */
TEST(Unit_ImagingPipeline, MosaicClosesForTheLastTenEpochsOfAHundred)
{
    EXPECT_FALSE(is_mosaic_closed(0, 100, 10));
    EXPECT_FALSE(is_mosaic_closed(89, 100, 10));
    EXPECT_TRUE(is_mosaic_closed(90, 100, 10));
    EXPECT_TRUE(is_mosaic_closed(99, 100, 10));
    EXPECT_FALSE(is_mosaic_closed(99, 100, 0));
    EXPECT_TRUE(is_mosaic_closed(0, 5, 10));
}

/** @brief Tests that a sample's generator is the seed derived by the epoch, then by the index. */
TEST(Unit_ImagingPipeline, SampleRandomDerivesFromTheSeedTheEpochAndTheIndex)
{
    const Random random = make_sample_random(2026u, 3, 17);

    EXPECT_TRUE(random == Random(2026u).derive(3u).derive(17u));
    EXPECT_TRUE(random != make_sample_random(2027u, 3, 17));
    EXPECT_TRUE(random != make_sample_random(2026u, 4, 17));
    EXPECT_TRUE(random != make_sample_random(2026u, 3, 18));
    EXPECT_TRUE(random != make_sample_random(2026u, 17, 3));
}

/** @brief Tests that a sample draws partners, gate and centre, then affine, jitter and flip. */
TEST(Unit_ImagingPipeline, TrainingDrawsMosaicThenAffineThenHSVThenFlip)
{
    MemorySource source;
    fill_source(source);
    TrainingSettings settings = make_settings();
    settings.seed = 41;
    TrainingPipeline pipeline(source, settings, whole_dataset(source));
    Sample sample;

    const TrainingTrace trace = pipeline.make_sample(2, 5, sample);

    Random replica = make_sample_random(41u, 5, 2);
    DatasetPartners partners(source.size());
    const MosaicDraw mosaic = draw_mosaic(settings.mosaic, K_SIDE, 2, partners, replica);
    const AffinePlan affine = plan_affine(settings.affine, ImageSize{2 * K_SIDE, 2 * K_SIDE},
                                          ImageSize{K_SIDE, K_SIDE}, replica);
    const HSVPlan hsv = plan_hsv(settings.hsv, replica);
    const FlipPlan flip = plan_flip(settings.flip, ImageSize{K_SIDE, K_SIDE}, replica);

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
TEST(Unit_ImagingPipeline, TrainingLoadsTheSampleThenItsPartnersInDrawOrder)
{
    MemorySource source;
    fill_source(source);
    TrainingSettings settings = make_settings();
    settings.seed = 8;
    TrainingPipeline pipeline(source, settings, whole_dataset(source));
    Sample sample;

    const TrainingTrace trace = pipeline.make_sample(3, 0, sample);

    const std::vector<std::size_t> expected = {3, trace.mosaic.partners[0],
                                               trace.mosaic.partners[1],
                                               trace.mosaic.partners[2]};
    EXPECT_EQ(source.loads(), expected);
}

/** @brief Tests that the sample equals the transforms applied by hand in the reference order. */
TEST(Unit_ImagingPipeline, TrainingAppliesItsPlansInTheReferenceOrder)
{
    MemorySource source;
    fill_source(source);
    const TrainingSettings settings = make_settings();
    TrainingPipeline pipeline(source, settings, whole_dataset(source));

    for (std::int32_t epoch = 0; epoch < 2; ++epoch)
    {
        for (std::size_t index = 0; index < source.size(); ++index)
        {
            SCOPED_TRACE(::testing::Message() << "epoch " << epoch << ", index " << index);
            Sample sample;

            const TrainingTrace trace = pipeline.make_sample(index, epoch, sample);
            const Sample expected = compose_from_trace(source, settings, index, trace);

            EXPECT_TRUE(images_match(sample.image, expected.image));
            expect_same_labels(sample.labels, expected.labels);
        }
    }
}

/** @brief Tests that a flip probability of one mirrors the sample a probability of zero gives. */
TEST(Unit_ImagingPipeline, TrainingAppliesAForcedFlipAndSkipsAForbiddenOne)
{
    MemorySource source;
    fill_source(source);
    TrainingSettings always = make_settings();
    always.seed = 77;
    always.flip.probability = 1.0f;
    TrainingSettings never = make_settings();
    never.seed = 77;
    never.flip.probability = 0.0f;
    TrainingPipeline flipping(source, always, whole_dataset(source));
    TrainingPipeline plain(source, never, whole_dataset(source));
    Sample flipped;
    Sample straight;

    const TrainingTrace flipped_trace = flipping.make_sample(0, 0, flipped);
    const TrainingTrace straight_trace = plain.make_sample(0, 0, straight);

    ASSERT_TRUE(flipped_trace.flip.is_flipped);
    ASSERT_FALSE(straight_trace.flip.is_flipped);
    ASSERT_EQ(flipped.image.size(), straight.image.size());
    std::size_t wrong = 0;
    for (std::int32_t y = 0; y < K_SIDE; ++y)
    {
        for (std::int32_t x = 0; x < K_SIDE; ++x)
        {
            if (std::memcmp(flipped.image.row(y) + 3 * x,
                            straight.image.row(y) + 3 * (K_SIDE - 1 - x), 3) != 0)
                ++wrong;
        }
    }

    EXPECT_EQ(wrong, 0u) << "the flipped sample is not the mirror of the unflipped one";
    EXPECT_TRUE(images_match(
        flipped.image, compose_from_trace(source, always, 0, flipped_trace).image));
}

/** @brief Tests that a sample has the training size, three channels and its own image's sizes. */
TEST(Unit_ImagingPipeline, TrainingProducesTheTrainingSizeAndTheSizesOfItsOwnImage)
{
    MemorySource source;
    fill_source(source);
    TrainingPipeline pipeline(source, make_settings());
    Sample sample;

    const TrainingTrace trace = pipeline.make_sample(0, 0, sample);

    EXPECT_EQ(sample.image.size(), (ImageSize{K_SIDE, K_SIDE}));
    EXPECT_EQ(sample.image.channels(), K_RGB_CHANNELS);
    EXPECT_EQ(sample.original_size, (ImageSize{32, 64}));
    EXPECT_EQ(sample.loaded_size, (ImageSize{16, 32}));
    EXPECT_EQ(trace.affine.source, (ImageSize{2 * K_SIDE, 2 * K_SIDE}));
    EXPECT_EQ(trace.affine.output, (ImageSize{K_SIDE, K_SIDE}));
}

/** @brief Tests that the quadrant of at least 16 by 16 pixels keeps its box in the output. */
TEST(Unit_ImagingPipeline, MosaicSampleKeepsItsLargestQuadrantBoxInsideTheTrainingSize)
{
    MemorySource source;
    for (std::int32_t i = 0; i < 4; ++i)
    {
        source.add(make_textured(ImageSize{K_SIDE, K_SIDE}, 50 * i),
                   one_object(i, NormalizedBox{0.5f, 0.5f, 1.0f, 1.0f}));
    }

    TrainingSettings settings = make_settings();
    settings.affine.scale = 0.0f;
    settings.affine.translate = 0.0f;
    settings.flip.probability = 0.0f;
    const float side = static_cast<float>(K_SIDE);

    for (std::uint64_t seed = 30; seed < 38; ++seed)
    {
        SCOPED_TRACE(::testing::Message() << "seed " << seed);
        settings.seed = seed;
        TrainingPipeline pipeline(source, settings, whole_dataset(source));
        Sample sample;

        const TrainingTrace trace = pipeline.make_sample(0, 0, sample);

        // The warp shows canvas columns and rows 16 to 48, so the centre lands at centre - 16.
        ASSERT_TRUE(trace.mosaic.is_enabled);
        const float split_x = static_cast<float>(trace.mosaic.center_x - K_SIDE / 2);
        const float split_y = static_cast<float>(trace.mosaic.center_y - K_SIDE / 2);
        const bool is_left_wider = trace.mosaic.center_x >= K_SIDE;
        const bool is_top_taller = trace.mosaic.center_y >= K_SIDE;
        const PixelBox largest{is_left_wider ? 0.0f : split_x, is_top_taller ? 0.0f : split_y,
                               is_left_wider ? split_x : side, is_top_taller ? split_y : side};

        ASSERT_FALSE(sample.labels.is_empty());
        std::size_t matches = 0;
        for (const PixelBox& box : sample.labels.boxes())
        {
            EXPECT_GE(box.x1, 0.0f);
            EXPECT_GE(box.y1, 0.0f);
            EXPECT_LE(box.x2, side);
            EXPECT_LE(box.y2, side);
            EXPECT_LT(box.x1, box.x2);
            EXPECT_LT(box.y1, box.y2);
            if (std::abs(box.x1 - largest.x1) <= K_BOX_TOLERANCE &&
                std::abs(box.y1 - largest.y1) <= K_BOX_TOLERANCE &&
                std::abs(box.x2 - largest.x2) <= K_BOX_TOLERANCE &&
                std::abs(box.y2 - largest.y2) <= K_BOX_TOLERANCE)
                ++matches;
        }

        EXPECT_EQ(matches, 1u) << "the quadrant of at least 16 by 16 pixels lost its box";
    }
}

/** @brief Tests that a closed epoch loads one image and draws affine, jitter and flip only. */
TEST(Unit_ImagingPipeline, ClosedEpochSkipsTheMosaicAndItsDraws)
{
    MemorySource source;
    fill_source(source);
    TrainingSettings settings = make_settings();
    settings.epochs = 10;
    settings.close_mosaic = 3;
    settings.seed = 12;
    TrainingPipeline pipeline(source, settings);
    Sample sample;

    const TrainingTrace trace = pipeline.make_sample(0, 7, sample);

    Random replica = make_sample_random(12u, 7, 0);
    const AffinePlan affine = plan_affine(settings.affine, ImageSize{16, 32},
                                          ImageSize{K_SIDE, K_SIDE}, replica);
    const HSVPlan hsv = plan_hsv(settings.hsv, replica);
    const FlipPlan flip = plan_flip(settings.flip, ImageSize{K_SIDE, K_SIDE}, replica);

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
TEST(Unit_ImagingPipeline, SampleWithoutMosaicKeepsTheBoxOfItsOwnImage)
{
    MemorySource source;
    fill_source(source);
    TrainingSettings settings = make_settings();
    settings.epochs = 10;
    settings.close_mosaic = 3;
    settings.affine.scale = 0.0f;
    settings.affine.translate = 0.0f;
    TrainingPipeline pipeline(source, settings);
    Sample sample;

    const TrainingTrace trace = pipeline.make_sample(0, 7, sample);

    // The box (8, 4, 24, 12) of the 16 by 32 image moves down by 8 and mirrors onto itself.
    ASSERT_FALSE(trace.mosaic.is_enabled);
    ASSERT_EQ(sample.labels.size(), 1u);
    EXPECT_EQ(sample.labels.classes()[0], 0);
    EXPECT_NEAR(sample.labels.boxes()[0].x1, 8.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(sample.labels.boxes()[0].y1, 12.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(sample.labels.boxes()[0].x2, 24.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(sample.labels.boxes()[0].y2, 20.0f, K_BOX_TOLERANCE);
}

/** @brief Tests that the last open epoch still composes a mosaic of four loads. */
TEST(Unit_ImagingPipeline, EpochBeforeTheClosedOnesStillMakesAMosaic)
{
    MemorySource source;
    fill_source(source);
    TrainingSettings settings = make_settings();
    settings.epochs = 10;
    settings.close_mosaic = 3;
    TrainingPipeline pipeline(source, settings);
    Sample sample;

    const TrainingTrace trace = pipeline.make_sample(0, 6, sample);

    EXPECT_FALSE(trace.is_mosaic_closed);
    EXPECT_TRUE(trace.mosaic.is_enabled);
    EXPECT_EQ(source.loads().size(), K_MOSAIC_PATCHES);
}

/** @brief Tests that a gate that stays shut in an open epoch warps the loaded image alone. */
TEST(Unit_ImagingPipeline, ClosedGateInAnOpenEpochWarpsTheLoadedImage)
{
    MemorySource source;
    fill_source(source);
    TrainingSettings settings = make_settings();
    settings.mosaic.probability = 0.0f;
    settings.seed = 3;
    TrainingPipeline pipeline(source, settings);
    Sample sample;

    const TrainingTrace trace = pipeline.make_sample(2, 0, sample);

    EXPECT_FALSE(trace.is_mosaic_closed);
    EXPECT_FALSE(trace.mosaic.is_enabled);
    EXPECT_EQ(trace.affine.source, (ImageSize{32, 16}));
    EXPECT_EQ(source.loads(), std::vector<std::size_t>{2});
    EXPECT_EQ(sample.image.size(), (ImageSize{K_SIDE, K_SIDE}));
    EXPECT_TRUE(images_match(sample.image,
                             compose_from_trace(source, settings, 2, trace).image));
}

/** @brief Tests that one seed repeats a sample and that another seed, epoch or index changes it. */
TEST(Unit_ImagingPipeline, SampleDependsOnTheSeedTheEpochAndTheIndexAlone)
{
    MemorySource source;
    fill_source(source);
    TrainingSettings settings = make_settings();
    settings.seed = 2026;
    TrainingSettings other_settings = make_settings();
    other_settings.seed = 2027;
    TrainingPipeline pipeline(source, settings, whole_dataset(source));
    TrainingPipeline other_pipeline(source, other_settings, whole_dataset(source));
    Sample first;
    Sample other_seed;
    Sample other_epoch;
    Sample again;

    const TrainingTrace first_trace = pipeline.make_sample(1, 0, first);
    const TrainingTrace seed_trace = other_pipeline.make_sample(1, 0, other_seed);
    const TrainingTrace epoch_trace = pipeline.make_sample(1, 1, other_epoch);
    const TrainingTrace again_trace = pipeline.make_sample(1, 0, again);

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
TEST(Unit_ImagingPipeline, EarlierSamplesDoNotShiftTheDrawsOfALaterOne)
{
    MemorySource source;
    fill_source(source);
    TrainingPipeline used(source, make_settings(), whole_dataset(source));
    TrainingPipeline fresh(source, make_settings(), whole_dataset(source));
    Sample warm;
    Sample from_used;
    Sample from_fresh;

    // A mosaic sample makes six draws more than a sample of a closed epoch.
    static_cast<void>(used.make_sample(4, 0, warm));
    static_cast<void>(used.make_sample(2, 95, warm));
    const TrainingTrace used_trace = used.make_sample(0, 0, from_used);
    const TrainingTrace fresh_trace = fresh.make_sample(0, 0, from_fresh);

    EXPECT_TRUE(images_match(from_used.image, from_fresh.image));
    expect_same_labels(from_used.labels, from_fresh.labels);
    expect_same_affine(used_trace.affine, fresh_trace.affine);
    expect_same_hsv(used_trace.hsv, fresh_trace.hsv);
    EXPECT_EQ(used_trace.mosaic.partners, fresh_trace.mosaic.partners);
    EXPECT_EQ(used_trace.flip.is_flipped, fresh_trace.flip.is_flipped);
}

/** @brief Tests that a pipeline built without a partner source draws from its recent loads. */
TEST(Unit_ImagingPipeline, DefaultPartnersComeFromTheImagesThePipelineLoaded)
{
    MemorySource source;
    fill_source(source);
    TrainingPipeline pipeline(source, make_settings());
    Sample sample;

    const TrainingTrace first = pipeline.make_sample(3, 0, sample);
    const TrainingTrace second = pipeline.make_sample(1, 0, sample);

    EXPECT_EQ(first.mosaic.partners, (MosaicPartners{3, 3, 3}));
    for (const std::size_t partner : second.mosaic.partners)
        EXPECT_TRUE(partner == 3 || partner == 1) << "partner " << partner << " was never loaded";
}

/** @brief Tests that a pipeline loads the partners the source it was given chooses. */
TEST(Unit_ImagingPipeline, GivenPartnerSourceChoosesThePartners)
{
    MemorySource source;
    fill_source(source);
    TrainingPipeline pipeline(source, make_settings(),
                              std::make_unique<FixedPartners>(MosaicPartners{4, 4, 0}));
    Sample sample;

    const TrainingTrace trace = pipeline.make_sample(2, 0, sample);

    ASSERT_TRUE(trace.mosaic.is_enabled);
    EXPECT_EQ(trace.mosaic.partners, (MosaicPartners{4, 4, 0}));
    EXPECT_EQ(source.loads(), (std::vector<std::size_t>{2, 4, 4, 0}));
}

/** @brief Tests that every setting is checked when the pipeline is constructed. */
TEST(Unit_ImagingPipeline, TrainingRejectsEverySettingOutOfRangeAtConstruction)
{
    MemorySource source;
    fill_source(source);
    TrainingSettings no_size = make_settings();
    no_size.image_size = 0;
    TrainingSettings large_size = make_settings();
    large_size.image_size = K_MAX_IMAGE_SIDE / 2 + 1;
    TrainingSettings no_epochs = make_settings();
    no_epochs.epochs = 0;
    TrainingSettings negative_close = make_settings();
    negative_close.close_mosaic = -1;
    TrainingSettings no_batch = make_settings();
    no_batch.batch_size = 0;
    TrainingSettings bad_mosaic = make_settings();
    bad_mosaic.mosaic.probability = 1.5f;
    TrainingSettings bad_affine = make_settings();
    bad_affine.affine.scale = 1.0f;
    TrainingSettings bad_hsv = make_settings();
    bad_hsv.hsv.value_gain = -0.4f;
    TrainingSettings bad_flip = make_settings();
    bad_flip.flip.probability = std::numeric_limits<float>::quiet_NaN();

    SA_EXPECT_ERROR(TrainingPipeline(source, no_size), "validate_mosaic_size",
                    "image_size of 0");
    SA_EXPECT_ERROR(TrainingPipeline(source, large_size), "validate_mosaic_size",
                    "image_size of 8193", "[1, 8192]");
    SA_EXPECT_ERROR(TrainingPipeline(source, no_epochs), "validate_training", "epoch count");
    SA_EXPECT_ERROR(TrainingPipeline(source, negative_close), "validate_training",
                    "close_mosaic of -1");
    SA_EXPECT_ERROR(TrainingPipeline(source, no_batch), "validate_training", "batch_size of 0");
    SA_EXPECT_ERROR(TrainingPipeline(source, bad_mosaic), "validate_mosaic", "probability");
    SA_EXPECT_ERROR(TrainingPipeline(source, bad_affine), "validate_affine", "scale");
    SA_EXPECT_ERROR(TrainingPipeline(source, bad_hsv), "validate_hsv", "gain");
    SA_EXPECT_ERROR(TrainingPipeline(source, bad_flip), "validate_flip", "probability");
    SA_EXPECT_ERROR(TrainingPipeline(source, bad_flip, whole_dataset(source)), "validate_flip",
                    "probability");
    EXPECT_TRUE(source.loads().empty()) << "a rejected construction loaded an image";
    EXPECT_NO_THROW(validate_training(make_settings()));
}

/** @brief Tests that an empty source and a null partner source are refused at construction. */
TEST(Unit_ImagingPipeline, TrainingRejectsAnEmptySourceAndANullPartnerSource)
{
    MemorySource source;
    fill_source(source);
    MemorySource empty;

    SA_EXPECT_ERROR(TrainingPipeline(empty, make_settings()), "TrainingPipeline", "no image");
    SA_EXPECT_ERROR(TrainingPipeline(empty, make_settings(), whole_dataset(source)),
                    "TrainingPipeline", "no image");
    SA_EXPECT_ERROR(TrainingPipeline(source, make_settings(), nullptr), "TrainingPipeline",
                    "partner source is null");
}

/** @brief Tests that an index or an epoch outside its range is refused. */
TEST(Unit_ImagingPipeline, TrainingRejectsIndicesAndEpochsOutsideTheirRange)
{
    MemorySource source;
    fill_source(source);
    TrainingPipeline pipeline(source, make_settings());
    Sample sample;

    SA_EXPECT_ERROR(static_cast<void>(pipeline.make_sample(5, 0, sample)),
                    "TrainingPipeline::make_sample", "index 5");
    SA_EXPECT_ERROR(static_cast<void>(pipeline.make_sample(0, -1, sample)),
                    "TrainingPipeline::make_sample", "epoch -1");
    SA_EXPECT_ERROR(static_cast<void>(pipeline.make_sample(0, 100, sample)),
                    "TrainingPipeline::make_sample", "epoch 100");
}

/** @brief Tests that a one-channel image leaves the training pipeline with three equal channels. */
TEST(Unit_ImagingPipeline, TrainingExpandsAGreyImageToRGB)
{
    MemorySource source;
    source.add(make_flat(ImageSize{K_SIDE, K_SIDE}, 1, 77), FileLabels{});
    TrainingPipeline pipeline(source, make_still_settings());
    Sample sample;

    static_cast<void>(pipeline.make_sample(0, 0, sample));

    ASSERT_EQ(sample.image.size(), (ImageSize{K_SIDE, K_SIDE}));
    ASSERT_EQ(sample.image.channels(), K_RGB_CHANNELS);
    const std::uint8_t* centre = sample.image.row(K_SIDE / 2) + 3 * (K_SIDE / 2);
    EXPECT_EQ(centre[0], 77);
    EXPECT_EQ(centre[1], 77);
    EXPECT_EQ(centre[2], 77);
}

/** @brief Tests that a mosaic of grey and colour images is composed in three channels. */
TEST(Unit_ImagingPipeline, TrainingComposesAMosaicOfGreyAndColourImages)
{
    MemorySource source;
    source.add(make_flat(ImageSize{K_SIDE, K_SIDE}, 1, 40), FileLabels{});
    source.add(make_constant(ImageSize{K_SIDE, K_SIDE}, 200, 10, 90), FileLabels{});
    source.add(make_flat(ImageSize{16, K_SIDE}, 1, 220), FileLabels{});
    const TrainingSettings settings = make_settings();
    TrainingPipeline pipeline(source, settings,
                              std::make_unique<FixedPartners>(MosaicPartners{1, 2, 0}));
    Sample sample;

    const TrainingTrace trace = pipeline.make_sample(0, 0, sample);

    ASSERT_TRUE(trace.mosaic.is_enabled);
    EXPECT_EQ(sample.image.channels(), K_RGB_CHANNELS);
    EXPECT_EQ(sample.image.size(), (ImageSize{K_SIDE, K_SIDE}));
    EXPECT_TRUE(images_match(sample.image,
                             compose_from_trace(source, settings, 0, trace).image));
}

/** @brief Tests that an image of neither one nor three channels is refused with its index. */
TEST(Unit_ImagingPipeline, TrainingRejectsAnImageOfFourChannels)
{
    MemorySource source;
    source.add(make_flat(ImageSize{K_SIDE, K_SIDE}, 4, 9), FileLabels{});
    TrainingPipeline pipeline(source, make_still_settings());
    Sample sample;

    SA_EXPECT_ERROR(static_cast<void>(pipeline.make_sample(0, 0, sample)),
                    "TrainingPipeline::make_sample", "4 channels", "image 0");
}

/** @brief Tests that a polygon label is resampled, carried through and bounds its box. */
TEST(Unit_ImagingPipeline, TrainingCarriesAPolygonAndTakesTheBoxFromIt)
{
    MemorySource source;
    FileLabels labels;
    labels.add(5, NormalizedBox{0.5f, 0.5f, 0.5f, 0.5f},
               Polygon{Point{0.25f, 0.25f}, Point{0.75f, 0.25f}, Point{0.5f, 0.75f}});
    source.add(make_textured(ImageSize{K_SIDE, K_SIDE}, 5), labels);
    TrainingPipeline pipeline(source, make_still_settings());
    Sample sample;

    static_cast<void>(pipeline.make_sample(0, 0, sample));

    // The still settings map the 32 by 32 image onto itself, so the triangle stays put.
    ASSERT_EQ(sample.labels.size(), 1u);
    EXPECT_EQ(sample.labels.classes()[0], 5);
    EXPECT_NEAR(sample.labels.boxes()[0].x1, 8.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(sample.labels.boxes()[0].y1, 8.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(sample.labels.boxes()[0].x2, 24.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(sample.labels.boxes()[0].y2, 24.0f, K_BOX_TOLERANCE);
    ASSERT_EQ(sample.labels.polygons()[0].size(), K_POLYGON_POINTS);
    EXPECT_NEAR(sample.labels.polygons()[0][0].x, 8.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(sample.labels.polygons()[0][0].y, 8.0f, K_BOX_TOLERANCE);
}

/** @brief Tests that validation resizes the long side and then only pads. */
TEST(Unit_ImagingPipeline, ValidationResizesTheLongSideThenOnlyPads)
{
    MemorySource source;
    source.add(make_constant(ImageSize{8, 16}, 10, 200, 30),
               one_object(4, NormalizedBox{0.5f, 0.5f, 0.5f, 0.5f}));
    ValidationSettings settings;
    settings.image_size = K_SIDE;
    ValidationPipeline pipeline(source, settings);
    Sample sample;

    const LetterboxPlan plan = pipeline.make_sample(0, ImageSize{K_SIDE, K_SIDE}, sample);

    EXPECT_EQ(sample.original_size, (ImageSize{8, 16}));
    EXPECT_EQ(sample.loaded_size, (ImageSize{16, 32}));
    EXPECT_EQ(plan.source, (ImageSize{16, 32}));
    EXPECT_EQ(plan.target, (ImageSize{K_SIDE, K_SIDE}));
    EXPECT_EQ(plan.resized, (ImageSize{16, 32}));
    EXPECT_NEAR(plan.ratio, 1.0, 1e-12);
    EXPECT_EQ(plan.top, 8);
    EXPECT_EQ(plan.bottom, 8);
    EXPECT_EQ(plan.left, 0);
    EXPECT_EQ(plan.right, 0);

    ASSERT_EQ(sample.image.size(), (ImageSize{K_SIDE, K_SIDE}));
    ASSERT_EQ(sample.image.channels(), 3);
    std::size_t wrong = 0;
    for (std::int32_t y = 0; y < K_SIDE; ++y)
    {
        const bool is_padding = y < 8 || y >= 24;
        const std::array<std::uint8_t, 3> expected =
            is_padding ? std::array<std::uint8_t, 3>{114, 114, 114}
                       : std::array<std::uint8_t, 3>{10, 200, 30};
        for (std::int32_t x = 0; x < K_SIDE; ++x)
        {
            if (std::memcmp(sample.image.row(y) + 3 * x, expected.data(), 3) != 0)
                ++wrong;
        }
    }

    EXPECT_EQ(wrong, 0u) << "a constant image must come out constant between two bands of 114";
    EXPECT_EQ(source.loads(), std::vector<std::size_t>{0});
}

/** @brief Tests that validation moves a box by the padding and clips nothing. */
TEST(Unit_ImagingPipeline, ValidationMovesBoxesByThePaddingWithoutClipping)
{
    MemorySource source;
    source.add(make_constant(ImageSize{8, 16}, 10, 200, 30),
               one_object(4, NormalizedBox{0.5f, 0.5f, 0.5f, 0.5f}));
    ValidationSettings settings;
    settings.image_size = K_SIDE;
    ValidationPipeline pipeline(source, settings);
    Sample sample;

    static_cast<void>(pipeline.make_sample(0, ImageSize{K_SIDE, K_SIDE}, sample));

    // On the loaded 16 by 32 image the box is (8, 4, 24, 12); the top padding adds 8 to y.
    ASSERT_EQ(sample.labels.size(), 1u);
    EXPECT_EQ(sample.labels.classes()[0], 4);
    EXPECT_NEAR(sample.labels.boxes()[0].x1, 8.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(sample.labels.boxes()[0].y1, 12.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(sample.labels.boxes()[0].x2, 24.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(sample.labels.boxes()[0].y2, 20.0f, K_BOX_TOLERANCE);
}

/** @brief Tests that a target of the loaded size adds no padding and copies the image. */
TEST(Unit_ImagingPipeline, ValidationIntoARectangularTargetAddsNoPadding)
{
    MemorySource source;
    source.add(make_textured(ImageSize{16, 32}, 17), FileLabels{});
    ValidationSettings settings;
    settings.image_size = K_SIDE;
    ValidationPipeline pipeline(source, settings);
    Sample sample;

    const LetterboxPlan plan = pipeline.make_sample(0, ImageSize{16, 32}, sample);

    EXPECT_EQ(plan.top + plan.bottom + plan.left + plan.right, 0);
    EXPECT_TRUE(images_match(sample.image, make_textured(ImageSize{16, 32}, 17)));
    EXPECT_TRUE(sample.labels.is_empty());
}

/** @brief Tests that a validation sample does not depend on the samples made before it. */
TEST(Unit_ImagingPipeline, ValidationRepeatsItsSampleAfterAnotherIndex)
{
    MemorySource source;
    fill_source(source);
    ValidationSettings settings;
    settings.image_size = K_SIDE;
    ValidationPipeline pipeline(source, settings);
    Sample first;
    Sample between;
    Sample again;

    static_cast<void>(pipeline.make_sample(0, ImageSize{K_SIDE, K_SIDE}, first));
    static_cast<void>(pipeline.make_sample(2, ImageSize{K_SIDE, K_SIDE}, between));
    static_cast<void>(pipeline.make_sample(0, ImageSize{K_SIDE, K_SIDE}, again));

    EXPECT_TRUE(images_match(again.image, first.image));
    expect_same_labels(again.labels, first.labels);
    EXPECT_FALSE(images_match(between.image, first.image));
}

/** @brief Tests that validation expands a grey image and pads it in three channels. */
TEST(Unit_ImagingPipeline, ValidationExpandsAGreyImageToRGB)
{
    MemorySource source;
    source.add(make_flat(ImageSize{8, 16}, 1, 61), FileLabels{});
    ValidationSettings settings;
    settings.image_size = K_SIDE;
    ValidationPipeline pipeline(source, settings);
    Sample sample;

    static_cast<void>(pipeline.make_sample(0, ImageSize{K_SIDE, K_SIDE}, sample));

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
TEST(Unit_ImagingPipeline, ValidationCarriesAPolygonThroughTheLetterbox)
{
    MemorySource source;
    FileLabels labels;
    labels.add(2, NormalizedBox{0.5f, 0.5f, 0.5f, 0.5f},
               Polygon{Point{0.25f, 0.25f}, Point{0.75f, 0.25f}, Point{0.5f, 0.75f}});
    source.add(make_constant(ImageSize{8, 16}, 10, 200, 30), labels);
    ValidationSettings settings;
    settings.image_size = K_SIDE;
    ValidationPipeline pipeline(source, settings);
    Sample sample;

    static_cast<void>(pipeline.make_sample(0, ImageSize{K_SIDE, K_SIDE}, sample));

    // On the loaded 16 by 32 image the first vertex is (8, 4); the top padding adds 8 to y.
    ASSERT_EQ(sample.labels.size(), 1u);
    ASSERT_EQ(sample.labels.polygons()[0].size(), K_POLYGON_POINTS);
    EXPECT_NEAR(sample.labels.polygons()[0][0].x, 8.0f, K_BOX_TOLERANCE);
    EXPECT_NEAR(sample.labels.polygons()[0][0].y, 12.0f, K_BOX_TOLERANCE);
}

/** @brief Tests that validation checks its settings at construction and its index per sample. */
TEST(Unit_ImagingPipeline, ValidationRejectsSettingsAndIndicesOutsideTheirRange)
{
    MemorySource source;
    fill_source(source);
    MemorySource empty;
    ValidationSettings no_size;
    no_size.image_size = 0;
    ValidationSettings settings;
    settings.image_size = K_SIDE;
    ValidationPipeline pipeline(source, settings);
    Sample sample;

    SA_EXPECT_ERROR(ValidationPipeline(source, no_size), "validate_validation",
                    "image_size of 0");
    SA_EXPECT_ERROR(ValidationPipeline(empty, settings), "ValidationPipeline", "no image");
    SA_EXPECT_ERROR(
        static_cast<void>(pipeline.make_sample(9, ImageSize{K_SIDE, K_SIDE}, sample)),
        "ValidationPipeline::make_sample", "index 9");
    EXPECT_NO_THROW(validate_validation(settings));
}
