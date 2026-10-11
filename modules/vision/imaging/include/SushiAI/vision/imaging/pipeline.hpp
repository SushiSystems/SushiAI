/**************************************************************************/
/* pipeline.hpp                                                           */
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
 * @file pipeline.hpp
 * @brief Declares the training and validation pipelines that turn a dataset index into a sample.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include <SushiAI/vision/imaging/affine.hpp>
#include <SushiAI/vision/imaging/flip.hpp>
#include <SushiAI/vision/imaging/hsv.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/label_set.hpp>
#include <SushiAI/vision/imaging/letterbox.hpp>
#include <SushiAI/vision/imaging/mosaic.hpp>
#include <SushiAI/vision/imaging/partner_source.hpp>
#include <SushiAI/vision/imaging/random.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Vision::Imaging
{
    /** @brief Holds one finished sample: an RGB image and its labels in that image's pixels. */
    struct Sample
    {
        /** @brief Holds the image a model is fed: 8-bit RGB, K_RGB_CHANNELS channels. */
        Image image;

        /** @brief Holds the classes, the boxes and the polygons in the pixels of image. */
        Labels labels;

        /** @brief Holds the size of the decoded image the sample's index names. */
        ImageSize original_size{};

        /** @brief Holds that image's size after resize_long_side. */
        ImageSize loaded_size{};
    };

    /**
     * @brief Supplies decoded images and their file labels by index.
     *
     * The pipelines read a dataset through this seam, so a test supplies images from
     * memory and a run supplies them from raw image files.
     */
    class SampleSource
    {
        public:
            /** @brief Destroys the source through a base pointer. */
            virtual ~SampleSource() = default;

            /** @brief Returns the number of images, above zero. */
            [[nodiscard]] virtual std::size_t size() const noexcept = 0;

            /**
             * @brief Loads one decoded image at its stored resolution with its labels.
             *
             * The image has one channel or K_RGB_CHANNELS; a pipeline expands the first to
             * the second. Throws Error when the image cannot be read. Callable from several
             * threads at once when one source feeds several pipelines.
             * @param index In [0, size()).
             */
            virtual void load(std::size_t index, Image& image, FileLabels& labels) const = 0;
    };

    /** @brief Holds what the caller chooses about the training pipeline. */
    struct TrainingSettings
    {
        /** @brief Holds the side s of the square output, in pixels (cfg/default.yaml:16). */
        std::int32_t image_size = 640;

        /** @brief Counts the epochs of the run (cfg/default.yaml:12). */
        std::int32_t epochs = 100;

        /** @brief Counts the final epochs that run without mosaic (cfg/default.yaml:34). */
        std::int32_t close_mosaic = 10;

        /** @brief Counts the images of a batch, for RecentPartners (cfg/default.yaml:15). */
        std::int32_t batch_size = 16;

        /** @brief Holds the seed every sample's generator derives from (cfg/default.yaml:29). */
        std::uint64_t seed = 0;

        /** @brief Holds the mosaic settings. */
        MosaicSettings mosaic{};

        /** @brief Holds the affine settings. */
        AffineSettings affine{};

        /** @brief Holds the colour jitter settings. */
        HSVSettings hsv{};

        /** @brief Holds the horizontal flip settings. */
        FlipSettings flip{};
    };

    /** @brief Holds what the training pipeline drew and planned for one sample. */
    struct TrainingTrace
    {
        /** @brief Tells whether the epoch was one of the closed ones; mosaic then holds no draw. */
        bool is_mosaic_closed = false;

        /** @brief Holds the mosaic draw. */
        MosaicDraw mosaic{};

        /** @brief Holds the affine plan. */
        AffinePlan affine{};

        /** @brief Holds the colour jitter plan. */
        HSVPlan hsv{};

        /** @brief Holds the flip plan. */
        FlipPlan flip{};
    };

    /**
     * @brief Reports whether an epoch runs without mosaic.
     * @param epoch Zero-based epoch index.
     * @return epoch >= epochs - close_mosaic (engine/trainer.py:505).
     */
    [[nodiscard]] bool is_mosaic_closed(std::int32_t epoch, std::int32_t epochs,
                                        std::int32_t close_mosaic) noexcept;

    /**
     * @brief Throws Error on a training setting outside its range.
     *
     * image_size must pass validate_mosaic_size; epochs and batch_size above zero, close_mosaic
     * not below zero, then validate_mosaic, validate_affine, validate_hsv and validate_flip.
     */
    void validate_training(const TrainingSettings& settings);

    /**
     * @brief Returns the generator that makes every draw of one sample.
     * @return Random(seed).derive(epoch).derive(index), so no sample's draws shift another's.
     */
    [[nodiscard]] Random make_sample_random(std::uint64_t seed, std::int32_t epoch,
                                            std::size_t index) noexcept;

    /**
     * @brief Produces training samples: load, mosaic, affine, colour jitter, flip.
     *
     * One pipeline serves one thread, because it reuses its images between samples and its
     * partner source may keep state.
     */
    class TrainingPipeline final
    {
        public:
            /**
             * @brief Constructs a pipeline whose partners come from a RecentPartners source.
             * @param source Must outlive the pipeline and hold at least one image.
             * @param settings Copied; must pass validate_training, otherwise throws Error.
             */
            TrainingPipeline(const SampleSource& source, const TrainingSettings& settings);

            /**
             * @brief Constructs a pipeline that takes its mosaic partners from a given source.
             * @param source Must outlive the pipeline and hold at least one image.
             * @param settings Copied; must pass validate_training, otherwise throws Error.
             * @param partners Not null; otherwise throws Error.
             */
            TrainingPipeline(const SampleSource& source, const TrainingSettings& settings,
                             std::unique_ptr<PartnerSource> partners);

            /** @brief Forbids copying, which would share the source and duplicate the images. */
            TrainingPipeline(const TrainingPipeline&) = delete;

            /** @brief Forbids copy assignment for the same reason. */
            TrainingPipeline& operator=(const TrainingPipeline&) = delete;

            /** @brief Moves the pipeline with its images. */
            TrainingPipeline(TrainingPipeline&&) noexcept = default;

            /** @brief Moves the pipeline with its images. */
            TrainingPipeline& operator=(TrainingPipeline&&) noexcept = default;

            /** @brief Destroys the pipeline and its images. */
            ~TrainingPipeline() = default;

            /**
             * @brief Produces the sample of one index in one epoch.
             *
             * Every draw comes from make_sample_random(settings().seed, epoch, index). The
             * image is loaded, resized and made RGB; unless the epoch is closed, draw_mosaic
             * runs and may compose a mosaic; then affine, colour jitter and flip are applied.
             * @param sample Receives the s by s image, its labels and the sizes of its own image.
             */
            TrainingTrace make_sample(std::size_t index, std::int32_t epoch, Sample& sample);

            /** @brief Returns the settings the pipeline was constructed with. */
            [[nodiscard]] const TrainingSettings& settings() const noexcept { return settings_; }

        private:
            /// @brief Points at the source, which the caller keeps alive.
            const SampleSource* source_ = nullptr;

            /// @brief Holds the settings.
            TrainingSettings settings_{};

            /// @brief Holds the source of the mosaic partners.
            std::unique_ptr<PartnerSource> partners_;

            /// @brief Holds the image a load returns before it is resized.
            Image decoded_;

            /// @brief Holds a resized one-channel image before it is expanded to RGB.
            Image gray_;

            /// @brief Holds the resized image of the sample and of each mosaic partner.
            std::array<Image, K_MOSAIC_PATCHES> patches_{};

            /// @brief Holds the mosaic canvas, then the jittered image.
            Image first_;

            /// @brief Holds the warped image.
            Image second_;
    };

    /** @brief Holds what the caller chooses about the validation pipeline. */
    struct ValidationSettings
    {
        /** @brief Holds the long side an image is loaded at, in pixels (cfg/default.yaml:16). */
        std::int32_t image_size = 640;

        /** @brief Holds the letterbox settings. */
        LetterboxSettings letterbox{};
    };

    /**
     * @brief Throws Error on a validation setting outside its range.
     *
     * image_size must pass is_valid_side; then validate_letterbox.
     */
    void validate_validation(const ValidationSettings& settings);

    /**
     * @brief Produces validation samples: load, then letterbox. It draws nothing.
     *
     * One pipeline serves one thread, because it reuses its images between samples.
     */
    class ValidationPipeline final
    {
        public:
            /**
             * @brief Constructs a pipeline over a source.
             * @param source Must outlive the pipeline and hold at least one image.
             * @param settings Copied; must pass validate_validation, otherwise throws Error.
             */
            ValidationPipeline(const SampleSource& source, const ValidationSettings& settings);

            /** @brief Forbids copying, which would share the source and duplicate the images. */
            ValidationPipeline(const ValidationPipeline&) = delete;

            /** @brief Forbids copy assignment for the same reason. */
            ValidationPipeline& operator=(const ValidationPipeline&) = delete;

            /** @brief Moves the pipeline with its images. */
            ValidationPipeline(ValidationPipeline&&) noexcept = default;

            /** @brief Moves the pipeline with its images. */
            ValidationPipeline& operator=(ValidationPipeline&&) noexcept = default;

            /** @brief Destroys the pipeline and its images. */
            ~ValidationPipeline() = default;

            /**
             * @brief Produces the sample of one index, letterboxed to a target size.
             *
             * Loads the image, resizes it with resize_long_side, makes it RGB, converts its
             * labels to pixels, then plans and applies the letterbox.
             * @param target The size of the result: s by s, or a plan_rectangular_batches shape.
             * @return The letterbox plan, whose ratio, left and top map a prediction back.
             */
            LetterboxPlan make_sample(std::size_t index, ImageSize target, Sample& sample);

            /** @brief Returns the settings the pipeline was constructed with. */
            [[nodiscard]] const ValidationSettings& settings() const noexcept { return settings_; }

        private:
            /// @brief Points at the source, which the caller keeps alive.
            const SampleSource* source_ = nullptr;

            /// @brief Holds the settings.
            ValidationSettings settings_{};

            /// @brief Holds the image a load returns before it is resized.
            Image decoded_;

            /// @brief Holds a resized one-channel image before it is expanded to RGB.
            Image gray_;

            /// @brief Holds the resized image before the letterbox.
            Image loaded_;
    };
} // namespace SushiAI::Vision::Imaging
