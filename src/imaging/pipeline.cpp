/**************************************************************************/
/* pipeline.cpp                                                           */
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
 * @file pipeline.cpp
 * @brief Implements the training and validation pipelines that turn a dataset index into a sample.
 * @author Mustafa Garip
 */

#include <SushiAI/imaging/pipeline.hpp>

#include <string_view>
#include <utility>

#include <SushiAI/core/common.hpp>

#include <SushiAI/imaging/gray.hpp>
#include <SushiAI/imaging/recent_partners.hpp>
#include <SushiAI/imaging/resize.hpp>

namespace SushiAI::Imaging
{
    namespace
    {
        /** @brief Holds what one load leaves beside the resized image. */
        struct LoadedLabels
        {
            /** @brief Holds the size of the decoded image before the resize. */
            ImageSize original_size{};

            /** @brief Holds the labels in the pixels of the resized image. */
            Labels labels;
        };

        /** @brief Holds the scratch images one load writes before the resized RGB image. */
        struct LoadScratch
        {
            /** @brief Receives the image at its stored resolution. */
            Image& decoded;

            /** @brief Receives a resized one-channel image before it is expanded. */
            Image& gray;
        };

        /**
         * @brief Loads one image, resizes its long side, makes it RGB and converts its labels.
         *
         * Follows the reference's get_image_and_label (data/base.py:445): the polygons are
         * resampled, then the labels are converted with the resized size.
         * @param loaded Receives the resized image with K_RGB_CHANNELS.
         * @param owner Names the calling function at the start of the message.
         */
        [[nodiscard]] LoadedLabels load_resized(const SampleSource& source, std::size_t index,
                                                std::int32_t image_size, LoadScratch scratch,
                                                Image& loaded, std::string_view owner)
        {
            SA_THROW_IF(index >= source.size(),
                        "{}: sample index {} is outside a source of {} images", owner, index,
                        source.size());

            FileLabels file_labels;
            source.load(index, scratch.decoded, file_labels);

            const std::int32_t channels = scratch.decoded.channels();
            SA_THROW_IF(!is_valid_size(scratch.decoded.size()),
                        "{}: the source returned no pixels for image {}", owner, index);
            SA_THROW_IF(channels != K_GRAY_CHANNELS && channels != K_RGB_CHANNELS,
                        "{}: the source returned {} channels for image {}; a detection image "
                        "has {} or {}",
                        owner, channels, index, K_GRAY_CHANNELS, K_RGB_CHANNELS);

            if (channels == K_GRAY_CHANNELS)
            {
                resize_long_side(scratch.decoded.view(), image_size, scratch.gray);
                expand_gray(scratch.gray.view(), loaded);
            }
            else
            {
                resize_long_side(scratch.decoded.view(), image_size, loaded);
            }

            LoadedLabels result;
            result.original_size = scratch.decoded.size();
            result.labels =
                to_pixel_labels(resample_polygons(std::move(file_labels)), loaded.size());
            return result;
        }

        /**
         * @brief Loads the three partners in draw order and composes the mosaic of one sample.
         * @param own_labels Pixel labels of patches[0], which already holds the sample's image.
         * @param canvas Receives the 2s by 2s mosaic.
         * @param owner Names the calling function at the start of the message.
         * @return The joined labels in canvas coordinates.
         */
        [[nodiscard]] Labels compose_mosaic(const SampleSource& source,
                                            const MosaicSettings& settings,
                                            const MosaicDraw& draw, std::int32_t image_size,
                                            Labels own_labels, LoadScratch scratch,
                                            std::array<Image, K_MOSAIC_PATCHES>& patches,
                                            Image& canvas, std::string_view owner)
        {
            std::array<Labels, K_MOSAIC_PATCHES> patch_labels;
            patch_labels[0] = std::move(own_labels);
            for (std::size_t i = 1; i < K_MOSAIC_PATCHES; ++i)
            {
                LoadedLabels partner = load_resized(source, draw.partners[i - 1], image_size,
                                                    scratch, patches[i], owner);
                patch_labels[i] = std::move(partner.labels);
            }

            std::array<ImageSize, K_MOSAIC_PATCHES> patch_sizes;
            std::array<ImageView, K_MOSAIC_PATCHES> patch_views;
            for (std::size_t i = 0; i < K_MOSAIC_PATCHES; ++i)
            {
                patch_sizes[i] = patches[i].size();
                patch_views[i] = patches[i].view();
            }

            const MosaicPlan plan = plan_mosaic(settings, draw, image_size, patch_sizes);
            apply_mosaic(patch_views, plan, canvas);
            return apply_mosaic(patch_labels, plan);
        }
    } // namespace

    bool is_mosaic_closed(std::int32_t epoch, std::int32_t epochs,
                          std::int32_t close_mosaic) noexcept
    {
        return epoch >= epochs - close_mosaic;
    }

    void validate_training(const TrainingSettings& settings)
    {
        validate_mosaic_size(settings.image_size);
        SA_THROW_IF(settings.epochs <= 0,
                    "validate_training: an epoch count of {} is not above zero", settings.epochs);
        SA_THROW_IF(settings.close_mosaic < 0,
                    "validate_training: a close_mosaic of {} is below zero",
                    settings.close_mosaic);
        SA_THROW_IF(settings.batch_size <= 0,
                    "validate_training: a batch_size of {} is not above zero",
                    settings.batch_size);

        validate_mosaic(settings.mosaic);
        validate_affine(settings.affine);
        validate_hsv(settings.hsv);
        validate_flip(settings.flip);
    }

    Random make_sample_random(std::uint64_t seed, std::int32_t epoch, std::size_t index) noexcept
    {
        return Random(seed)
            .derive(static_cast<std::uint64_t>(epoch))
            .derive(static_cast<std::uint64_t>(index));
    }

    TrainingPipeline::TrainingPipeline(const SampleSource& source,
                                       const TrainingSettings& settings)
        : source_(&source), settings_(settings)
    {
        validate_training(settings_);
        SA_THROW_IF(source.size() == 0, "TrainingPipeline: the source holds no image");

        partners_ = std::make_unique<RecentPartners>(source.size(), settings_.batch_size);
    }

    TrainingPipeline::TrainingPipeline(const SampleSource& source,
                                       const TrainingSettings& settings,
                                       std::unique_ptr<PartnerSource> partners)
        : source_(&source), settings_(settings), partners_(std::move(partners))
    {
        validate_training(settings_);
        SA_THROW_IF(source.size() == 0, "TrainingPipeline: the source holds no image");
        SA_THROW_IF(partners_ == nullptr, "TrainingPipeline: the partner source is null");
    }

    TrainingTrace TrainingPipeline::make_sample(std::size_t index, std::int32_t epoch,
                                                Sample& sample)
    {
        const std::string_view owner = "TrainingPipeline::make_sample";
        SA_THROW_IF(epoch < 0 || epoch >= settings_.epochs,
                    "{}: epoch {} is outside a run of {} epochs", owner, epoch, settings_.epochs);

        const std::int32_t side = settings_.image_size;
        const LoadScratch scratch{decoded_, gray_};
        LoadedLabels own = load_resized(*source_, index, side, scratch, patches_[0], owner);
        sample.original_size = own.original_size;
        sample.loaded_size = patches_[0].size();

        Random random = make_sample_random(settings_.seed, epoch, index);

        TrainingTrace trace;
        trace.is_mosaic_closed = is_mosaic_closed(epoch, settings_.epochs, settings_.close_mosaic);
        if (!trace.is_mosaic_closed)
            trace.mosaic = draw_mosaic(settings_.mosaic, side, index, *partners_, random);

        Labels labels = std::move(own.labels);
        ImageView warp_source = patches_[0].view();
        if (trace.mosaic.is_enabled)
        {
            labels = compose_mosaic(*source_, settings_.mosaic, trace.mosaic, side,
                                    std::move(labels), scratch, patches_, first_, owner);
            warp_source = first_.view();
        }

        trace.affine =
            plan_affine(settings_.affine, warp_source.size(), ImageSize{side, side}, random);
        apply_affine(warp_source, trace.affine, second_);
        labels = apply_affine(labels, trace.affine);

        // first_ is free here: the warp has already read the canvas it held.
        trace.hsv = plan_hsv(settings_.hsv, random);
        apply_hsv(second_.view(), trace.hsv, first_);

        trace.flip = plan_flip(settings_.flip, first_.size(), random);
        apply_flip(first_.view(), trace.flip, sample.image);
        sample.labels = apply_flip(labels, trace.flip);
        return trace;
    }

    void validate_validation(const ValidationSettings& settings)
    {
        SA_THROW_IF(!is_valid_side(settings.image_size),
                    "validate_validation: an image_size of {} is outside [1, {}]",
                    settings.image_size, K_MAX_IMAGE_SIDE);

        validate_letterbox(settings.letterbox);
    }

    ValidationPipeline::ValidationPipeline(const SampleSource& source,
                                           const ValidationSettings& settings)
        : source_(&source), settings_(settings)
    {
        validate_validation(settings_);
        SA_THROW_IF(source.size() == 0, "ValidationPipeline: the source holds no image");
    }

    LetterboxPlan ValidationPipeline::make_sample(std::size_t index, ImageSize target,
                                                  Sample& sample)
    {
        LoadedLabels own = load_resized(*source_, index, settings_.image_size,
                                        LoadScratch{decoded_, gray_}, loaded_,
                                        "ValidationPipeline::make_sample");
        sample.original_size = own.original_size;
        sample.loaded_size = loaded_.size();

        const LetterboxPlan plan = plan_letterbox(settings_.letterbox, loaded_.size(), target);
        apply_letterbox(loaded_.view(), plan, sample.image);
        sample.labels = apply_letterbox(own.labels, plan);
        return plan;
    }
} // namespace SushiAI::Imaging
