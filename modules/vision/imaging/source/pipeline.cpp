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

#include <SushiAI/vision/imaging/pipeline.hpp>

#include <string_view>
#include <utility>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

#include <SushiAI/vision/imaging/gray.hpp>
#include <SushiAI/vision/imaging/recent_partners.hpp>
#include <SushiAI/vision/imaging/resize.hpp>

namespace SushiAI::Vision::Imaging
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
         * @param loaded Receives the resized image with K_RGB_CHANNELS.
         * @param owner Names the calling function at the start of the message.
         * @return The labels and the decoded size; the failure of the load or the resize;
         *         InvalidFile for a source that returned no pixels or a bad channel count.
         * @pre @p index is below source.size().
         */
        [[nodiscard]] Result<LoadedLabels> load_resized(const SampleSource& source,
                                                        std::size_t index,
                                                        std::int32_t image_size,
                                                        LoadScratch scratch, Image& loaded,
                                                        std::string_view owner)
        {
            SA_FATAL_IF(index >= source.size(),
                        "{}: sample index {} is outside a source of {} images", owner, index,
                        source.size());

            FileLabels file_labels;
            SA_TRY(source.load(index, scratch.decoded, file_labels));

            const std::int32_t channels = scratch.decoded.channels();
            SA_FAIL_IF(!is_valid_size(scratch.decoded.size()), StatusCode::InvalidFile,
                       "{}: the source returned no pixels for image {}", owner, index);
            SA_FAIL_IF(channels != K_GRAY_CHANNELS && channels != K_RGB_CHANNELS,
                       StatusCode::InvalidFile,
                       "{}: the source returned {} channels for image {}; a detection image "
                       "has {} or {}",
                       owner, channels, index, K_GRAY_CHANNELS, K_RGB_CHANNELS);

            if (channels == K_GRAY_CHANNELS)
            {
                SA_TRY(resize_long_side(scratch.decoded.view(), image_size, scratch.gray));
                SA_TRY(expand_gray(scratch.gray.view(), loaded));
            }
            else
            {
                SA_TRY(resize_long_side(scratch.decoded.view(), image_size, loaded));
            }

            LoadedLabels result;
            result.original_size = scratch.decoded.size();
            result.labels =
                to_pixel_labels(resample_polygons(std::move(file_labels)), loaded.size());
            return Result<LoadedLabels>(std::move(result));
        }

        /**
         * @brief Loads the three partners in draw order and composes the mosaic of one sample.
         * @param own_labels Pixel labels of patches[0], which already holds the sample's image.
         * @param canvas Receives the 2s by 2s mosaic.
         * @param owner Names the calling function at the start of the message.
         * @return The joined labels in canvas coordinates, or the first failure of a partner's
         *         load, of the plan or of the canvas.
         */
        [[nodiscard]] Result<Labels> compose_mosaic(const SampleSource& source,
                                                    const MosaicSettings& settings,
                                                    const MosaicDraw& draw,
                                                    std::int32_t image_size, Labels own_labels,
                                                    LoadScratch scratch,
                                                    std::array<Image, K_MOSAIC_PATCHES>& patches,
                                                    Image& canvas, std::string_view owner)
        {
            std::array<Labels, K_MOSAIC_PATCHES> patch_labels;
            patch_labels[0] = std::move(own_labels);
            for (std::size_t i = 1; i < K_MOSAIC_PATCHES; ++i)
            {
                SA_TRY_ASSIGN(LoadedLabels partner,
                              load_resized(source, draw.partners[i - 1], image_size, scratch,
                                           patches[i], owner));
                patch_labels[i] = std::move(partner.labels);
            }

            std::array<ImageSize, K_MOSAIC_PATCHES> patch_sizes;
            std::array<ImageView, K_MOSAIC_PATCHES> patch_views;
            for (std::size_t i = 0; i < K_MOSAIC_PATCHES; ++i)
            {
                patch_sizes[i] = patches[i].size();
                patch_views[i] = patches[i].view();
            }

            SA_TRY_ASSIGN(const MosaicPlan plan,
                          plan_mosaic(settings, draw, image_size, patch_sizes));
            SA_TRY(apply_mosaic(patch_views, plan, canvas));
            return Result<Labels>(apply_mosaic(patch_labels, plan));
        }
    } // namespace

    bool is_mosaic_closed(std::int32_t epoch, std::int32_t epochs,
                          std::int32_t close_mosaic) noexcept
    {
        return epoch >= epochs - close_mosaic;
    }

    Status validate_training(const TrainingSettings& settings)
    {
        SA_TRY(validate_mosaic_size(settings.image_size));
        SA_FAIL_IF(settings.epochs <= 0, StatusCode::InvalidDescription,
                   "validate_training: an epoch count of {} is not above zero", settings.epochs);
        SA_FAIL_IF(settings.close_mosaic < 0, StatusCode::InvalidDescription,
                   "validate_training: a close_mosaic of {} is below zero",
                   settings.close_mosaic);
        SA_FAIL_IF(settings.batch_size <= 0, StatusCode::InvalidDescription,
                   "validate_training: a batch_size of {} is not above zero",
                   settings.batch_size);

        SA_TRY(validate_mosaic(settings.mosaic));
        SA_TRY(validate_affine(settings.affine));
        SA_TRY(validate_hsv(settings.hsv));
        return validate_flip(settings.flip);
    }

    Random make_sample_random(std::uint64_t seed, std::int32_t epoch, std::size_t index) noexcept
    {
        return Random(seed)
            .derive(static_cast<std::uint64_t>(epoch))
            .derive(static_cast<std::uint64_t>(index));
    }

    TrainingPipeline::TrainingPipeline(const SampleSource& source,
                                       const TrainingSettings& settings,
                                       std::unique_ptr<PartnerSource> partners) noexcept
        : source_(&source), settings_(settings), partners_(std::move(partners))
    {

    }

    Result<TrainingPipeline> TrainingPipeline::make(const SampleSource& source,
                                                    const TrainingSettings& settings)
    {
        SA_TRY(validate_training(settings));
        SA_FAIL_IF(source.size() == 0, StatusCode::InvalidDescription,
                   "TrainingPipeline: the source holds no image");

        SA_TRY_ASSIGN(std::unique_ptr<RecentPartners> recent,
                      RecentPartners::make(source.size(), settings.batch_size));
        std::unique_ptr<PartnerSource> partners = std::move(recent);

        return make(source, settings, std::move(partners));
    }

    Result<TrainingPipeline> TrainingPipeline::make(const SampleSource& source,
                                                    const TrainingSettings& settings,
                                                    std::unique_ptr<PartnerSource> partners)
    {
        SA_TRY(validate_training(settings));
        SA_FAIL_IF(source.size() == 0, StatusCode::InvalidDescription,
                   "TrainingPipeline: the source holds no image");
        SA_FATAL_IF(partners == nullptr, "TrainingPipeline: the partner source is null");

        return Result<TrainingPipeline>(
            TrainingPipeline(source, settings, std::move(partners)));
    }

    Result<TrainingTrace> TrainingPipeline::make_sample(std::size_t index, std::int32_t epoch,
                                                        Sample& sample)
    {
        const std::string_view owner = "TrainingPipeline::make_sample";
        SA_FATAL_IF(epoch < 0 || epoch >= settings_.epochs,
                    "{}: epoch {} is outside a run of {} epochs", owner, epoch, settings_.epochs);

        const std::int32_t side = settings_.image_size;
        const LoadScratch scratch{decoded_, gray_};
        SA_TRY_ASSIGN(LoadedLabels own,
                      load_resized(*source_, index, side, scratch, patches_[0], owner));
        sample.original_size = own.original_size;
        sample.loaded_size = patches_[0].size();

        Random random = make_sample_random(settings_.seed, epoch, index);

        TrainingTrace trace;
        trace.is_mosaic_closed = is_mosaic_closed(epoch, settings_.epochs, settings_.close_mosaic);
        if (!trace.is_mosaic_closed)
        {
            SA_TRY_ASSIGN(trace.mosaic,
                          draw_mosaic(settings_.mosaic, side, index, *partners_, random));
        }

        Labels labels = std::move(own.labels);
        ImageView warp_source = patches_[0].view();
        if (trace.mosaic.is_enabled)
        {
            SA_TRY_ASSIGN(labels,
                          compose_mosaic(*source_, settings_.mosaic, trace.mosaic, side,
                                         std::move(labels), scratch, patches_, first_, owner));
            warp_source = first_.view();
        }

        SA_TRY_ASSIGN(trace.affine, plan_affine(settings_.affine, warp_source.size(),
                                                ImageSize{side, side}, random));
        apply_affine(warp_source, trace.affine, second_);
        labels = apply_affine(labels, trace.affine);

        // first_ is free here: the warp has already read the canvas it held.
        SA_TRY_ASSIGN(trace.hsv, plan_hsv(settings_.hsv, random));
        apply_hsv(second_.view(), trace.hsv, first_);

        SA_TRY_ASSIGN(trace.flip, plan_flip(settings_.flip, first_.size(), random));
        apply_flip(first_.view(), trace.flip, sample.image);
        sample.labels = apply_flip(labels, trace.flip);
        return Result<TrainingTrace>(trace);
    }

    Status validate_validation(const ValidationSettings& settings)
    {
        SA_FAIL_IF(!is_valid_side(settings.image_size), StatusCode::InvalidDescription,
                   "validate_validation: an image_size of {} is outside [1, {}]",
                   settings.image_size, K_MAX_IMAGE_SIDE);

        validate_letterbox(settings.letterbox);
        return Status();
    }

    ValidationPipeline::ValidationPipeline(const SampleSource& source,
                                           const ValidationSettings& settings) noexcept
        : source_(&source), settings_(settings)
    {

    }

    Result<ValidationPipeline> ValidationPipeline::make(const SampleSource& source,
                                                        const ValidationSettings& settings)
    {
        SA_TRY(validate_validation(settings));
        SA_FAIL_IF(source.size() == 0, StatusCode::InvalidDescription,
                   "ValidationPipeline: the source holds no image");

        return Result<ValidationPipeline>(ValidationPipeline(source, settings));
    }

    Result<LetterboxPlan> ValidationPipeline::make_sample(std::size_t index, ImageSize target,
                                                          Sample& sample)
    {
        SA_TRY_ASSIGN(LoadedLabels own,
                      load_resized(*source_, index, settings_.image_size,
                                   LoadScratch{decoded_, gray_}, loaded_,
                                   "ValidationPipeline::make_sample"));
        sample.original_size = own.original_size;
        sample.loaded_size = loaded_.size();

        SA_TRY_ASSIGN(const LetterboxPlan plan,
                      plan_letterbox(settings_.letterbox, loaded_.size(), target));
        apply_letterbox(loaded_.view(), plan, sample.image);
        sample.labels = apply_letterbox(own.labels, plan);
        return Result<LetterboxPlan>(plan);
    }
} // namespace SushiAI::Vision::Imaging
