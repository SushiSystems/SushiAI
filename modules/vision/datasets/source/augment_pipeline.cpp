/**************************************************************************/
/* augment_pipeline.cpp                                                   */
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
 * @file augment_pipeline.cpp
 * @brief Implements the pipeline that runs the training transforms of imaging on each sample.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/augment_pipeline.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include <SushiLog/text/text_format.hpp>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/codecs/decode_options.hpp>
#include <SushiAI/vision/datasets/fit_mask.hpp>
#include <SushiAI/vision/datasets/load_sample_image.hpp>
#include <SushiAI/vision/datasets/load_sample_mask.hpp>
#include <SushiAI/vision/imaging/dataset_partners.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/label_set.hpp>
#include <SushiAI/vision/imaging/mask_pipeline.hpp>
#include <SushiAI/vision/imaging/mosaic.hpp>
#include <SushiAI/vision/imaging/partner_source.hpp>
#include <SushiAI/vision/imaging/pipeline.hpp>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_PIPELINE = "AugmentPipeline";

        /** @brief Holds the largest epoch count Imaging::TrainingSettings can state. */
        constexpr std::int32_t K_EPOCH_LIMIT = std::numeric_limits<std::int32_t>::max();

        /** @brief Holds the largest sample count Imaging::DatasetPartners draws from. */
        constexpr std::size_t K_SAMPLE_LIMIT = std::numeric_limits<std::uint32_t>::max();

        /** @brief Supplies the images and the labels of a sample list to imaging. */
        class ListImages final : public Imaging::SampleSource
        {
            public:
                /** @brief Views a list, which outlives the source. */
                ListImages(const SampleList& list, const ByteSource& source) noexcept
                    : list_(&list), source_(&source)
                {

                }

                [[nodiscard]] std::size_t size() const noexcept override
                {
                    return list_->samples.size();
                }

                /** @brief Loads one image in colour, with the labels of its sample. */
                [[nodiscard]] Status load(std::size_t index, Imaging::Image& image,
                                          Imaging::FileLabels& labels) const override
                {
                    Codecs::DecodeOptions decoding;
                    decoding.layout = Codecs::ChannelLayout::Color;

                    SA_TRY(load_sample_image(*list_, index, decoding, image, *source_));
                    labels = list_->samples[index].objects;
                    return Status();
                }

            private:
                /// @brief Points at the list, which the caller keeps alive.
                const SampleList* list_;

                /// @brief Points at the source the images are read through.
                const ByteSource* source_;
        };

        /** @brief Supplies the masks of a sample list to imaging. */
        class ListMasks final : public Imaging::MaskSource
        {
            public:
                /** @brief Views a list, which outlives the source. */
                ListMasks(const SampleList& list, const ByteSource& source) noexcept
                    : list_(&list), source_(&source)
                {

                }

                [[nodiscard]] Status load(std::size_t index, Imaging::Image& mask) const override
                {
                    return load_sample_mask(*list_, index, mask, *source_);
                }

            private:
                /// @brief Points at the list, which the caller keeps alive.
                const SampleList* list_;

                /// @brief Points at the source the masks are read through.
                const ByteSource* source_;
        };

        /** @brief Returns what a message calls the mask file of a sample that has a mask. */
        [[nodiscard]] std::string mask_name(const SampleList& list, const Sample& record)
        {
            if (*record.mask < list.files.size())
                return Detail::display_of(list.files[*record.mask]);

            return SushiLog::format_text("mask file {}", *record.mask);
        }

        /** @brief Holds the two pipelines and the sample they reuse within one batch. */
        struct BatchPipelines
        {
            Imaging::TrainingPipeline images;    /**< Makes the image and the labels. */
            Imaging::TrainingMaskPipeline masks; /**< Replays the trace on the mask. */
            Imaging::Sample made;                /**< Holds what images last made. */
        };

        /** @brief Makes one sample, then clips its labels and makes its mask. */
        [[nodiscard]] Status produce_one(const SampleList& list, std::size_t sample,
                                         std::int32_t epoch, BatchPipelines& pipelines,
                                         FittedSample& fitted)
        {
            SA_FAIL_IF(sample >= list.samples.size(), StatusCode::InvalidDescription,
                       "{}: sample {} is outside a list of {} samples", K_PIPELINE, sample,
                       list.samples.size());

            SA_TRY_ASSIGN(const Imaging::TrainingTrace trace,
                          pipelines.images.make_sample(sample, epoch, pipelines.made));
            const Sample& record = list.samples[sample];

            std::swap(fitted.image, pipelines.made.image);
            fitted.objects = std::move(pipelines.made.labels);
            Imaging::clip_labels(fitted.objects, fitted.image.size());
            Imaging::remove_empty_boxes(fitted.objects);
            fitted.file = record.file;
            fitted.class_index = record.class_index;

            if (!record.mask.has_value())
                return Status();

            return SA_ANNOTATE(pipelines.masks.make_mask(sample, trace,
                                                         pipelines.made.loaded_size, fitted.mask),
                               "'{}'", mask_name(list, record));
        }
    } // namespace

    Result<std::unique_ptr<AugmentPipeline>> AugmentPipeline::make(const AugmentOptions& options) noexcept
    {
        SA_FAIL_IF(options.image_size < 1 ||
                       options.image_size > Imaging::K_MAX_MOSAIC_IMAGE_SIZE,
                   StatusCode::InvalidDescription,
                   "{}: a training size of {} is outside [1, {}]", K_PIPELINE,
                   options.image_size, Imaging::K_MAX_MOSAIC_IMAGE_SIZE);
        SA_FAIL_IF(options.mosaic_epochs < 0, StatusCode::InvalidDescription,
                   "{}: {} epochs of mosaic; the count is not below zero", K_PIPELINE,
                   options.mosaic_epochs);

        Imaging::TrainingSettings settings;
        settings.image_size = options.image_size;
        settings.seed = options.seed;
        settings.epochs = K_EPOCH_LIMIT;
        settings.close_mosaic = K_EPOCH_LIMIT - options.mosaic_epochs;
        settings.batch_size = 1;

        SA_TRY(Imaging::validate_training(settings));
        return Result<std::unique_ptr<AugmentPipeline>>(
            std::unique_ptr<AugmentPipeline>(new AugmentPipeline(settings, options.source != nullptr
                                                               ? *options.source
                                                               : Detail::file_system())));
    }

    AugmentPipeline::AugmentPipeline(const Imaging::TrainingSettings& settings,
                                     const ByteSource& source) noexcept
        : settings_(settings), source_(&source)
    {

    }

    std::string_view AugmentPipeline::name() const noexcept
    {
        return K_PIPELINE_DEFAULT;
    }

    Status AugmentPipeline::produce(const SampleList& list,
                                    SushiRuntime::span<const std::size_t> samples,
                                    std::int64_t epoch,
                                    SushiRuntime::span<FittedSample> fitted) const
    {
        SA_FAIL_IF(samples.size() != fitted.size(), StatusCode::InvalidDescription,
                   "{}: {} samples were asked for into {} fitted samples", K_PIPELINE,
                   samples.size(), fitted.size());
        SA_FAIL_IF(epoch < 0 || epoch >= settings_.epochs, StatusCode::InvalidDescription,
                   "{}: epoch {} is outside [0, {})", K_PIPELINE, epoch, settings_.epochs);
        SA_FAIL_IF(list.samples.empty() || list.samples.size() > K_SAMPLE_LIMIT,
                   StatusCode::InvalidDescription,
                   "{}: a list of {} samples; mosaic partners are drawn from 1 to {}",
                   K_PIPELINE, list.samples.size(), K_SAMPLE_LIMIT);

        const ListImages images(list, *source_);
        const ListMasks masks(list, *source_);

        return Detail::at_boundary(K_PIPELINE, [&]() -> Status
        {
            SA_TRY_ASSIGN(std::unique_ptr<Imaging::DatasetPartners> drawn,
                          Imaging::DatasetPartners::make(list.samples.size()));
            std::unique_ptr<Imaging::PartnerSource> partners = std::move(drawn);
            SA_TRY_ASSIGN(Imaging::TrainingPipeline training,
                          Imaging::TrainingPipeline::make(images, settings_,
                                                          std::move(partners)));

            BatchPipelines pipelines{std::move(training),
                                     Imaging::TrainingMaskPipeline(masks, settings_, K_MASK_VOID),
                                     Imaging::Sample{}};

            for (std::size_t row = 0; row < samples.size(); ++row)
                SA_TRY(produce_one(list, samples[row], static_cast<std::int32_t>(epoch),
                                   pipelines, fitted[row]));

            return Status();
        });
    }
} // namespace SushiAI::Vision::Datasets
