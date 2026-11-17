/**************************************************************************/
/* darknet_pipeline.cpp                                                   */
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
 * @file darknet_pipeline.cpp
 * @brief Implements the pipeline of Darknet's detector augmentation.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/darknet_pipeline.hpp>

#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/datasets/fit_pipeline.hpp>
#include <SushiAI/vision/datasets/load_sample_image.hpp>
#include <SushiAI/vision/imaging/darknet_augment.hpp>
#include <SushiAI/vision/imaging/label_set.hpp>
#include <SushiAI/vision/imaging/pipeline.hpp>
#include <SushiAI/vision/imaging/random.hpp>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_PIPELINE = "DarknetPipeline";

        /** @brief Holds the largest epoch the draws of a sample can key on. */
        constexpr std::int64_t K_EPOCH_LIMIT = std::numeric_limits<std::int32_t>::max();

        /**
         * @brief Augments one loaded sample into @p fitted.
         * @param loaded Holds the sample's pixels at their stored resolution.
         */
        [[nodiscard]] Status augment_one(const Sample& record, std::size_t index,
                                         std::int32_t epoch, const DarknetOptions& options,
                                         const Imaging::Image& loaded, FittedSample& fitted)
        {
            Imaging::Random random = Imaging::make_sample_random(options.seed, epoch, index);

            SA_TRY_ASSIGN(const Imaging::DarknetAugmentPlan plan,
                          Imaging::plan_darknet_augment(options.augment, loaded.size(), random));
            SA_TRY(Imaging::apply_darknet_augment(loaded.view(), plan, options.size,
                                                  fitted.image));

            fitted.objects = Imaging::apply_darknet_augment(
                Imaging::to_pixel_labels(record.objects, loaded.size()), plan, options.size);
            Imaging::clip_labels(fitted.objects, options.size);
            Imaging::remove_empty_boxes(fitted.objects);

            fitted.fit = Imaging::darknet_augment_transform(plan, options.size);
            fitted.file = record.file;
            fitted.class_index = record.class_index;
            return Status();
        }
    } // namespace

    Status validate_darknet_settings(const Imaging::DarknetAugmentSettings& settings) noexcept
    {
        return SA_ANNOTATE(Imaging::validate_darknet_augment(settings), "{}", K_PIPELINE);
    }

    Result<std::unique_ptr<DarknetPipeline>> DarknetPipeline::make(
        const DarknetOptions& options) noexcept
    {
        SA_FAIL_IF(!Imaging::is_valid_size(options.size), StatusCode::InvalidDescription,
                   "{}: an input of {} by {} pixels is outside the bounds of an image",
                   K_PIPELINE, options.size.height, options.size.width);
        SA_FAIL_IF(!Imaging::is_valid_channel_count(options.channels),
                   StatusCode::InvalidDescription,
                   "{}: an input of {} channels is outside [1, {}]", K_PIPELINE, options.channels,
                   Imaging::K_MAX_CHANNELS);
        SA_TRY(validate_darknet_settings(options.augment));

        return Detail::at_boundary(K_PIPELINE, [&]() -> Result<std::unique_ptr<DarknetPipeline>>
        {
            return std::unique_ptr<DarknetPipeline>(
                new DarknetPipeline(options, options.source != nullptr ? *options.source
                                                                       : Detail::file_system()));
        });
    }

    DarknetPipeline::DarknetPipeline(const DarknetOptions& options,
                                     const ByteSource& source) noexcept
        : options_(options), source_(&source)
    {
        decoding_.layout = FitPipeline::layout_of(options.channels);
    }

    std::string_view DarknetPipeline::name() const noexcept
    {
        return K_PIPELINE_DARKNET;
    }

    Status DarknetPipeline::produce(const SampleList& list,
                                    SushiRuntime::span<const std::size_t> samples,
                                    std::int64_t epoch,
                                    SushiRuntime::span<FittedSample> fitted) const
    {
        SA_FAIL_IF(samples.size() != fitted.size(), StatusCode::InvalidDescription,
                   "{}: {} samples were asked for into {} fitted samples", K_PIPELINE,
                   samples.size(), fitted.size());
        SA_FAIL_IF(epoch < 0 || epoch >= K_EPOCH_LIMIT, StatusCode::InvalidDescription,
                   "{}: epoch {} is outside [0, {})", K_PIPELINE, epoch, K_EPOCH_LIMIT);

        return Detail::at_boundary(K_PIPELINE, [&]() -> Status
        {
            Imaging::Image loaded;

            for (std::size_t row = 0; row < samples.size(); ++row)
            {
                SA_TRY(load_sample_image(list, samples[row], decoding_, loaded, *source_));

                // load_sample_image checked that the sample and its file are in the list.
                const Sample& record = list.samples[samples[row]];
                const std::filesystem::path& file = list.files[record.file];
                SA_FAIL_IF(record.mask.has_value(), StatusCode::InvalidDescription,
                           "{}: '{}' has a mask, and the pipeline '{}' augments no mask",
                           K_PIPELINE, Detail::display_of(file), K_PIPELINE_DARKNET);

                SA_TRY(SA_ANNOTATE(augment_one(record, samples[row],
                                               static_cast<std::int32_t>(epoch), options_, loaded,
                                               fitted[row]),
                                   "'{}'", Detail::display_of(file)));
            }

            return Status();
        });
    }
} // namespace SushiAI::Vision::Datasets
