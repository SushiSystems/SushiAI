/**************************************************************************/
/* pipeline_table.cpp                                                     */
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
 * @file pipeline_table.cpp
 * @brief Implements the table that maps the name of a pipeline to what builds it.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/pipeline_table.hpp>

#include <cstddef>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>

#include <SushiAI/config/section_reader.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/datasets/augment_pipeline.hpp>
#include <SushiAI/vision/datasets/darknet_pipeline.hpp>
#include <SushiAI/vision/datasets/fit_pipeline.hpp>
#include <SushiAI/vision/imaging/darknet_augment.hpp>
#include <SushiAI/vision/imaging/image.hpp>

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Builds the pipeline that fits and draws nothing. */
        [[nodiscard]] Result<std::unique_ptr<const SamplePipeline>> make_fit_pipeline(
            const PipelineRequest& request)
        {
            SA_TRY_ASSIGN(std::unique_ptr<FitPipeline> pipeline,
                          FitPipeline::make(request.fit, request.channels, request.source,
                                            request.decode_masks));

            return std::unique_ptr<const SamplePipeline>(std::move(pipeline));
        }

        /**
         * @brief Builds the pipeline of the training transforms.
         * @return The pipeline; InvalidDescription for an input of another channel count
         *         than three, which the transforms do not produce.
         */
        [[nodiscard]] Result<std::unique_ptr<const SamplePipeline>> make_augment_pipeline(
            const PipelineRequest& request)
        {
            SA_FAIL_IF(request.channels != Imaging::K_RGB_CHANNELS,
                       StatusCode::InvalidDescription,
                       "the pipeline '{}' makes samples of {} channels; the input has {}",
                       K_PIPELINE_DEFAULT, Imaging::K_RGB_CHANNELS, request.channels);

            AugmentOptions options;
            options.image_size = request.fit.size;
            options.seed = request.seed;
            options.source = request.source;
            options.decode_masks = request.decode_masks;

            SA_TRY_ASSIGN(std::unique_ptr<AugmentPipeline> pipeline,
                          AugmentPipeline::make(options));

            return std::unique_ptr<const SamplePipeline>(std::move(pipeline));
        }

        /** @brief Lists the keys the darknet row reads. */
        constexpr std::string_view K_DARKNET_KEYS[] = {
            K_DARKNET_JITTER_KEY, K_DARKNET_HUE_KEY, K_DARKNET_SATURATION_KEY,
            K_DARKNET_EXPOSURE_KEY,
        };

        /** @brief Reads one float32 key of the section, or returns @p fallback when it is absent. */
        [[nodiscard]] Result<float> read_darknet_key(const Config::Value& section,
                                                     std::string_view context,
                                                     std::string_view key, float fallback)
        {
            const Config::Value stated = Config::float32_value(fallback);
            return Config::read_float32(section.at_or(key, stated),
                                        Config::member_of(context, key));
        }

        /**
         * @brief Reads the darknet keys of the section; an absent key keeps its identity value.
         * @return The settings; InvalidDescription for a key that is no number float32 holds.
         */
        [[nodiscard]] Result<Imaging::DarknetAugmentSettings> read_darknet_settings(
            const Config::Value& section, std::string_view context)
        {
            const Imaging::DarknetAugmentSettings identity;
            Imaging::DarknetAugmentSettings settings;
            SA_TRY_ASSIGN(settings.jitter, read_darknet_key(section, context, K_DARKNET_JITTER_KEY,
                                                            identity.jitter));
            SA_TRY_ASSIGN(settings.hue,
                          read_darknet_key(section, context, K_DARKNET_HUE_KEY, identity.hue));
            SA_TRY_ASSIGN(settings.saturation,
                          read_darknet_key(section, context, K_DARKNET_SATURATION_KEY,
                                           identity.saturation));
            SA_TRY_ASSIGN(settings.exposure, read_darknet_key(section, context,
                                                              K_DARKNET_EXPOSURE_KEY,
                                                              identity.exposure));
            return settings;
        }

        /**
         * @brief Checks the darknet keys of the section as the darknet factory would.
         * @return Ok; InvalidDescription for a key that is no number float32 holds and for
         *         settings validate_darknet_settings refuses.
         */
        [[nodiscard]] Status check_darknet_keys(const Config::Value& section,
                                                std::string_view context)
        {
            SA_TRY_ASSIGN(const Imaging::DarknetAugmentSettings settings,
                          read_darknet_settings(section, context));
            return validate_darknet_settings(settings);
        }

        /**
         * @brief Builds the pipeline of Darknet's crop and colour distortion.
         * @return The pipeline; InvalidDescription for a key that is no number float32 holds,
         *         for settings Imaging::validate_darknet_augment refuses and for a channel count
         *         outside [1, Imaging::K_MAX_CHANNELS].
         */
        [[nodiscard]] Result<std::unique_ptr<const SamplePipeline>> make_darknet_pipeline(
            const PipelineRequest& request)
        {
            DarknetOptions options;
            options.size = request.fit.size;
            options.channels = request.channels;
            options.seed = request.seed;
            options.source = request.source;
            SA_TRY_ASSIGN(options.augment, read_darknet_settings(request.section, request.context));

            SA_TRY_ASSIGN(std::unique_ptr<DarknetPipeline> pipeline,
                          DarknetPipeline::make(options));

            return std::unique_ptr<const SamplePipeline>(std::move(pipeline));
        }
    } // namespace

    SushiRuntime::span<const PipelineEntry> pipeline_table() noexcept
    {
        // The rows hold spans, so they are built on the first call; the first row is the default.
        static const PipelineEntry K_PIPELINES[] = {
            PipelineEntry{K_PIPELINE_NONE, &make_fit_pipeline, {}},
            PipelineEntry{K_PIPELINE_DEFAULT, &make_augment_pipeline, {}},
            PipelineEntry{K_PIPELINE_DARKNET, &make_darknet_pipeline,
                          SushiRuntime::span<const std::string_view>(K_DARKNET_KEYS,
                                                                     std::size(K_DARKNET_KEYS)),
                          &check_darknet_keys},
        };

        return SushiRuntime::span<const PipelineEntry>(K_PIPELINES, std::size(K_PIPELINES));
    }

    const PipelineEntry* find_pipeline(std::string_view name) noexcept
    {
        for (const PipelineEntry& row : pipeline_table())
        {
            if (row.name == name)
                return &row;
        }

        return nullptr;
    }
} // namespace SushiAI::Vision::Datasets
