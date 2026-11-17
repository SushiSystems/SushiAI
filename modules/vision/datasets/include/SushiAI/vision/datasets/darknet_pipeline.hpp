/**************************************************************************/
/* darknet_pipeline.hpp                                                   */
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
 * @file darknet_pipeline.hpp
 * @brief Declares the pipeline that augments each sample as Darknet's detector training does.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/codecs/decode_options.hpp>
#include <SushiAI/vision/datasets/byte_source.hpp>
#include <SushiAI/vision/datasets/sample_pipeline.hpp>
#include <SushiAI/vision/imaging/darknet_augment.hpp>
#include <SushiAI/vision/imaging/image.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Names the pipeline of Darknet's augmentation, in a data section. */
    inline constexpr std::string_view K_PIPELINE_DARKNET = "darknet";

    /** @brief Names the key of the crop jitter of the darknet pipeline. */
    inline constexpr std::string_view K_DARKNET_JITTER_KEY = "jitter";

    /** @brief Names the key of the hue shift of the darknet pipeline. */
    inline constexpr std::string_view K_DARKNET_HUE_KEY = "hue";

    /** @brief Names the key of the saturation bound of the darknet pipeline. */
    inline constexpr std::string_view K_DARKNET_SATURATION_KEY = "saturation";

    /** @brief Names the key of the exposure bound of the darknet pipeline. */
    inline constexpr std::string_view K_DARKNET_EXPOSURE_KEY = "exposure";

    /** @brief Holds what the caller chooses about the Darknet augmentation. */
    struct DarknetOptions
    {
        /** @brief Holds the input height and width, each in [1, Imaging::K_MAX_IMAGE_SIDE]. */
        Imaging::ImageSize size{};

        /** @brief Counts the channels of the input; 1 decodes to gray, any other count to colour. */
        std::int32_t channels = Imaging::K_RGB_CHANNELS;

        /** @brief Holds the seed every sample's draws derive from, with its epoch and index. */
        std::uint64_t seed = 0;

        /** @brief Holds the jitter, hue, saturation and exposure of the .cfg. */
        Imaging::DarknetAugmentSettings augment{};

        /**
         * @brief Serves the files of the samples; null reads the file system.
         * @pre The source, when given, outlives the pipeline.
         */
        const ByteSource* source = nullptr;
    };

    /**
     * @brief Checks augmentation settings as DarknetPipeline::make does.
     * @return Ok; InvalidDescription, naming DarknetPipeline, for settings
     *         Imaging::validate_darknet_augment refuses.
     */
    [[nodiscard]] Status validate_darknet_settings(
        const Imaging::DarknetAugmentSettings& settings) noexcept;

    /**
     * @brief Loads each sample, then crops, resizes and recolours it as Darknet does.
     *
     * A sample's draws come from Imaging::make_sample_random(seed, epoch, sample index), so its
     * result does not depend on the batch it is in or on the thread that makes it. The crop is
     * resized to the input size on both axes, with no padding; a model imported with
     * `letter_box=1` trains on this and runs on the fit its model file states.
     */
    class DarknetPipeline final : public SamplePipeline
    {
        public:
            /**
             * @brief Builds the pipeline of one input size and one set of settings.
             * @return The pipeline, never null; InvalidDescription for an input size outside the
             *         bounds of an image, for settings validate_darknet_augment refuses and for
             *         a channel count outside [1, Imaging::K_MAX_CHANNELS].
             */
            [[nodiscard]] static Result<std::unique_ptr<DarknetPipeline>> make(
                const DarknetOptions& options) noexcept;

            [[nodiscard]] std::string_view name() const noexcept override;

            /**
             * @brief Produces the augmented sample of each index.
             * @return What SamplePipeline::produce states; InvalidDescription for an epoch
             *         outside [0, the largest int32) and for a sample that has a mask, which
             *         Darknet's training does not augment.
             */
            [[nodiscard]] Status produce(const SampleList& list,
                                         SushiRuntime::span<const std::size_t> samples,
                                         std::int64_t epoch,
                                         SushiRuntime::span<FittedSample> fitted) const override;

        private:
            /** @brief Holds the options make() checked. */
            DarknetPipeline(const DarknetOptions& options, const ByteSource& source) noexcept;

            /// @brief Holds the input size, the seed and the settings of the augmentation.
            DarknetOptions options_;

            /// @brief Holds the decoding that yields the channel count.
            Codecs::DecodeOptions decoding_;

            /// @brief Points at the source the files are read through, never null.
            const ByteSource* source_;
    };
} // namespace SushiAI::Vision::Datasets
