/**************************************************************************/
/* fit_pipeline.hpp                                                       */
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
 * @file fit_pipeline.hpp
 * @brief Declares the pipeline that fits each sample to the input size and draws nothing.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/codecs/decode_options.hpp>
#include <SushiAI/vision/datasets/fit_sample.hpp>
#include <SushiAI/vision/datasets/sample_pipeline.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Names the pipeline that only fits, in a data section. */
    inline constexpr std::string_view K_PIPELINE_NONE = "none";

    /**
     * @brief Loads each sample, fits its image with fit_sample and its mask with fit_mask.
     *
     * The result of a sample is the same in every epoch.
     */
    class FitPipeline final : public SamplePipeline
    {
        public:
            /**
             * @brief Builds the pipeline of one fit and one channel count.
             * @param channels 1 decodes to gray, any other count to colour.
             * @return The pipeline, never null; InvalidDescription for an input size outside
             *         the bounds of an image.
             */
            [[nodiscard]] static Result<std::unique_ptr<FitPipeline>> make(
                const SampleFitOptions& fit, std::int32_t channels) noexcept;

            /** @brief Returns the layout an image of a channel count is decoded to. */
            [[nodiscard]] static Codecs::ChannelLayout layout_of(std::int32_t channels) noexcept;

            [[nodiscard]] std::string_view name() const noexcept override;

            /**
             * @brief Loads and fits each sample; the epoch is not read.
             * @return What SamplePipeline::produce states; Incompatible, opening with the
             *         mask file, for a mask fit_mask refuses.
             */
            [[nodiscard]] Status produce(const SampleList& list,
                                         SushiRuntime::span<const std::size_t> samples,
                                         std::int64_t epoch,
                                         SushiRuntime::span<FittedSample> fitted) const override;

        private:
            /** @brief Holds the fit and the decoding of the channel count. */
            FitPipeline(const SampleFitOptions& fit, std::int32_t channels) noexcept;

            /// @brief Holds the input size and how an image is fitted to it.
            SampleFitOptions fit_;

            /// @brief Holds the decoding that yields the channel count.
            Codecs::DecodeOptions decoding_;
    };
} // namespace SushiAI::Vision::Datasets
