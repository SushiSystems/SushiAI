/**************************************************************************/
/* batch_inputs.cpp                                                       */
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
 * @file batch_inputs.cpp
 * @brief Implements the copy of the fitted images of one batch into its input tensor.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/batch_inputs.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/vision/imaging/image.hpp>

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_WRITER = "write_batch_inputs";

        /** @brief Holds the rank of an NHWC tensor. */
        constexpr std::int32_t K_INPUT_RANK = 4;

        /** @brief Counts the levels a byte can hold. */
        constexpr std::size_t K_LEVELS = 256;

        /** @brief Holds the channel count the colours of a normalisation are stated for. */
        constexpr std::int64_t K_COLOUR_CHANNELS = 3;

        /** @brief Holds the value of every byte level on one channel. */
        using LevelTable = std::array<float, K_LEVELS>;

        /** @brief Holds one table for each channel the mean and the deviation are stated for. */
        using ChannelTables = std::array<LevelTable, Config::K_NORMALISE_CHANNELS>;

        /** @brief Holds the table of each channel and the stored channel each output reads. */
        struct ChannelPlan
        {
            ChannelTables tables;                                       /**< Value of each level. */
            std::array<std::size_t, Imaging::K_MAX_CHANNELS> source{};  /**< Stored channel read. */
        };

        /** @brief Builds the tables and the channel each output channel reads. */
        ChannelPlan plan_channels(const Config::InputNormalise& normalise, std::int64_t channels)
        {
            ChannelPlan plan;
            for (std::size_t c = 0; c < Config::K_NORMALISE_CHANNELS; ++c)
            {
                for (std::size_t level = 0; level < K_LEVELS; ++level)
                {
                    const double scaled = static_cast<double>(level) * normalise.scale;
                    plan.tables[c][level] =
                        static_cast<float>((scaled - normalise.mean[c]) / normalise.deviation[c]);
                }
            }

            const bool swapped = channels == K_COLOUR_CHANNELS &&
                                 normalise.channels == Config::ChannelOrder::Bgr;
            for (std::size_t c = 0; c < static_cast<std::size_t>(channels); ++c)
                plan.source[c] = swapped ? Config::K_NORMALISE_CHANNELS - 1 - c : c;

            return plan;
        }
    } // namespace

    Status write_batch_inputs(SushiRuntime::span<const FittedSample> samples,
                              SushiBLAS::Tensor& inputs,
                              const Config::InputNormalise& normalise) noexcept
    {
        SA_FAIL_IF(inputs.dtype != DType::Float32 || inputs.rank != K_INPUT_RANK,
                   StatusCode::Incompatible,
                   "{}: the input tensor has rank {}; it must be float32 of rank {}", K_WRITER,
                   inputs.rank, K_INPUT_RANK);
        SA_FAIL_IF(inputs.shape[0] != static_cast<std::int64_t>(samples.size()),
                   StatusCode::Incompatible, "{}: the input tensor holds {} images; got {}",
                   K_WRITER, inputs.shape[0], samples.size());

        const std::int64_t height = inputs.shape[1];
        const std::int64_t width = inputs.shape[2];
        const std::int64_t channels = inputs.shape[3];
        SA_FAIL_IF(!Imaging::is_valid_channel_count(channels), StatusCode::Incompatible,
                   "{}: the input has {} channels; an image has 1 to {}", K_WRITER, channels,
                   Imaging::K_MAX_CHANNELS);
        SA_FAIL_IF(channels != K_COLOUR_CHANNELS && normalise.colours_differ(),
                   StatusCode::Incompatible,
                   "{}: the input has {} channels; a channel order, a mean or a deviation is "
                   "stated for {}",
                   K_WRITER, channels, K_COLOUR_CHANNELS);

        const ChannelPlan plan = plan_channels(normalise, channels);
        const std::size_t pixels = static_cast<std::size_t>(height * width);
        const std::size_t stride = static_cast<std::size_t>(channels);
        float* out = inputs.data_as<float>();

        for (std::size_t i = 0; i < samples.size(); ++i)
        {
            const Imaging::Image& image = samples[i].image;

            SA_FAIL_IF(image.height() != height || image.width() != width ||
                           image.channels() != channels,
                       StatusCode::Incompatible,
                       "{}: image {} of the batch is {} by {} of {} channels; the input is {} "
                       "by {} of {}",
                       K_WRITER, i, image.height(), image.width(), image.channels(), height,
                       width, channels);

            const std::uint8_t* bytes = image.data();
            for (std::size_t p = 0; p < pixels; ++p)
            {
                for (std::size_t c = 0; c < stride; ++c)
                {
                    const std::size_t table = std::min(c, Config::K_NORMALISE_CHANNELS - 1);
                    out[c] = plan.tables[table][bytes[plan.source[c]]];
                }

                bytes += stride;
                out += stride;
            }
        }

        return Status();
    }
} // namespace SushiAI::Vision::Datasets
