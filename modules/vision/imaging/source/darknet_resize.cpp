/**************************************************************************/
/* darknet_resize.cpp                                                     */
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
 * @file darknet_resize.cpp
 * @brief Implements the bilinear resize of Darknet's resize_image.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/darknet_resize.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI::Vision::Imaging
{
    namespace
    {
        /** @brief Holds the largest 8-bit level. */
        constexpr double K_FULL_LEVEL = 255.0;

        /** @brief Holds one half, which rounds a level to the nearest integer. */
        constexpr double K_HALF = 0.5;

        /** @brief Holds the two source positions one destination index blends. */
        struct Tap
        {
            /** @brief Holds the lower source index. */
            std::int32_t lower = 0;

            /** @brief Holds the upper source index; the lower one at the last source index. */
            std::int32_t upper = 0;

            /** @brief Holds the weight of the upper index, in [0, 1). */
            double weight = 0.0;
        };

        /** @brief Returns the taps of every destination index of one axis. */
        [[nodiscard]] std::vector<Tap> make_taps(std::int32_t source, std::int32_t destination)
        {
            const double step = destination > 1 ? static_cast<double>(source - 1) /
                                                      static_cast<double>(destination - 1)
                                                : 0.0;
            std::vector<Tap> taps(static_cast<std::size_t>(destination));

            for (std::int32_t at = 0; at < destination; ++at)
            {
                Tap& tap = taps[static_cast<std::size_t>(at)];

                // The last index is the last source index exactly, whatever the rounding of step.
                if (at == destination - 1 && destination > 1)
                {
                    tap.lower = source - 1;
                    tap.upper = source - 1;
                    continue;
                }

                const double position = static_cast<double>(at) * step;
                tap.lower = static_cast<std::int32_t>(position);
                tap.weight = position - static_cast<double>(tap.lower);
                tap.upper = std::min(tap.lower + 1, source - 1);
            }

            return taps;
        }

        /** @brief Rounds a blended level half up and clamps it to [0, 255]. */
        [[nodiscard]] std::uint8_t to_level(double level) noexcept
        {
            return static_cast<std::uint8_t>(std::clamp(std::floor(level + K_HALF), 0.0,
                                                        K_FULL_LEVEL));
        }
    } // namespace

    Status resize_darknet(ImageView source, ImageSize size, Image& destination) noexcept
    {
        check_source(source, destination, "resize_darknet");
        SA_FAIL_IF(!is_valid_size(size), StatusCode::InvalidDescription,
                   "resize_darknet: a destination size of {}x{} is outside [1, {}]", size.height,
                   size.width, K_MAX_IMAGE_SIDE);

        const std::vector<Tap> columns = make_taps(source.width(), size.width);
        const std::vector<Tap> rows = make_taps(source.height(), size.height);
        const auto channels = static_cast<std::size_t>(source.channels());
        const std::size_t part_row = static_cast<std::size_t>(size.width) * channels;

        // The columns are blended first, in every source row, as Darknet does.
        std::vector<double> part(static_cast<std::size_t>(source.height()) * part_row);
        for (std::int32_t y = 0; y < source.height(); ++y)
        {
            const std::uint8_t* const in = source.row(y);
            double* out = part.data() + static_cast<std::size_t>(y) * part_row;

            for (const Tap& tap : columns)
            {
                for (std::size_t c = 0; c < channels; ++c)
                {
                    const double lower = in[static_cast<std::size_t>(tap.lower) * channels + c];
                    const double upper = in[static_cast<std::size_t>(tap.upper) * channels + c];
                    *out++ = (1.0 - tap.weight) * lower + tap.weight * upper;
                }
            }
        }

        destination.reset(size, source.channels());

        for (std::int32_t y = 0; y < size.height; ++y)
        {
            const Tap& tap = rows[static_cast<std::size_t>(y)];
            const double* const lower = part.data() + static_cast<std::size_t>(tap.lower) * part_row;
            const double* const upper = part.data() + static_cast<std::size_t>(tap.upper) * part_row;
            std::uint8_t* const out = destination.row(y);

            for (std::size_t i = 0; i < part_row; ++i)
                out[i] = to_level((1.0 - tap.weight) * lower[i] + tap.weight * upper[i]);
        }

        return Status();
    }
} // namespace SushiAI::Vision::Imaging
