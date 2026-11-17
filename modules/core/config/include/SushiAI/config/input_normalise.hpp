/**************************************************************************/
/* input_normalise.hpp                                                    */
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
 * @file input_normalise.hpp
 * @brief Declares the colour normalisation a model file states for its image input.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace Config
    {
        /** @brief Names the channel order of a decoded image, which is also the default. */
        inline constexpr std::string_view K_CHANNELS_RGB = "rgb";

        /** @brief Names the channel order that swaps the red and the blue channel. */
        inline constexpr std::string_view K_CHANNELS_BGR = "bgr";

        /** @brief Holds the channel count the mean and the deviation of a normalisation cover. */
        inline constexpr std::size_t K_NORMALISE_CHANNELS = 3;

        /** @brief Holds the default scale, the reciprocal of the largest byte level. */
        inline constexpr double K_DEFAULT_NORMALISE_SCALE = 1.0 / 255.0;

        /** @brief Names the channel order an image is written to the input in. */
        enum class ChannelOrder : std::uint8_t
        {
            Rgb, /**< The order a decoded image has: red first. */
            Bgr, /**< Blue first, as OpenCV and ByteTrack's YOLOX read an image. */
        };

        /**
         * @brief Holds how the byte levels of a fitted image become the values of the input.
         *
         * Output channel c of a pixel is (level * scale - mean[c]) / deviation[c], where level
         * is the byte of channel c once the channels are in the stated order. A default
         * value is the conversion every image input had before the model file could state
         * one: the decoded order and a division by 255.
         */
        struct InputNormalise
        {
            /** @brief Names the order of the channels of the input. */
            ChannelOrder channels = ChannelOrder::Rgb;

            /** @brief Multiplies each byte level before the mean is taken off; positive. */
            double scale = K_DEFAULT_NORMALISE_SCALE;

            /** @brief Holds the value taken off each channel after the scale. */
            std::array<double, K_NORMALISE_CHANNELS> mean{0.0, 0.0, 0.0};

            /** @brief Holds the positive divisor of each channel after the mean. */
            std::array<double, K_NORMALISE_CHANNELS> deviation{1.0, 1.0, 1.0};

            /** @brief Reports whether the order, the mean or the deviation is not the identity. */
            [[nodiscard]] bool colours_differ() const noexcept
            {
                const InputNormalise identity;
                return channels != identity.channels || mean != identity.mean ||
                       deviation != identity.deviation;
            }

            /** @brief Reports whether every field of two normalisations is equal. */
            [[nodiscard]] friend bool operator==(const InputNormalise& a,
                                                 const InputNormalise& b) noexcept
            {
                return a.channels == b.channels && a.scale == b.scale && a.mean == b.mean &&
                       a.deviation == b.deviation;
            }

            /** @brief Reports whether two normalisations differ in a field. */
            [[nodiscard]] friend bool operator!=(const InputNormalise& a,
                                                 const InputNormalise& b) noexcept
            {
                return !(a == b);
            }
        };

        /**
         * @brief Reads the "normalise" object of a model's input or of a data block.
         * @param object The object; every key is optional and an absent one keeps its default.
         * @param context What the object is, for a diagnostic.
         * @return The normalisation; InvalidDescription naming the key for an unknown key, a
         *         "channels" other than "rgb" and "bgr", a "scale" that is not finite and above
         *         0, a "mean" or "std" that is not three finite numbers, or a "std" entry of 0.
         */
        [[nodiscard]] Result<InputNormalise> parse_input_normalise(
            const Value& object, std::string_view context) noexcept;

        /**
         * @brief Builds the object that states @p normalise with every key.
         * @return The object, which parse_input_normalise reads back to an equal value.
         */
        [[nodiscard]] Value input_normalise_value(const InputNormalise& normalise) noexcept;

    } // namespace Config
} // namespace SushiAI
