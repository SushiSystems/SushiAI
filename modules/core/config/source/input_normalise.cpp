/**************************************************************************/
/* input_normalise.cpp                                                    */
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
 * @file input_normalise.cpp
 * @brief Implements the reading and the writing of the colour normalisation of an input.
 * @author Mustafa Garip
 */

#include <SushiAI/config/input_normalise.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Config
    {
        namespace
        {
            /** @brief Names the key of the channel order. */
            constexpr std::string_view K_CHANNELS_KEY = "channels";

            /** @brief Names the key of the scale. */
            constexpr std::string_view K_SCALE_KEY = "scale";

            /** @brief Names the key of the per-channel mean. */
            constexpr std::string_view K_MEAN_KEY = "mean";

            /** @brief Names the key of the per-channel standard deviation. */
            constexpr std::string_view K_STD_KEY = "std";

            /** @brief Lists every key a normalisation may carry. */
            constexpr std::string_view K_NORMALISE_KEYS[] = {K_CHANNELS_KEY, K_SCALE_KEY,
                                                             K_MEAN_KEY, K_STD_KEY};

            /** @brief Returns the text that names @p key of the object in a diagnostic. */
            std::string named(std::string_view context, std::string_view key)
            {
                return std::string(context) + " '" + std::string(key) + "'";
            }

            /** @brief Reads "channels"; an absent key gives the decoded order. */
            Result<ChannelOrder> parse_channels(const Value& object, std::string_view context)
            {
                if (!object.has(K_CHANNELS_KEY))
                    return ChannelOrder::Rgb;

                const std::string where = named(context, K_CHANNELS_KEY);
                SA_TRY_ASSIGN(const Value& stated, object.at(K_CHANNELS_KEY, context));
                SA_TRY_ASSIGN(const std::string& order, stated.as_string(where));

                if (order == K_CHANNELS_RGB)
                    return ChannelOrder::Rgb;

                SA_FAIL_IF(order != K_CHANNELS_BGR, StatusCode::InvalidDescription,
                           "{} is '{}', which is no channel order; it is '{}' or '{}'", where,
                           order, K_CHANNELS_RGB, K_CHANNELS_BGR);
                return ChannelOrder::Bgr;
            }

            /** @brief Reads "scale"; an absent key gives the reciprocal of 255. */
            Result<double> parse_scale(const Value& object, std::string_view context)
            {
                if (!object.has(K_SCALE_KEY))
                    return K_DEFAULT_NORMALISE_SCALE;

                const std::string where = named(context, K_SCALE_KEY);
                SA_TRY_ASSIGN(const Value& stated, object.at(K_SCALE_KEY, context));
                SA_TRY_ASSIGN(const double scale, stated.as_real(where));

                SA_FAIL_IF(!std::isfinite(scale) || scale <= 0.0, StatusCode::InvalidDescription,
                           "{} is {}; it is a finite number above 0", where, scale);
                return scale;
            }

            /**
             * @brief Reads an array of three numbers; @p fallback when the key is absent.
             * @param positive True when every entry must be above 0, as a deviation must be.
             */
            Result<std::array<double, K_NORMALISE_CHANNELS>> parse_triple(
                const Value& object, std::string_view context, std::string_view key,
                const std::array<double, K_NORMALISE_CHANNELS>& fallback, bool positive)
            {
                if (!object.has(key))
                    return fallback;

                const std::string where = named(context, key);
                SA_TRY_ASSIGN(const Value& stated, object.at(key, context));
                SA_TRY_ASSIGN(const std::vector<Value>& entries, stated.as_array(where));
                SA_FAIL_IF(entries.size() != K_NORMALISE_CHANNELS, StatusCode::InvalidDescription,
                           "{} holds {} entries; it holds one number for each of {} channels",
                           where, entries.size(), K_NORMALISE_CHANNELS);

                std::array<double, K_NORMALISE_CHANNELS> numbers{};
                for (std::size_t i = 0; i < K_NORMALISE_CHANNELS; ++i)
                {
                    const std::string entry = where + " entry " + std::to_string(i);
                    SA_TRY_ASSIGN(numbers[i], entries[i].as_real(entry));
                    SA_FAIL_IF(!std::isfinite(numbers[i]), StatusCode::InvalidDescription,
                               "{} is {}; it is a finite number", entry, numbers[i]);
                    SA_FAIL_IF(positive && numbers[i] <= 0.0, StatusCode::InvalidDescription,
                               "{} is {}; it is above 0", entry, numbers[i]);
                }
                return numbers;
            }

            /** @brief Builds the array value of three numbers. */
            Value triple_value(const std::array<double, K_NORMALISE_CHANNELS>& numbers) noexcept
            {
                std::vector<Value> entries;
                for (const double number : numbers)
                    entries.push_back(Value::real(number));
                return Value::array(std::move(entries));
            }
        } // namespace

        Result<InputNormalise> parse_input_normalise(const Value& object,
                                                     std::string_view context) noexcept
        {
            SA_FAIL_IF(object.kind() != Kind::Object, StatusCode::InvalidDescription,
                       "{} must be an object, but it is {}", context, kind_name(object.kind()));
            SA_TRY(object.reject_unknown_keys(
                context, SushiRuntime::span<const std::string_view>(
                             K_NORMALISE_KEYS, std::size(K_NORMALISE_KEYS))));

            const InputNormalise identity;
            InputNormalise normalise;
            SA_TRY_ASSIGN(normalise.channels, parse_channels(object, context));
            SA_TRY_ASSIGN(normalise.scale, parse_scale(object, context));
            SA_TRY_ASSIGN(normalise.mean,
                          parse_triple(object, context, K_MEAN_KEY, identity.mean, false));
            SA_TRY_ASSIGN(normalise.deviation,
                          parse_triple(object, context, K_STD_KEY, identity.deviation, true));
            return normalise;
        }

        Value input_normalise_value(const InputNormalise& normalise) noexcept
        {
            std::vector<std::string> keys{std::string(K_CHANNELS_KEY), std::string(K_SCALE_KEY),
                                          std::string(K_MEAN_KEY), std::string(K_STD_KEY)};
            std::vector<Value> values;
            values.push_back(Value::string(normalise.channels == ChannelOrder::Rgb
                                               ? std::string(K_CHANNELS_RGB)
                                               : std::string(K_CHANNELS_BGR)));
            values.push_back(Value::real(normalise.scale));
            values.push_back(triple_value(normalise.mean));
            values.push_back(triple_value(normalise.deviation));
            return Value::object(std::move(keys), std::move(values));
        }
    } // namespace Config
} // namespace SushiAI
