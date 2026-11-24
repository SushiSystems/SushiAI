/**************************************************************************/
/* darknet_export_progress.cpp                                            */
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
 * @file darknet_export_progress.cpp
 * @brief Defines the helpers the Darknet row writers share.
 * @author Mustafa Garip
 */

#include "darknet_export_progress.hpp"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include "export_walk.hpp"
#include "row_keys.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Holds the characters a printed double needs at most. */
                constexpr std::size_t K_DIGITS_CAPACITY = 64;

                /** @brief Holds the most significant digits a double needs to come back. */
                constexpr int K_MOST_DIGITS = 17;

                /** @brief Returns @p value printed with @p digits significant digits. */
                std::string print_digits(double value, int digits)
                {
                    char text[K_DIGITS_CAPACITY];
                    const std::to_chars_result end = std::to_chars(
                        text, text + sizeof(text), value, std::chars_format::general, digits);
                    return std::string(text, end.ptr);
                }

                /** @brief Returns the double @p text parses to, as the importer parses it. */
                double parse_real(const std::string& text)
                {
                    double value = 0.0;
                    std::from_chars(text.data(), text.data() + text.size(), value);
                    return value;
                }
            } // namespace

            RowKeys row_keys(const Config::LayerSpec& spec)
            {
                return RowKeys{&spec, std::nullopt, K_DARKNET_FORMAT};
            }

            Result<std::int64_t> read_square_key(const RowKeys& keys, std::string_view key,
                                                 std::int64_t fallback)
            {
                SA_TRY_ASSIGN(const std::vector<std::int64_t> pair,
                              read_pair_key(keys, key, fallback));
                SA_FAIL_IF(pair[0] != pair[1], StatusCode::Unsupported,
                           "it states a '{}' of {} by {}, and a Darknet section states one side",
                           key, pair[0], pair[1]);
                return pair[0];
            }

            Status check_source_count(const DarknetRowCall& call, std::size_t count)
            {
                return check_read_count(call.index, *call.spec, call.values.size(), count);
            }

            std::int64_t layer_of(const DarknetExportProgress& progress, std::int64_t row) noexcept
            {
                if (row < 0)
                    return progress.input.layer;
                return progress.values[static_cast<std::size_t>(row)]->layer;
            }

            Status follow_row(DarknetExportProgress& progress, std::int64_t row)
            {
                if (progress.current_row == row)
                    return Status();
                SA_FAIL_IF(row == Config::K_MODEL_INPUT, StatusCode::Unsupported,
                           "it reads the input after another section, and a Darknet [route] "
                           "names layers, not the input");
                add_section(progress, "route",
                            {DarknetOptionText{"layers", spell_integer(layer_of(progress, row))}});
                progress.current_row = row;
                return Status();
            }

            std::int64_t add_section(DarknetExportProgress& progress, std::string type,
                                     std::vector<DarknetOptionText> options)
            {
                progress.sections.push_back(
                    DarknetSectionText{std::move(type), std::move(options)});
                return static_cast<std::int64_t>(progress.sections.size()) - 1;
            }

            void define_row_value(DarknetExportProgress& progress, std::size_t row,
                                  std::vector<std::int64_t> shape)
            {
                const std::int64_t layer = static_cast<std::int64_t>(progress.sections.size()) - 1;
                progress.values[row] = DarknetRowValue{std::move(shape), layer};
                progress.current_row = static_cast<std::int64_t>(row);
            }

            std::string spell_integer(std::int64_t value)
            {
                return std::to_string(value);
            }

            std::string spell_real(double value)
            {
                for (int digits = 1; digits < K_MOST_DIGITS; ++digits)
                {
                    std::string text = print_digits(value, digits);
                    if (parse_real(text) == value)
                        return text;
                }
                return print_digits(value, K_MOST_DIGITS);
            }

            std::string spell_float(float value)
            {
                for (int digits = 1; digits < K_MOST_DIGITS; ++digits)
                {
                    std::string text = print_digits(static_cast<double>(value), digits);
                    if (static_cast<float>(parse_real(text)) == value)
                        return text;
                }
                return print_digits(static_cast<double>(value), K_MOST_DIGITS);
            }

            std::string spell_anchor_side(float fraction, std::int64_t side)
            {
                const double divisor = static_cast<double>(side);
                const double pixels = static_cast<double>(fraction) * divisor;
                for (int digits = 1; digits < K_MOST_DIGITS; ++digits)
                {
                    std::string text = print_digits(pixels, digits);
                    if (static_cast<float>(parse_real(text) / divisor) == fraction)
                        return text;
                }
                return print_digits(pixels, K_MOST_DIGITS);
            }

            std::string spell_integer_list(const std::vector<std::int64_t>& values)
            {
                std::string text;
                for (const std::int64_t value : values)
                    text.append(text.empty() ? "" : ",").append(std::to_string(value));
                return text;
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
