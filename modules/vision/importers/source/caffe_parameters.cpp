/**************************************************************************/
/* caffe_parameters.cpp                                                   */
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
 * @file caffe_parameters.cpp
 * @brief Defines the reader of one parameter message of a Caffe layer.
 * @author Mustafa Garip
 */

#include "caffe_parameters.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/caffe_net.hpp>
#include <SushiAI/io/protobuf_text.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail::Caffe
        {
            namespace
            {
                /** @brief Holds the most entries a repeated window setting may state. */
                constexpr std::size_t K_MOST_WINDOW_ENTRIES = 2;
            } // namespace

            ParameterReader::ParameterReader(const Interchange::CaffeLayer& layer,
                                             std::string_view message) noexcept
                : layer_(&layer), message_(IO::find_field(layer.parameters, message))
            {

            }

            bool ParameterReader::has(std::string_view key) const noexcept
            {
                return message_ != nullptr && IO::find_field(*message_, key) != nullptr;
            }

            Status ParameterReader::expect_number(const IO::TextField& field) const
            {
                SA_FAIL_IF(field.kind != IO::TextKind::Number, StatusCode::InvalidDescription,
                           "layer '{}': line {}, column {}: '{}' needs a number, found '{}'",
                           layer_->name, field.line, field.column, field.name, field.text);
                return Status();
            }

            Result<std::int64_t> ParameterReader::parse_integer(const IO::TextField& field) const
            {
                SA_TRY(expect_number(field));
                std::int64_t value = 0;
                const char* const last = field.text.data() + field.text.size();
                const std::from_chars_result parsed =
                    std::from_chars(field.text.data(), last, value);
                SA_FAIL_IF(parsed.ec != std::errc() || parsed.ptr != last,
                           StatusCode::InvalidDescription,
                           "layer '{}': line {}, column {}: '{}' needs a whole number, found '{}'",
                           layer_->name, field.line, field.column, field.name, field.text);
                return value;
            }

            Result<std::int64_t> ParameterReader::read_integer(std::string_view key,
                                                               std::int64_t fallback) const
            {
                if (!has(key))
                    return fallback;
                return parse_integer(*IO::find_field(*message_, key));
            }

            Result<double> ParameterReader::parse_real(const IO::TextField& field) const
            {
                SA_TRY(expect_number(field));
                double value = 0.0;
                const char* const last = field.text.data() + field.text.size();
                const std::from_chars_result parsed =
                    std::from_chars(field.text.data(), last, value);
                SA_FAIL_IF(parsed.ec != std::errc() || parsed.ptr != last || !std::isfinite(value),
                           StatusCode::InvalidDescription,
                           "layer '{}': line {}, column {}: '{}' needs a finite number, found '{}'",
                           layer_->name, field.line, field.column, field.name, field.text);
                return value;
            }

            Result<double> ParameterReader::read_real(std::string_view key,
                                                      double fallback) const
            {
                if (!has(key))
                    return fallback;
                return parse_real(*IO::find_field(*message_, key));
            }

            Result<bool> ParameterReader::read_bool(std::string_view key, bool fallback) const
            {
                if (!has(key))
                    return fallback;
                const IO::TextField& field = *IO::find_field(*message_, key);
                const bool is_true = field.kind == IO::TextKind::Identifier && field.text == "true";
                const bool is_false =
                    field.kind == IO::TextKind::Identifier && field.text == "false";
                SA_FAIL_IF(!is_true && !is_false, StatusCode::InvalidDescription,
                           "layer '{}': line {}, column {}: '{}' needs true or false, found '{}'",
                           layer_->name, field.line, field.column, field.name, field.text);
                return is_true;
            }

            Result<std::string> ParameterReader::read_word(std::string_view key,
                                                           std::string_view fallback) const
            {
                if (!has(key))
                    return std::string(fallback);
                const IO::TextField& field = *IO::find_field(*message_, key);
                SA_FAIL_IF(field.kind != IO::TextKind::Identifier &&
                               field.kind != IO::TextKind::String,
                           StatusCode::InvalidDescription,
                           "layer '{}': line {}, column {}: '{}' needs a word, found '{}'",
                           layer_->name, field.line, field.column, field.name, field.text);
                return field.text;
            }

            Result<std::vector<std::int64_t>> ParameterReader::read_integers(
                std::string_view key) const
            {
                std::vector<std::int64_t> values;
                if (message_ == nullptr)
                    return values;
                for (const IO::TextField* field : IO::find_fields(*message_, key))
                {
                    SA_TRY_ASSIGN(const std::int64_t value, parse_integer(*field));
                    values.push_back(value);
                }
                return values;
            }

            Result<std::vector<double>> ParameterReader::read_reals(std::string_view key) const
            {
                std::vector<double> values;
                if (message_ == nullptr)
                    return values;
                for (const IO::TextField* field : IO::find_fields(*message_, key))
                {
                    SA_TRY_ASSIGN(const double value, parse_real(*field));
                    values.push_back(value);
                }
                return values;
            }

            Result<std::vector<std::int64_t>> ParameterReader::read_nested_integers(
                std::string_view inner, std::string_view key) const
            {
                std::vector<std::int64_t> values;
                if (message_ == nullptr)
                    return values;
                const IO::TextField* nested = IO::find_field(*message_, inner);
                if (nested == nullptr)
                    return values;
                for (const IO::TextField* field : IO::find_fields(*nested, key))
                {
                    SA_TRY_ASSIGN(const std::int64_t value, parse_integer(*field));
                    values.push_back(value);
                }
                return values;
            }

            Result<WindowPair> ParameterReader::read_window(std::string_view list_key,
                                                            std::string_view height_key,
                                                            std::string_view width_key,
                                                            WindowPair fallback) const
            {
                const bool pair = has(height_key) || has(width_key);
                SA_TRY_ASSIGN(const std::vector<std::int64_t> listed, read_integers(list_key));
                SA_FAIL_IF(pair && !listed.empty(), StatusCode::InvalidDescription,
                           "layer '{}' states '{}' and '{}' or '{}' together; Caffe refuses "
                           "both spellings at once",
                           layer_->name, list_key, height_key, width_key);
                if (pair)
                {
                    SA_FAIL_IF(!has(height_key) || !has(width_key), StatusCode::InvalidDescription,
                               "layer '{}' states one of '{}' and '{}'; it needs both",
                               layer_->name, height_key, width_key);
                    WindowPair stated;
                    SA_TRY_ASSIGN(stated.height, read_integer(height_key, 0));
                    SA_TRY_ASSIGN(stated.width, read_integer(width_key, 0));
                    return stated;
                }

                SA_FAIL_IF(listed.size() > K_MOST_WINDOW_ENTRIES, StatusCode::InvalidDescription,
                           "layer '{}' states {} entries of '{}'; a 2-D window takes one or two",
                           layer_->name, listed.size(), list_key);
                if (listed.empty())
                    return fallback;
                return WindowPair{listed.front(), listed.back()};
            }
        } // namespace Detail::Caffe
    } // namespace Vision::Importers
} // namespace SushiAI
