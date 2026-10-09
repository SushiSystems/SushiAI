/**************************************************************************/
/* darknet_layer_table.cpp                                                */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file darknet_layer_table.cpp
 * @brief Implements the table of section translators and the helpers every translator shares.
 * @author Mustafa Garip
 */

#include "darknet_layer_table.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Lists every section type this importer translates. */
                constexpr LayerTableRow K_LAYER_TABLE[] = {
                    {"convolutional", &translate_convolutional},
                    {"maxpool", &translate_maxpool},
                    {"upsample", &translate_upsample},
                    {"route", &translate_route},
                    {"shortcut", &translate_shortcut},
                    {"yolo", &translate_yolo},
                };

                /** @brief Holds the largest magnitude a whole-number option may have. */
                constexpr std::int64_t K_LARGEST_INTEGER = std::numeric_limits<std::int32_t>::max();

                /**
                 * @brief Parses @p text as a whole number of 32 bits.
                 * @return InvalidDescription naming @p line and @p key when it is not one.
                 */
                [[nodiscard]] Result<std::int64_t> parse_integer(std::string_view text,
                                                                 std::string_view key,
                                                                 std::int64_t line)
                {
                    std::int64_t value = 0;
                    const char* const end = text.data() + text.size();
                    const std::from_chars_result parsed = std::from_chars(text.data(), end, value);
                    SA_FAIL_IF(text.empty() || parsed.ec != std::errc() || parsed.ptr != end ||
                                   value > K_LARGEST_INTEGER || value < -K_LARGEST_INTEGER,
                               StatusCode::InvalidDescription,
                               "line {}: '{}' holds '{}', which is not a whole number of 32 bits",
                               line, key, text);

                    return value;
                }

                /**
                 * @brief Parses @p text as a finite number.
                 * @return InvalidDescription naming @p line and @p key when it is not one.
                 */
                [[nodiscard]] Result<double> parse_real(std::string_view text, std::string_view key,
                                                        std::int64_t line)
                {
                    double value = 0.0;
                    const char* const end = text.data() + text.size();
                    const std::from_chars_result parsed = std::from_chars(text.data(), end, value);
                    SA_FAIL_IF(text.empty() || parsed.ec != std::errc() || parsed.ptr != end ||
                                   !std::isfinite(value),
                               StatusCode::InvalidDescription,
                               "line {}: '{}' holds '{}', which is not a finite number", line, key,
                               text);

                    return value;
                }

                /**
                 * @brief Returns the entries of the list option @p key, split at each comma.
                 * @return InvalidDescription when the key is absent or its value is empty.
                 */
                [[nodiscard]] Result<std::vector<std::string_view>> split_list(
                    const DarknetSection& section, std::string_view key)
                {
                    const DarknetOption* option = find_option(section, key);
                    SA_FAIL_IF(option == nullptr, StatusCode::InvalidDescription,
                               "line {}: the [{}] section has no '{}', which it needs",
                               section.line, section.type, key);
                    SA_FAIL_IF(option->value.empty(), StatusCode::InvalidDescription,
                               "line {}: '{}' is empty; it needs a comma-separated list",
                               option->line, key);

                    std::vector<std::string_view> entries;
                    std::string_view rest = option->value;
                    while (true)
                    {
                        const std::size_t comma = rest.find(',');
                        entries.push_back(rest.substr(0, comma));
                        if (comma == std::string_view::npos)
                            break;

                        rest.remove_prefix(comma + 1);
                    }
                    return entries;
                }

            } // namespace

            const LayerTableRow* find_layer_translator(std::string_view type) noexcept
            {
                for (const LayerTableRow& row : K_LAYER_TABLE)
                {
                    if (row.type == type)
                        return &row;
                }
                return nullptr;
            }

            Status reject_unknown_options(const DarknetSection& section,
                                          SushiRuntime::span<const std::string_view> known)
            {
                for (const DarknetOption& option : section.options)
                {
                    const bool is_known =
                        std::find(known.begin(), known.end(), std::string_view(option.key)) !=
                        known.end();
                    SA_FAIL_IF(!is_known, StatusCode::Unsupported,
                               "line {}: the [{}] section has the key '{}', which this importer "
                               "does not translate",
                               option.line, section.type, option.key);
                }
                return Status();
            }

            const DarknetOption* find_option(const DarknetSection& section,
                                             std::string_view key) noexcept
            {
                for (const DarknetOption& option : section.options)
                {
                    if (option.key == key)
                        return &option;
                }
                return nullptr;
            }

            Result<std::int64_t> read_integer(const DarknetSection& section, std::string_view key,
                                              std::int64_t fallback)
            {
                const DarknetOption* option = find_option(section, key);
                if (option == nullptr)
                    return fallback;

                return parse_integer(option->value, key, option->line);
            }

            Result<double> read_real(const DarknetSection& section, std::string_view key,
                                     double fallback)
            {
                const DarknetOption* option = find_option(section, key);
                if (option == nullptr)
                    return fallback;

                return parse_real(option->value, key, option->line);
            }

            Result<std::vector<std::int64_t>> read_integer_list(const DarknetSection& section,
                                                                std::string_view key)
            {
                SA_TRY_ASSIGN(const std::vector<std::string_view> entries,
                              split_list(section, key));
                const std::int64_t line = find_option(section, key)->line;

                std::vector<std::int64_t> values;
                values.reserve(entries.size());
                for (std::string_view entry : entries)
                {
                    SA_TRY_ASSIGN(const std::int64_t value, parse_integer(entry, key, line));
                    values.push_back(value);
                }
                return values;
            }

            Result<std::vector<double>> read_real_list(const DarknetSection& section,
                                                       std::string_view key)
            {
                SA_TRY_ASSIGN(const std::vector<std::string_view> entries,
                              split_list(section, key));
                const std::int64_t line = find_option(section, key)->line;

                std::vector<double> values;
                values.reserve(entries.size());
                for (std::string_view entry : entries)
                {
                    SA_TRY_ASSIGN(const double value, parse_real(entry, key, line));
                    values.push_back(value);
                }
                return values;
            }

            std::int64_t current_layer(const TranslationProgress& progress) noexcept
            {
                return static_cast<std::int64_t>(progress.layers.size());
            }

            const LayerValue& previous_value(const TranslationProgress& progress) noexcept
            {
                return progress.layers.empty() ? progress.input : progress.layers.back();
            }

            Result<LayerValue> resolve_layer(const TranslationProgress& progress,
                                             std::int64_t reference, std::int64_t line)
            {
                const std::int64_t layer = current_layer(progress);
                const std::int64_t index = reference < 0 ? layer + reference : reference;
                SA_FAIL_IF(index < 0 || index >= layer, StatusCode::InvalidDescription,
                           "line {}: layer {} names layer {} as {}, and only layers 0 to {} are "
                           "above it",
                           line, layer, index, reference, layer - 1);

                return progress.layers[static_cast<std::size_t>(index)];
            }

            Config::Value make_attributes(std::vector<std::string> keys,
                                          std::vector<Config::Value> values)
            {
                SA_FATAL_IF(keys.size() != values.size(),
                            "a row's attributes have {} keys and {} values", keys.size(),
                            values.size());

                return Config::Value::object(std::move(keys), std::move(values));
            }

            std::int64_t append_row(TranslationProgress& progress, std::string_view type,
                                    Config::Value attributes,
                                    SushiRuntime::span<const std::int64_t> sources)
            {
                Config::LayerSpec row;
                row.type = std::string(type);
                row.attributes = std::move(attributes);
                row.from.reserve(sources.size());
                for (std::int64_t source : sources)
                    row.from.push_back(Config::SourceSpec{source, std::string()});

                std::vector<Config::LayerSpec>& rows = progress.model.spec.layers;
                rows.push_back(std::move(row));
                return static_cast<std::int64_t>(rows.size()) - 1;
            }

        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
