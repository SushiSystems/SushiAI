/**************************************************************************/
/* row_keys.hpp                                                           */
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
 * @file row_keys.hpp
 * @brief Declares the readers of a row's keys that every exporter shares.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            /** @brief Holds a row whose keys an exporter reads, and how a diagnostic names it. */
            struct RowKeys
            {
                /** @brief Holds the row as the model file states it. */
                const Config::LayerSpec* spec = nullptr;

                /** @brief Holds the index a diagnostic names; empty names no index. */
                std::optional<std::size_t> row;

                /** @brief Holds the format whose importer a refused key names, such as "ONNX". */
                std::string_view importer;
            };

            /** @brief Returns Unsupported naming the first key of the row not in @p allowed. */
            [[nodiscard]] Status reject_unknown_keys(const RowKeys& keys,
                                                     SushiRuntime::span<const std::string_view>
                                                         allowed);

            /** @brief Returns the integer key @p key of the row, or @p fallback when absent. */
            [[nodiscard]] Result<std::int64_t> read_int_key(const RowKeys& keys,
                                                            std::string_view key,
                                                            std::int64_t fallback);

            /** @brief Returns the boolean key @p key of the row, or @p fallback when absent. */
            [[nodiscard]] Result<bool> read_bool_key(const RowKeys& keys, std::string_view key,
                                                     bool fallback);

            /** @brief Returns the number key @p key of the row, or @p fallback when absent. */
            [[nodiscard]] Result<double> read_real_key(const RowKeys& keys, std::string_view key,
                                                       double fallback);

            /** @brief Returns the text key @p key of the row, or @p fallback when absent. */
            [[nodiscard]] Result<std::string> read_text_key(const RowKeys& keys,
                                                            std::string_view key,
                                                            std::string_view fallback);

            /**
             * @brief Returns the key @p key, an integer or an [h, w] array, as {h, w}.
             * @return {@p fallback, @p fallback} when the key is absent.
             */
            [[nodiscard]] Result<std::vector<std::int64_t>> read_pair_key(const RowKeys& keys,
                                                                          std::string_view key,
                                                                          std::int64_t fallback);
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
