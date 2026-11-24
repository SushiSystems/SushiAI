/**************************************************************************/
/* export_walk.hpp                                                        */
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
 * @file export_walk.hpp
 * @brief Declares the walk over a model's rows that every exporter shares.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            /** @brief Takes the row @p node reads, K_MODEL_INPUT for the input; false if absent. */
            using SourceTaker = std::function<bool(std::int64_t node)>;

            /** @brief Writes or checks the row @p row; Unsupported refuses that row alone. */
            using RowStep = std::function<Status(std::size_t row)>;

            /** @brief Returns whether a source or an output names the row's whole value. */
            [[nodiscard]] bool is_whole_value(const std::string& port) noexcept;

            /**
             * @brief Hands each source of the row @p row to @p take, in the order of its "from".
             * @param whole_value What the format's value is, as a refused port names it.
             * @return false at the first source @p take finds no value for; Unsupported for a
             *         port or a named input; InvalidDescription for a row not before @p row.
             */
            [[nodiscard]] Result<bool> gather_sources(const Config::LayerSpec& spec,
                                                      std::size_t row,
                                                      std::string_view whole_value,
                                                      const SourceTaker& take);

            /** @brief Returns InvalidDescription unless @p read, the operand count, is @p count. */
            [[nodiscard]] Status check_read_count(std::size_t row, const Config::LayerSpec& spec,
                                                  std::size_t read, std::size_t count);

            /** @brief Returns `row <i> '<type>': <reason>`. */
            [[nodiscard]] std::string describe_refusal(std::size_t row,
                                                       const Config::LayerSpec& spec,
                                                       std::string_view reason);

            /**
             * @brief Runs @p write_row on each row, and @p after_row, when set, on each row it
             *        wrote; an Unsupported row goes to @p refusals.
             * @return A failure other than Unsupported at once; else success.
             */
            [[nodiscard]] Status write_rows(const std::vector<Config::LayerSpec>& layers,
                                            const RowStep& write_row, const RowStep& after_row,
                                            std::vector<std::string>& refusals);

            /** @brief Returns Unsupported listing @p refusals, when any, for the @p format. */
            [[nodiscard]] Status check_refusals(const std::vector<std::string>& refusals,
                                                std::string_view format);
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
