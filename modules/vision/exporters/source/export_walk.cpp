/**************************************************************************/
/* export_walk.cpp                                                        */
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
 * @file export_walk.cpp
 * @brief Defines the walk over a model's rows that every exporter shares.
 * @author Mustafa Garip
 */

#include "export_walk.hpp"

#include <cstddef>
#include <cstdint>
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
            namespace
            {
                /** @brief Names the one port a row writes its value to. */
                constexpr std::string_view K_PORT = "output";
            } // namespace

            bool is_whole_value(const std::string& port) noexcept
            {
                return port.empty() || port == K_PORT;
            }

            Result<bool> gather_sources(const Config::LayerSpec& spec, std::size_t row,
                                        std::string_view whole_value, const SourceTaker& take)
            {
                for (const Config::SourceSpec& source : spec.from)
                {
                    SA_FAIL_IF(!is_whole_value(source.port), StatusCode::Unsupported,
                               "it reads the port '{}' of row {}, and {}", source.port,
                               source.node, whole_value);
                    SA_FAIL_IF(source.node == Config::K_NAMED_INPUT, StatusCode::Unsupported,
                               "it reads the named input '{}'", source.input);
                    if (source.node != Config::K_MODEL_INPUT)
                    {
                        SA_FAIL_IF(source.node < 0 || static_cast<std::size_t>(source.node) >= row,
                                   StatusCode::InvalidDescription,
                                   "it reads row {}, which is not before it", source.node);
                    }
                    if (!take(source.node))
                        return false;
                }
                return true;
            }

            Status check_read_count(std::size_t row, const Config::LayerSpec& spec,
                                    std::size_t read, std::size_t count)
            {
                SA_FAIL_IF(read != count, StatusCode::InvalidDescription,
                           "row {} '{}' reads {} values, and the row reads {}", row, spec.type,
                           read, count);
                return Status();
            }

            std::string describe_refusal(std::size_t row, const Config::LayerSpec& spec,
                                         std::string_view reason)
            {
                return "row " + std::to_string(row) + " '" + spec.type + "': " +
                       std::string(reason);
            }

            Status write_rows(const std::vector<Config::LayerSpec>& layers,
                              const RowStep& write_row, const RowStep& after_row,
                              std::vector<std::string>& refusals)
            {
                for (std::size_t row = 0; row < layers.size(); ++row)
                {
                    const Status written = write_row(row);
                    if (!written.is_ok())
                    {
                        SA_FAIL_IF(written.code() != StatusCode::Unsupported, written.code(),
                                   "{}", written.message());
                        refusals.push_back(describe_refusal(row, layers[row], written.message()));
                        continue;
                    }
                    if (after_row)
                        SA_TRY(after_row(row));
                }
                return Status();
            }

            Status check_refusals(const std::vector<std::string>& refusals,
                                  std::string_view format)
            {
                std::string listed;
                for (const std::string& refusal : refusals)
                    listed.append(listed.empty() ? "" : "; ").append(refusal);
                SA_FAIL_IF(!refusals.empty(), StatusCode::Unsupported,
                           "the model cannot be written as {}: {} refused: {}", format,
                           refusals.size(), listed);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
