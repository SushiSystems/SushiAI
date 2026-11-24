/**************************************************************************/
/* darknet_files.cpp                                                      */
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
 * @file darknet_files.cpp
 * @brief Defines the writers of the .cfg text and the .weights bytes.
 * @author Mustafa Garip
 */

#include "darknet_files.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "darknet_export_progress.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Holds the minor version whose header carries a 64-bit seen count. */
                constexpr std::int32_t K_MINOR_VERSION = 2;

                /** @brief Appends the bytes of @p count values at @p data to @p bytes. */
                void append_bytes(std::vector<std::byte>& bytes, const void* data,
                                  std::size_t count)
                {
                    const std::size_t at = bytes.size();
                    bytes.resize(at + count);
                    if (count != 0)
                        std::memcpy(bytes.data() + at, data, count);
                }

                /** @brief Appends the float32 values of @p values to @p bytes. */
                void append_floats(std::vector<std::byte>& bytes, const std::vector<float>& values)
                {
                    append_bytes(bytes, values.data(), values.size() * sizeof(float));
                }

                /** @brief Appends `[type]` and its lines to @p text. */
                void append_section(std::string& text, const std::string& type,
                                    const std::vector<DarknetOptionText>& options)
                {
                    text.append("[").append(type).append("]\n");
                    for (const DarknetOptionText& option : options)
                        text.append(option.key).append("=").append(option.value).append("\n");
                }
            } // namespace

            std::string write_darknet_cfg(const std::vector<DarknetOptionText>& network,
                                          const std::vector<DarknetSectionText>& sections)
            {
                std::string text;
                append_section(text, "net", network);
                for (const DarknetSectionText& section : sections)
                {
                    text.append("\n");
                    append_section(text, section.type, section.options);
                }
                return text;
            }

            std::vector<std::byte> write_darknet_weights(
                const std::vector<DarknetConvolutionFloats>& convolutions)
            {
                std::vector<std::byte> bytes;
                const std::int32_t version[] = {0, K_MINOR_VERSION, 0};
                const std::uint64_t seen = 0;
                append_bytes(bytes, version, sizeof(version));
                append_bytes(bytes, &seen, sizeof(seen));
                for (const DarknetConvolutionFloats& convolution : convolutions)
                {
                    append_floats(bytes, convolution.biases);
                    append_floats(bytes, convolution.scales);
                    append_floats(bytes, convolution.means);
                    append_floats(bytes, convolution.variances);
                    append_floats(bytes, convolution.weights);
                }
                return bytes;
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
