/**************************************************************************/
/* decode_file.cpp                                                        */
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
 * @file decode_file.cpp
 * @brief Implements the decoder of an image file a ByteSource serves.
 * @author Mustafa Garip
 */

#include "decode_file.hpp"

#include <cstdint>
#include <string>
#include <utility>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/vision/codecs/decode_image.hpp>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets::Detail
{
    Result<Imaging::Image> decode_file(const ByteSource& source, const std::filesystem::path& file,
                                       const Codecs::DecodeOptions& options,
                                       std::string_view reader) noexcept
    {
        return at_boundary(reader, [&]() -> Result<Imaging::Image>
        {
            SA_TRY_ASSIGN(const std::string bytes, source.read(file, reader));

            Result<Imaging::Image> decoded = Codecs::decode_image(
                SushiRuntime::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()),
                options);
            if (decoded.has_value())
                return decoded;

            return SA_ANNOTATE(decoded.status(), "'{}'", display_of(file));
        }, file);
    }
} // namespace SushiAI::Vision::Datasets::Detail
