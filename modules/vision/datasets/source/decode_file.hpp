/**************************************************************************/
/* decode_file.hpp                                                        */
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
 * @file decode_file.hpp
 * @brief Declares the decoder of an image file a ByteSource serves.
 * @author Mustafa Garip
 */

#pragma once

#include <filesystem>
#include <string_view>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/codecs/decode_options.hpp>
#include <SushiAI/vision/datasets/byte_source.hpp>
#include <SushiAI/vision/imaging/image.hpp>

/** @brief Holds the units of the datasets module that no caller names. */
namespace SushiAI::Vision::Datasets::Detail
{
    /**
     * @brief Reads a file from a source and decodes it with Codecs::decode_image.
     * @param reader Names the calling loader in a message.
     * @return The image; what ByteSource::read returns for a file the source lacks or cannot
     *         read; otherwise the failure of the decoder. Every message opens with the path.
     */
    [[nodiscard]] Result<Imaging::Image> decode_file(const ByteSource& source,
                                                     const std::filesystem::path& file,
                                                     const Codecs::DecodeOptions& options,
                                                     std::string_view reader) noexcept;
} // namespace SushiAI::Vision::Datasets::Detail
