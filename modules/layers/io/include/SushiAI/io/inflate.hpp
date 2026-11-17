/**************************************************************************/
/* inflate.hpp                                                            */
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
 * @file inflate.hpp
 * @brief Declares the decoder of a raw DEFLATE stream, RFC 1951.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/ULTRALYTICS.md
 */

#pragma once

#include <cstddef>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace IO
    {
        /**
         * @brief Decodes the raw DEFLATE stream @p compressed into @p output.
         * @param output Sized by the caller to the bytes the stream is known to hold.
         * @param origin Names the stream in a message.
         * @return The bytes written; InvalidFile naming @p origin and the bit offset where the
         *         stream is corrupt, ends inside a block or would write past @p output.
         */
        [[nodiscard]] Result<std::size_t> inflate(SushiRuntime::span<const std::byte> compressed,
                                                  SushiRuntime::span<std::byte> output,
                                                  std::string_view origin) noexcept;
    } // namespace IO
} // namespace SushiAI
