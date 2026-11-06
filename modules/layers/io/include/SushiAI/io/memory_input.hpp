/**************************************************************************/
/* memory_input.hpp                                                       */
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
 * @file memory_input.hpp
 * @brief Declares the input stream that reads bytes already in memory.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <streambuf>
#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace Detail
        {
            /** @brief Serves bytes another object owns to a stream, copying none of them. */
            class MemoryBuffer : public std::streambuf
            {
                public:
                    /** @brief Constructs a read-only buffer over bytes that must outlive it. */
                    explicit MemoryBuffer(SushiRuntime::span<const std::uint8_t> bytes)
                    {
                        // The get area is only read: std::streambuf has no const form of setg.
                        char* first =
                            const_cast<char*>(reinterpret_cast<const char*>(bytes.data()));
                        setg(first, first, first + bytes.size());
                    }
            };
        } // namespace Detail

        /**
         * @class MemoryInput
         * @brief Reads forward through bytes that are already in memory.
         *
         * It takes the place of a file stream wherever a reader takes a std::istream. The
         * stream does not seek, and the bytes must outlive it.
         */
        class MemoryInput final : private Detail::MemoryBuffer, public std::istream
        {
            public:
                /** @brief Constructs a stream positioned at the first byte of @p bytes. */
                explicit MemoryInput(SushiRuntime::span<const std::uint8_t> bytes)
                    : Detail::MemoryBuffer(bytes), std::istream(this)
                {
                }
        };
    } // namespace IO
} // namespace SushiAI
