/**************************************************************************/
/* tensor_bytes.hpp                                                       */
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
 * @file tensor_bytes.hpp
 * @brief Declares the bytes of one foreign tensor, either a view into the file or an owned copy.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <cstddef>
#include <vector>
#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace Interchange
    {
        /**
         * @brief Holds the bytes of a tensor as the foreign file spells them.
         *
         * A tensor stored as one length-delimited field is a view into the file, so reading it
         * copies nothing; a tensor the file spells in typed or unpacked fields is assembled
         * once into an owned buffer in the layout of its element type, little-endian. Either
         * way bytes() is the tensor's elements in row-major order.
         */
        class TensorBytes final
        {
            public:
                /** @brief Constructs an empty tensor that views no file. */
                TensorBytes() = default;

                /**
                 * @brief Returns a tensor that views @p bytes.
                 * @param bytes A range inside the file; must outlive the tensor.
                 * @param offset Holds the offset of bytes[0] from the start of the file.
                 */
                [[nodiscard]] static TensorBytes viewing(SushiRuntime::span<const std::byte> bytes,
                                                         std::size_t offset) noexcept;

                /** @brief Returns a tensor that owns @p bytes. */
                [[nodiscard]] static TensorBytes owning(std::vector<std::byte> bytes) noexcept;

                /** @brief Returns the tensor's bytes. */
                [[nodiscard]] SushiRuntime::span<const std::byte> bytes() const noexcept;

                /** @brief Returns the number of bytes. */
                [[nodiscard]] std::size_t size() const noexcept;

                /** @brief Returns whether the bytes lie in the file and no copy was made. */
                [[nodiscard]] bool is_view() const noexcept;

                /** @brief Returns the offset of the first byte in the file, or 0 when owned. */
                [[nodiscard]] std::size_t offset() const noexcept;

            private:
                /** @brief Holds the viewed range; empty when the bytes are owned. */
                SushiRuntime::span<const std::byte> view_;

                /** @brief Holds the owned bytes; empty when the tensor is a view. */
                std::vector<std::byte> owned_;

                /** @brief Holds whether owned_ is the tensor's storage. */
                bool owns_ = false;

                /** @brief Holds the offset of the viewed range from the start of the file. */
                std::size_t offset_ = 0;
        };
    } // namespace Interchange
} // namespace SushiAI
