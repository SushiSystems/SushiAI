/**************************************************************************/
/* tensor_bytes.cpp                                                       */
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
 * @file tensor_bytes.cpp
 * @brief Defines the bytes of one foreign tensor.
 * @author Mustafa Garip
 */

#include <SushiAI/interchange/tensor_bytes.hpp>

#include <cstddef>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace Interchange
    {
        TensorBytes TensorBytes::viewing(SushiRuntime::span<const std::byte> bytes,
                                         std::size_t offset) noexcept
        {
            TensorBytes out;
            out.view_ = bytes;
            out.offset_ = offset;
            return out;
        }

        TensorBytes TensorBytes::owning(std::vector<std::byte> bytes) noexcept
        {
            TensorBytes out;
            out.owned_ = std::move(bytes);
            out.owns_ = true;
            return out;
        }

        SushiRuntime::span<const std::byte> TensorBytes::bytes() const noexcept
        {
            if (owns_)
                return SushiRuntime::span<const std::byte>(owned_.data(), owned_.size());
            return view_;
        }

        std::size_t TensorBytes::size() const noexcept
        {
            return owns_ ? owned_.size() : view_.size();
        }

        bool TensorBytes::is_view() const noexcept
        {
            return !owns_;
        }

        std::size_t TensorBytes::offset() const noexcept
        {
            return owns_ ? 0 : offset_;
        }
    } // namespace Interchange
} // namespace SushiAI
