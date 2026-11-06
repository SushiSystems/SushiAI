/**************************************************************************/
/* tensor_shape.hpp                                                       */
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
 * @file tensor_shape.hpp
 * @brief Declares shape_of, which reads the extents of a SushiBLAS tensor as a Shape.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>

#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>

#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Eager
    {
        /** @brief Returns the extents of @p tensor as a Shape. */
        [[nodiscard]] inline Shape shape_of(const SushiBLAS::Tensor& tensor)
        {
            return Shape(SushiRuntime::span<const int64_t>(
                tensor.shape.data(), static_cast<std::size_t>(tensor.rank)));
        }

    } // namespace Eager
} // namespace SushiAI
