/**************************************************************************/
/* tensor_views.hpp                                                       */
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
 * @file tensor_views.hpp
 * @brief Declares the reshaped view of a contiguous operand that the lowering rows read.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiRuntime/core/span.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/graph/lowering/context.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace Detail
        {
            /**
             * @brief Returns @p tensor seen with the extents @p dims, over the same storage.
             * @param context The node being lowered; its name and id head the failure message.
             * @param tensor The operand to view; a strided one is fatal.
             * @param dims The extents of the view; their product must equal the operand's
             *        element count, or the call is fatal.
             * @return A tensor sharing @p tensor's storage, offset, element type and layout.
             */
            [[nodiscard]] inline SushiBLAS::Tensor view_as(
                const LoweringContext& context, const SushiBLAS::Tensor& tensor,
                SushiRuntime::span<const std::int64_t> dims)
            {
                SA_FATAL_IF(!tensor.is_contiguous(),
                            "{} (node {}) needs contiguous operands; a strided view cannot be "
                            "read with other extents",
                            context.name(), context.id());

                std::int64_t elements = 1;
                for (const std::int64_t extent : dims)
                    elements *= extent;

                SA_FATAL_IF(elements != tensor.num_elements,
                            "{} (node {}) views {} elements as {} elements", context.name(),
                            context.id(), tensor.num_elements, elements);

                return SushiBLAS::Tensor(tensor.storage, dims, tensor.dtype,
                                         tensor.storage_offset, tensor.layout);
            }
        } // namespace Detail
    } // namespace Graph
} // namespace SushiAI
