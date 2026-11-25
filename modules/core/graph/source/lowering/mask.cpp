/**************************************************************************/
/* mask.cpp                                                               */
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
 * @file mask.cpp
 * @brief Implements the lowering row of CAUSAL_MASK.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/lowering/mask.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>
#include <SushiBLAS/broadcast/dispatch.hpp>
#include <SushiBLAS/functor_map/scalar_binding.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/allocation/host_copy.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering/context.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /** @brief Names the scratch slot of the keep vector. */
            constexpr int K_KEEP_SLOT = 0;

            /** @brief Returns the lowest finite fill a mask may take in @p dtype. */
            [[nodiscard]] constexpr double lowest_finite_fill(DType dtype) noexcept
            {
                constexpr double K_HALF_LOWEST = -65504.0;
                return dtype == DType::Half ? K_HALF_LOWEST
                                            : static_cast<double>(
                                                  std::numeric_limits<float>::lowest());
            }

            /**
             * @brief Writes one at each entry on or below the diagonal of a [length, length] table,
             *        and zero above it.
             * @tparam Element The host type whose bits are the tensor's element type.
             * @param keep The table, dense.
             * @param one The bits of one in the element type.
             * @return Success, or the failure of the allocation or of the copy.
             */
            template <typename Element>
            [[nodiscard]] Status write_triangle(SushiBLAS::Tensor& keep, std::int64_t length,
                                                Element one)
            {
                SA_TRY_ASSIGN(const std::vector<Element> entries,
                              ::SushiAI::Detail::guard_value(
                                  ::SushiAI::Detail::SeamPhase::Allocate, [length, one]()
                {
                    std::vector<Element> table(static_cast<std::size_t>(length * length));
                    for (std::int64_t row = 0; row < length; ++row)
                    {
                        for (std::int64_t column = 0; column < length; ++column)
                            table[static_cast<std::size_t>(row * length + column)] =
                                column <= row ? one : Element(0);
                    }
                    return table;
                }));

                return write_elements(keep, entries);
            }

            /**
             * @brief Returns the [T * T] keep vector of a causal mask in @p dtype, which the host
             *        fills once, at lowering, so no conversion task runs.
             * @param length T, the extent of each of the last two axes.
             * @return The vector, or the failure of an allocation.
             */
            [[nodiscard]] Result<SushiBLAS::Tensor&> keep_vector(const LoweringContext& context,
                                                                 std::int64_t length, DType dtype)
            {
                SA_TRY_ASSIGN(SushiBLAS::Tensor& keep,
                              context.scratch(K_KEEP_SLOT, Shape{length * length}, dtype));

                // 0x3C00 holds the bits of one in an IEEE half.
                constexpr std::uint16_t K_HALF_ONE = 0x3C00;
                switch (dtype)
                {
                    case DType::Half:
                        SA_TRY(write_triangle<std::uint16_t>(keep, length, K_HALF_ONE));
                        break;
                    case DType::Float32:
                        SA_TRY(write_triangle<float>(keep, length, 1.0f));
                        break;
                    case DType::Float64:
                        SA_TRY(write_triangle<double>(keep, length, 1.0));
                        break;
                    default:
                        SA_FATAL_IF(true, "CAUSAL_MASK needs a floating-point operand; got {}",
                                    static_cast<int>(dtype));
                }
                return keep;
            }
        } // namespace

        Status lower_causal_mask(const LoweringContext& context) noexcept
        {
            context.require_arity(1, 1);

            const Value& scores_value = context.input_value(0);
            const Shape& shape = scores_value.shape;
            const std::int32_t rank = shape.rank();
            SA_FATAL_IF(rank < 2 || shape[rank - 1] != shape[rank - 2] || shape[rank - 1] < 1,
                        "CAUSAL_MASK (node {}) needs [..., T, T] scores; got {}", context.id(),
                        shape.to_string());

            const std::int64_t length = shape[rank - 1];
            const double fill = std::max(context.node().attributes.get(0),
                                         lowest_finite_fill(scores_value.dtype));

            const SushiBLAS::Tensor& scores = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& masked, context.output(0));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& keep,
                          keep_vector(context, length, scores_value.dtype));

            // The select replaces a masked score even when it is infinite, which `x * 0` cannot.
            constexpr const char* K_TASK_NAME = Ops::K_CAUSAL_MASK.name.data();
            const std::array<SushiBLAS::BroadcastOperand, 1> vectors{
                {{&keep, SushiBLAS::BroadcastAxis::Trailing}}};
            SushiBLAS::Dispatch::execute_broadcast(
                context.recorder(), scores, vectors, masked, K_TASK_NAME,
                Ops::K_CAUSAL_MASK.id,
                SushiBLAS::Dispatch::bind_scalars(
                    [](auto x, auto keep_factor, auto replacement)
                    {
                        return keep_factor != decltype(keep_factor)(0) ? x : replacement;
                    },
                    SushiBLAS::Dispatch::ElementScalar(fill)),
                {fill});
            context.count(1);
            return Status();
        }
    } // namespace Graph
} // namespace SushiAI
