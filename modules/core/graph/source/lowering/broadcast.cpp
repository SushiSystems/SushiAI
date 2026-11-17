/**************************************************************************/
/* broadcast.cpp                                                          */
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
 * @file broadcast.cpp
 * @brief Implements the lowering rows of the broadcast arithmetic and the sum along an axis.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/lowering/broadcast.hpp>

#include <cstdint>
#include <SushiBLAS/broadcast/broadcast.hpp>
#include <SushiBLAS/reductions/reductions.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/lowering/context.hpp>
#include "tensor_views.hpp"

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /** @brief Returns the SushiBLAS axis that a node's side names. */
            SushiBLAS::BroadcastAxis axis_of(const Node& node) noexcept
            {
                return broadcast_side(node.attributes) == BroadcastSide::Leading
                           ? SushiBLAS::BroadcastAxis::Leading
                           : SushiBLAS::BroadcastAxis::Trailing;
            }

            /**
             * @brief Records one kernel of BroadcastOps: result = kernel(x, vector) along the side.
             * @param context The node to lower; it has a tensor and a vector and one result.
             * @param kernel The member of BroadcastOps that records the kernel.
             * @return Ok when the kernel is recorded.
             */
            [[nodiscard]] Status record_broadcast(
                const LoweringContext& context,
                SushiBLAS::TaskHandle (SushiBLAS::BroadcastOps::*kernel)(
                    const SushiBLAS::Tensor&, const SushiBLAS::Tensor&, SushiBLAS::Tensor&,
                    SushiBLAS::BroadcastAxis))
            {
                context.require_arity(2, 1);
                const SushiBLAS::Tensor& x = context.input(0);
                const SushiBLAS::Tensor& vector = context.input(1);
                SA_TRY_ASSIGN(SushiBLAS::Tensor& result, context.output(0));

                SushiBLAS::BroadcastOps broadcast(context.recorder());
                (broadcast.*kernel)(x, vector, result, axis_of(context.node()));
                context.count(1);
                return Status();
            }
        } // namespace

        Status lower_broadcast_add(const LoweringContext& context) noexcept
        {
            return record_broadcast(context, &SushiBLAS::BroadcastOps::add);
        }

        Status lower_broadcast_sub(const LoweringContext& context) noexcept
        {
            return record_broadcast(context, &SushiBLAS::BroadcastOps::sub);
        }

        Status lower_broadcast_mul(const LoweringContext& context) noexcept
        {
            return record_broadcast(context, &SushiBLAS::BroadcastOps::mul);
        }

        Status lower_broadcast_div(const LoweringContext& context) noexcept
        {
            return record_broadcast(context, &SushiBLAS::BroadcastOps::div);
        }

        Status lower_sum_along(const LoweringContext& context) noexcept
        {
            context.require_arity(1, 1);

            const SushiBLAS::Tensor& x = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& sums, context.output(0));

            const std::int64_t kept = sums.num_elements;
            SA_FATAL_IF(kept < 1 || x.num_elements % kept != 0,
                        "SUM_ALONG (node {}) keeps {} of {} elements; the kept extent must divide "
                        "the tensor",
                        context.id(), kept, x.num_elements);
            const std::int64_t rest = x.num_elements / kept;

            // The kernel reads x as [outer, inner]: axis 1 keeps a prefix, axis 0 keeps a suffix.
            const bool leading = broadcast_side(context.node().attributes) == BroadcastSide::Leading;
            const std::int64_t grid[2] = {leading ? kept : rest, leading ? rest : kept};
            const int axis = leading ? 1 : 0;

            const SushiBLAS::Tensor matrix = Detail::view_as(
                context, x, SushiRuntime::span<const std::int64_t>(grid, 2));
            SushiBLAS::Tensor vector = Detail::view_as(
                context, sums, SushiRuntime::span<const std::int64_t>(&kept, 1));

            SushiBLAS::ReductionOps reductions(context.recorder());
            reductions.sum_along(matrix, axis, vector);
            context.count(1);
            return Status();
        }
    } // namespace Graph
} // namespace SushiAI
