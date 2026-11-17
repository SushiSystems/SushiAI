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
 * @brief Defines the host rules of the broadcast arithmetic and the sum back down to a side.
 * @author Mustafa Garip
 */

#include "rules.hpp"

#include <cstddef>
#include <vector>

namespace SushiAI
{
    namespace Graph
    {
        namespace Detail
        {
            namespace
            {
                /**
                 * @brief Returns the element of a vector of @p length that element @p at of a
                 *        tensor of @p total elements meets.
                 *
                 * Attribute flag 0 set is the trailing side: the vector repeats along the tensor.
                 * Clear is the leading side: each vector element covers one run of
                 * total / length consecutive elements.
                 */
                std::size_t vector_index(std::size_t at, std::size_t total, std::size_t length,
                                         bool trailing) noexcept
                {
                    return trailing ? at % length : at / (total / length);
                }

                /** @brief Writes @p op of each element of operand 0 and its vector element. */
                template<typename Op>
                Status broadcast_with(const HostNodeContext& context, Op op) noexcept
                {
                    const std::vector<double>& x = context.input(0);
                    const std::vector<double>& vector = context.input(1);
                    const bool trailing = context.node().attributes.flag(0);
                    std::vector<double>& y = context.output(0);
                    for (std::size_t i = 0; i < y.size(); ++i)
                        y[i] = op(x[i], vector[vector_index(i, x.size(), vector.size(), trailing)]);

                    return Status();
                }
            } // namespace

            Status host_broadcast_add(const HostNodeContext& context) noexcept
            {
                return broadcast_with(context, [](double a, double b) { return a + b; });
            }

            Status host_broadcast_sub(const HostNodeContext& context) noexcept
            {
                return broadcast_with(context, [](double a, double b) { return a - b; });
            }

            Status host_broadcast_mul(const HostNodeContext& context) noexcept
            {
                return broadcast_with(context, [](double a, double b) { return a * b; });
            }

            Status host_broadcast_div(const HostNodeContext& context) noexcept
            {
                return broadcast_with(context, [](double a, double b) { return a / b; });
            }

            Status host_sum_along(const HostNodeContext& context) noexcept
            {
                const std::vector<double>& x = context.input(0);
                const bool trailing = context.node().attributes.flag(0);
                std::vector<double>& sums = context.output(0);
                for (std::size_t i = 0; i < x.size(); ++i)
                    sums[vector_index(i, x.size(), sums.size(), trailing)] += x[i];

                return Status();
            }
        } // namespace Detail
    } // namespace Graph
} // namespace SushiAI
