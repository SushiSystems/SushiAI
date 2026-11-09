/**************************************************************************/
/* elementwise.cpp                                                        */
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
 * @file elementwise.cpp
 * @brief Defines the host rules of the elementwise operations, the fills and the cast.
 * @author Mustafa Garip
 */

#include "rules.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include <SushiAI/dtype/dtype.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Holds sqrt(2 / pi), the slope of the GELU tanh argument. */
                constexpr double K_GELU_SLOPE = 0.7978845608028654;

                /** @brief Holds the cubic coefficient of the GELU tanh argument. */
                constexpr double K_GELU_CUBIC = 0.044715;

                /** @brief Writes @p op of each element of operand 0 into result 0. */
                template<typename Op>
                Status map_unary(const HostNodeContext& context, Op op) noexcept
                {
                    const std::vector<double>& x = context.input(0);
                    std::vector<double>& y = context.output(0);
                    std::transform(x.begin(), x.end(), y.begin(), op);
                    return Status();
                }

                /** @brief Writes @p op of each pair of operands 0 and 1 into result 0. */
                template<typename Op>
                Status map_binary(const HostNodeContext& context, Op op) noexcept
                {
                    const std::vector<double>& a = context.input(0);
                    const std::vector<double>& b = context.input(1);
                    std::vector<double>& y = context.output(0);
                    std::transform(a.begin(), a.end(), b.begin(), y.begin(), op);
                    return Status();
                }
            } // namespace

            Status host_zero(const HostNodeContext& context) noexcept
            {
                std::vector<double>& y = context.output(0);
                std::fill(y.begin(), y.end(), 0.0);
                return Status();
            }

            Status host_constant(const HostNodeContext& context) noexcept
            {
                std::vector<double>& y = context.output(0);
                std::fill(y.begin(), y.end(), context.node().attributes.get(0));
                return Status();
            }

            Status host_cast(const HostNodeContext& context) noexcept
            {
                const DType target = context.output_value(0).dtype;
                SA_FAIL_IF(target != DType::Float32 && target != DType::Float64,
                           StatusCode::Unsupported,
                           "HostEvaluator casts to float32 or float64 only; got {}",
                           dtype_name(target));

                if (target == DType::Float64)
                    return map_unary(context, [](double v) { return v; });

                return map_unary(context, [](double v)
                {
                    return static_cast<double>(static_cast<float>(v));
                });
            }

            Status host_add(const HostNodeContext& context) noexcept
            {
                return map_binary(context, [](double a, double b) { return a + b; });
            }

            Status host_sub(const HostNodeContext& context) noexcept
            {
                return map_binary(context, [](double a, double b) { return a - b; });
            }

            Status host_mul(const HostNodeContext& context) noexcept
            {
                return map_binary(context, [](double a, double b) { return a * b; });
            }

            Status host_scale(const HostNodeContext& context) noexcept
            {
                const double alpha = context.node().attributes.get(0);
                return map_unary(context, [alpha](double v) { return alpha * v; });
            }

            Status host_exp(const HostNodeContext& context) noexcept
            {
                return map_unary(context, [](double v) { return std::exp(v); });
            }

            Status host_less_equal(const HostNodeContext& context) noexcept
            {
                return map_binary(context, [](double a, double b) { return a <= b ? 1.0 : 0.0; });
            }

            Status host_select(const HostNodeContext& context) noexcept
            {
                const std::vector<double>& condition = context.input(0);
                const std::vector<double>& a = context.input(1);
                const std::vector<double>& b = context.input(2);
                std::vector<double>& y = context.output(0);
                for (std::size_t i = 0; i < y.size(); ++i)
                    y[i] = condition[i] != 0.0 ? a[i] : b[i];

                return Status();
            }

            Status host_gelu(const HostNodeContext& context) noexcept
            {
                return map_unary(context, [](double v)
                {
                    const double inner = K_GELU_SLOPE * (v + K_GELU_CUBIC * v * v * v);
                    return 0.5 * v * (1.0 + std::tanh(inner));
                });
            }
        } // namespace Detail
    } // namespace Graph
} // namespace SushiAI
