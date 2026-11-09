/**************************************************************************/
/* linalg.cpp                                                             */
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
 * @file linalg.cpp
 * @brief Defines the host rules of the matrix product, the bias row and the column sums.
 * @author Mustafa Garip
 */

#include "rules.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace Detail
        {
            Status host_matmul(const HostNodeContext& context) noexcept
            {
                const bool transpose_a = context.node().attributes.flag(0);
                const bool transpose_b = context.node().attributes.flag(1);
                const Shape& a_shape = context.input_shape(0);
                const Shape& c_shape = context.output_value(0).shape;
                const std::size_t rank = c_shape.rank();
                const int64_t batches = rank == 3 ? c_shape[0] : 1;
                const int64_t m = c_shape[rank - 2];
                const int64_t n = c_shape[rank - 1];
                const int64_t k = transpose_a ? a_shape[rank - 2] : a_shape[rank - 1];
                const std::vector<double>& a = context.input(0);
                const std::vector<double>& b = context.input(1);
                std::vector<double>& c = context.output(0);

                for (int64_t batch = 0; batch < batches; ++batch)
                {
                    const double* a_batch = a.data() + batch * m * k;
                    const double* b_batch = b.data() + batch * k * n;
                    double* c_batch = c.data() + batch * m * n;
                    for (int64_t i = 0; i < m; ++i)
                    {
                        for (int64_t j = 0; j < n; ++j)
                        {
                            double sum = 0.0;
                            for (int64_t p = 0; p < k; ++p)
                            {
                                const double left = transpose_a ? a_batch[p * m + i]
                                                                : a_batch[i * k + p];
                                const double right = transpose_b ? b_batch[j * k + p]
                                                                 : b_batch[p * n + j];
                                sum += left * right;
                            }

                            c_batch[i * n + j] = sum;
                        }
                    }
                }

                return Status();
            }

            Status host_bias_add(const HostNodeContext& context) noexcept
            {
                const std::vector<double>& x = context.input(0);
                const std::vector<double>& bias = context.input(1);
                std::vector<double>& y = context.output(0);
                for (std::size_t i = 0; i < y.size(); ++i)
                    y[i] = x[i] + bias[i % bias.size()];

                return Status();
            }

            Status host_sum_rows(const HostNodeContext& context) noexcept
            {
                const std::vector<double>& x = context.input(0);
                std::vector<double>& y = context.output(0);
                for (double& sum : y)
                    sum = 0.0;

                for (std::size_t i = 0; i < x.size(); ++i)
                    y[i % y.size()] += x[i];

                return Status();
            }
        } // namespace Detail
    } // namespace Graph
} // namespace SushiAI
