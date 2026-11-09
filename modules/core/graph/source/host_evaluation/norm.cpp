/**************************************************************************/
/* norm.cpp                                                               */
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
 * @file norm.cpp
 * @brief Defines the host rules of the row-wise log-softmax and layer normalisation.
 * @author Mustafa Garip
 */

#include "rules.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace Detail
        {
            Status host_log_softmax(const HostNodeContext& context) noexcept
            {
                const std::vector<double>& x = context.input(0);
                const std::size_t width = static_cast<std::size_t>(context.input_shape(0)[1]);
                std::vector<double>& y = context.output(0);

                for (std::size_t start = 0; start < x.size(); start += width)
                {
                    const auto row = x.begin() + static_cast<std::ptrdiff_t>(start);
                    const double largest =
                        *std::max_element(row, row + static_cast<std::ptrdiff_t>(width));
                    double sum = 0.0;
                    for (std::size_t c = 0; c < width; ++c)
                        sum += std::exp(x[start + c] - largest);

                    const double shift = largest + std::log(sum);
                    for (std::size_t c = 0; c < width; ++c)
                        y[start + c] = x[start + c] - shift;
                }

                return Status();
            }

            Status host_layer_norm(const HostNodeContext& context) noexcept
            {
                const std::vector<double>& x = context.input(0);
                const std::vector<double>& gamma = context.input(1);
                const std::vector<double>& beta = context.input(2);
                const double epsilon = context.node().attributes.get(0);
                const std::size_t width = gamma.size();
                std::vector<double>& y = context.output(0);
                std::vector<double>& means = context.output(1);
                std::vector<double>& inv_stds = context.output(2);

                for (std::size_t row = 0; row < means.size(); ++row)
                {
                    const std::size_t start = row * width;
                    double mean = 0.0;
                    for (std::size_t c = 0; c < width; ++c)
                        mean += x[start + c];

                    mean /= static_cast<double>(width);
                    double variance = 0.0;
                    for (std::size_t c = 0; c < width; ++c)
                        variance += (x[start + c] - mean) * (x[start + c] - mean);

                    variance /= static_cast<double>(width);
                    const double inv_std = 1.0 / std::sqrt(variance + epsilon);
                    for (std::size_t c = 0; c < width; ++c)
                        y[start + c] = (x[start + c] - mean) * inv_std * gamma[c] + beta[c];

                    means[row] = mean;
                    inv_stds[row] = inv_std;
                }

                return Status();
            }
        } // namespace Detail
    } // namespace Graph
} // namespace SushiAI
