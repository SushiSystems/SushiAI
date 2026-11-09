/**************************************************************************/
/* loss.cpp                                                               */
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
 * @file loss.cpp
 * @brief Defines the host rule of the mean negative log-likelihood.
 * @author Mustafa Garip
 */

#include "rules.hpp"

#include <cstddef>
#include <vector>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace Detail
        {
            Status host_nll_loss(const HostNodeContext& context) noexcept
            {
                const std::vector<double>& log_probs = context.input(0);
                const std::vector<double>& one_hot = context.input(1);
                const std::size_t rows = static_cast<std::size_t>(context.input_shape(0)[0]);
                double total = 0.0;
                for (std::size_t at = 0; at < log_probs.size(); ++at)
                    total += log_probs[at] * one_hot[at];

                context.output(0)[0] = -total / static_cast<double>(rows);
                return Status();
            }
        } // namespace Detail
    } // namespace Graph
} // namespace SushiAI
