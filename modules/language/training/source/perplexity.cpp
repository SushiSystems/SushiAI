/**************************************************************************/
/* perplexity.cpp                                                         */
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
 * @file perplexity.cpp
 * @brief Implements the metrics of the next-token objective.
 * @author Mustafa Garip
 */

#include <SushiAI/language/training/perplexity.hpp>

#include <cmath>

namespace SushiAI::Language::Training
{
    double perplexity(double mean_loss) noexcept
    {
        return std::exp(mean_loss);
    }

    std::array<Train::Metric, 2> next_token_metrics(double mean_loss) noexcept
    {
        return {Train::Metric{"loss", mean_loss, false},
                Train::Metric{"perplexity", perplexity(mean_loss), false}};
    }

} // namespace SushiAI::Language::Training
