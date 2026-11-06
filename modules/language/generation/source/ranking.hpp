/**************************************************************************/
/* ranking.hpp                                                            */
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
 * @file ranking.hpp
 * @brief Declares the order of a row of logits the top-k and the nucleus samplers cut.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <vector>
#include <SushiAI/language/generation/sampler.hpp>

namespace SushiAI::Language::Generation::Detail
{
    /**
     * @brief Returns the indices of @p logits, largest logit first, the lower index first
     *        among equals.
     * @param leading How many leading places are put in order; at most logits.size(). The
     *        places behind them hold the other indices in no stated order.
     */
    [[nodiscard]] std::vector<std::size_t> rank_descending(LogitSpan logits,
                                                           std::size_t leading);

} // namespace SushiAI::Language::Generation::Detail
