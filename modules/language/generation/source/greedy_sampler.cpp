/**************************************************************************/
/* greedy_sampler.cpp                                                     */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file greedy_sampler.cpp
 * @brief Implements GreedySampler.
 * @author Mustafa Garip
 */

#include <SushiAI/language/generation/greedy_sampler.hpp>

#include <algorithm>
#include <SushiAI/core/check.hpp>

namespace SushiAI::Language::Generation
{
    std::string_view GreedySampler::name() const noexcept
    {
        return K_GREEDY_SAMPLER;
    }

    Text::TokenID GreedySampler::sample(LogitSpan logits, RandomGenerator&) const
    {
        SA_FATAL_IF(logits.empty(), "a sampler was given no logit; a position has one per token");

        // max_element returns the first of equal elements, which is the lowest index.
        return static_cast<Text::TokenID>(std::max_element(logits.begin(), logits.end())
                                          - logits.begin());
    }

} // namespace SushiAI::Language::Generation
