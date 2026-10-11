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
