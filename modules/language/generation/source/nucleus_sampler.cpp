/**************************************************************************/
/* nucleus_sampler.cpp                                                    */
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
 * @file nucleus_sampler.cpp
 * @brief Implements NucleusSampler.
 * @author Mustafa Garip
 */

#include <SushiAI/language/generation/nucleus_sampler.hpp>

#include <cstddef>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/status.hpp>
#include "ranking.hpp"
#include "weights.hpp"

namespace SushiAI::Language::Generation
{
    Result<std::unique_ptr<NucleusSampler>> NucleusSampler::make(double mass, double temperature)
    {
        SA_FAIL_IF(!(mass > 0.0) || !(mass <= 1.0), StatusCode::InvalidDescription,
                   "the '{}' sampler was given the mass {}; it is above 0 and at most 1",
                   K_NUCLEUS_SAMPLER, mass);
        SA_TRY(Detail::check_temperature(temperature, K_NUCLEUS_SAMPLER));

        return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            return std::unique_ptr<NucleusSampler>(new NucleusSampler(mass, temperature));
        });
    }

    NucleusSampler::NucleusSampler(double mass, double temperature) noexcept
        : mass_(mass), temperature_(temperature)
    {

    }

    std::string_view NucleusSampler::name() const noexcept
    {
        return K_NUCLEUS_SAMPLER;
    }

    Text::TokenID NucleusSampler::sample(LogitSpan logits, RandomGenerator& random) const
    {
        std::vector<double> weights;
        const double total = Detail::softmax_weights(logits, temperature_, weights);
        const std::vector<std::size_t> order = Detail::rank_descending(logits, logits.size());

        // The kept tokens are the shortest leading run whose weights reach mass_ of the sum.
        double kept_total = 0.0;
        std::vector<double> kept_weights(weights.size(), 0.0);
        for (const std::size_t index : order)
        {
            kept_weights[index] = weights[index];
            kept_total += weights[index];
            if (kept_total >= mass_ * total)
                break;
        }

        return static_cast<Text::TokenID>(Detail::draw_index(
            SushiRuntime::span<const double>(kept_weights.data(), kept_weights.size()),
            kept_total, random));
    }

} // namespace SushiAI::Language::Generation
