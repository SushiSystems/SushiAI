/**************************************************************************/
/* top_k_sampler.cpp                                                      */
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
 * @file top_k_sampler.cpp
 * @brief Implements TopKSampler.
 * @author Mustafa Garip
 */

#include <SushiAI/language/generation/top_k_sampler.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/failure/status.hpp>
#include "ranking.hpp"
#include "weights.hpp"

namespace SushiAI::Language::Generation
{
    Result<std::unique_ptr<TopKSampler>> TopKSampler::make(std::int64_t k, double temperature)
    {
        SA_FAIL_IF(k < 1, StatusCode::InvalidDescription,
                   "the '{}' sampler was asked to keep {} tokens; it keeps at least one",
                   K_TOP_K_SAMPLER, k);
        SA_TRY(Detail::check_temperature(temperature, K_TOP_K_SAMPLER));

        return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            return std::unique_ptr<TopKSampler>(new TopKSampler(k, temperature));
        });
    }

    TopKSampler::TopKSampler(std::int64_t k, double temperature) noexcept
        : k_(k), temperature_(temperature)
    {

    }

    std::string_view TopKSampler::name() const noexcept
    {
        return K_TOP_K_SAMPLER;
    }

    Text::TokenID TopKSampler::sample(LogitSpan logits, RandomGenerator& random) const
    {
        std::vector<double> weights;
        (void)Detail::softmax_weights(logits, temperature_, weights);

        const std::size_t kept =
            std::min(logits.size(), static_cast<std::size_t>(k_));
        const std::vector<std::size_t> order = Detail::rank_descending(logits, kept);

        double total = 0.0;
        std::vector<double> kept_weights(weights.size(), 0.0);
        for (std::size_t rank = 0; rank < kept; ++rank)
        {
            kept_weights[order[rank]] = weights[order[rank]];
            total += weights[order[rank]];
        }

        return static_cast<Text::TokenID>(Detail::draw_index(
            SushiRuntime::span<const double>(kept_weights.data(), kept_weights.size()), total,
            random));
    }

} // namespace SushiAI::Language::Generation
