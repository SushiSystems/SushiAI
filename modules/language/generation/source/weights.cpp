/**************************************************************************/
/* weights.cpp                                                            */
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
 * @file weights.cpp
 * @brief Implements the check, the softmax weights and the draw of the drawing samplers.
 * @author Mustafa Garip
 */

#include "weights.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <SushiAI/failure/check.hpp>

namespace SushiAI::Language::Generation::Detail
{
    namespace
    {
        /** @brief Holds the bits of a draw a double's fraction keeps. */
        constexpr int K_FRACTION_BITS = 53;

        /** @brief Holds 2^-53, the step between two values draw_unit() returns. */
        constexpr double K_UNIT_STEP = 1.0 / 9007199254740992.0;
    } // namespace

    Status check_temperature(double temperature, std::string_view sampler)
    {
        SA_FAIL_IF(!(temperature > 0.0) || !std::isfinite(temperature),
                   StatusCode::InvalidDescription,
                   "the '{}' sampler was given the temperature {}; it is positive and finite",
                   sampler, temperature);

        return Status();
    }

    double softmax_weights(LogitSpan logits, double temperature, std::vector<double>& weights)
    {
        SA_FATAL_IF(logits.empty(), "a sampler was given no logit; a position has one per token");

        const double largest =
            static_cast<double>(*std::max_element(logits.begin(), logits.end()));
        weights.resize(logits.size());

        double total = 0.0;
        for (std::size_t index = 0; index < logits.size(); ++index)
        {
            weights[index] = std::exp((static_cast<double>(logits[index]) - largest)
                                      / temperature);
            total += weights[index];
        }

        return total;
    }

    double draw_unit(RandomGenerator& random)
    {
        const std::uint64_t drawn = random();
        return static_cast<double>(drawn >> (64 - K_FRACTION_BITS)) * K_UNIT_STEP;
    }

    std::size_t draw_index(SushiRuntime::span<const double> weights, double total,
                           RandomGenerator& random)
    {
        const double threshold = draw_unit(random) * total;

        double running = 0.0;
        std::size_t last_positive = 0;
        for (std::size_t index = 0; index < weights.size(); ++index)
        {
            if (weights[index] <= 0.0)
                continue;

            running += weights[index];
            last_positive = index;
            if (running > threshold)
                return index;
        }

        return last_positive;
    }

} // namespace SushiAI::Language::Generation::Detail
