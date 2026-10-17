/**************************************************************************/
/* temperature_sampler.cpp                                                */
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
 * @file temperature_sampler.cpp
 * @brief Implements TemperatureSampler.
 * @author Mustafa Garip
 */

#include <SushiAI/language/generation/temperature_sampler.hpp>

#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/failure/status.hpp>
#include "weights.hpp"

namespace SushiAI::Language::Generation
{
    Result<std::unique_ptr<TemperatureSampler>> TemperatureSampler::make(double temperature)
    {
        SA_TRY(Detail::check_temperature(temperature, K_TEMPERATURE_SAMPLER));

        return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            return std::unique_ptr<TemperatureSampler>(new TemperatureSampler(temperature));
        });
    }

    TemperatureSampler::TemperatureSampler(double temperature) noexcept
        : temperature_(temperature)
    {

    }

    std::string_view TemperatureSampler::name() const noexcept
    {
        return K_TEMPERATURE_SAMPLER;
    }

    Text::TokenID TemperatureSampler::sample(LogitSpan logits, RandomGenerator& random) const
    {
        std::vector<double> weights;
        const double total = Detail::softmax_weights(logits, temperature_, weights);

        return static_cast<Text::TokenID>(Detail::draw_index(
            SushiRuntime::span<const double>(weights.data(), weights.size()), total, random));
    }

} // namespace SushiAI::Language::Generation
