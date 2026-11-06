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
    Result<std::unique_ptr<TemperatureSampler>> TemperatureSampler::make(double temperature) noexcept
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
