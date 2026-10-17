/**************************************************************************/
/* temperature_sampler.hpp                                                */
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
 * @file temperature_sampler.hpp
 * @brief Declares TemperatureSampler, which draws from the softmax of scaled logits.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <string_view>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/language/generation/sampler.hpp>

namespace SushiAI::Language::Generation
{
    /** @brief Names the temperature sampler in a task section. */
    inline constexpr std::string_view K_TEMPERATURE_SAMPLER = "temperature";

    /** @brief Draws a token with the probability softmax(logits / temperature) gives it. */
    class TemperatureSampler final : public Sampler
    {
        public:
            /**
             * @brief Builds the sampler.
             * @return The sampler, or InvalidDescription unless @p temperature is positive
             *         and finite.
             */
            [[nodiscard]] static Result<std::unique_ptr<TemperatureSampler>> make(
                double temperature);

            /** @brief Returns K_TEMPERATURE_SAMPLER. */
            [[nodiscard]] std::string_view name() const noexcept override;

            /** @brief Returns one token drawn over every logit, with one draw. */
            [[nodiscard]] Text::TokenID sample(LogitSpan logits,
                                               RandomGenerator& random) const override;

        private:
            /** @brief Takes the temperature make() checked. */
            explicit TemperatureSampler(double temperature) noexcept;

            /// @brief Divides every logit before the softmax; positive.
            double temperature_;
    };

} // namespace SushiAI::Language::Generation
