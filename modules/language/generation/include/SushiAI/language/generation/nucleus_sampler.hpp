/**************************************************************************/
/* nucleus_sampler.hpp                                                    */
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
 * @file nucleus_sampler.hpp
 * @brief Declares NucleusSampler, which draws among the most probable tokens of a stated mass.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <string_view>
#include <SushiAI/core/result.hpp>
#include <SushiAI/language/generation/sampler.hpp>

namespace SushiAI::Language::Generation
{
    /** @brief Names the nucleus sampler in a task section. */
    inline constexpr std::string_view K_NUCLEUS_SAMPLER = "nucleus";

    /**
     * @brief Draws from softmax(logits / temperature) over the fewest most probable tokens
     *        whose probabilities sum to at least a stated mass.
     */
    class NucleusSampler final : public Sampler
    {
        public:
            /**
             * @brief Builds the sampler.
             * @param mass The probability the kept tokens reach, called top-p; in (0, 1].
             * @return The sampler, or InvalidDescription unless @p mass is in (0, 1] and
             *         @p temperature is positive and finite.
             */
            [[nodiscard]] static Result<std::unique_ptr<NucleusSampler>> make(
                double mass, double temperature);

            /** @brief Returns K_NUCLEUS_SAMPLER. */
            [[nodiscard]] std::string_view name() const noexcept override;

            /**
             * @brief Returns one token drawn among the kept ones, with one draw.
             *
             * Tokens are taken most probable first, the lower index first among equals.
             */
            [[nodiscard]] Text::TokenID sample(LogitSpan logits,
                                               RandomGenerator& random) const override;

        private:
            /** @brief Takes the values make() checked. */
            NucleusSampler(double mass, double temperature) noexcept;

            /// @brief Holds the probability the kept tokens reach; in (0, 1].
            double mass_;

            /// @brief Divides every logit before the softmax; positive.
            double temperature_;
    };

} // namespace SushiAI::Language::Generation
