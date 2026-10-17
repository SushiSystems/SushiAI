/**************************************************************************/
/* top_k_sampler.hpp                                                      */
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
 * @file top_k_sampler.hpp
 * @brief Declares TopKSampler, which draws among the tokens of the k largest logits.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/language/generation/sampler.hpp>

namespace SushiAI::Language::Generation
{
    /** @brief Names the top-k sampler in a task section. */
    inline constexpr std::string_view K_TOP_K_SAMPLER = "top_k";

    /** @brief Draws from softmax(logits / temperature) over the k largest logits only. */
    class TopKSampler final : public Sampler
    {
        public:
            /**
             * @brief Builds the sampler.
             * @param k The tokens kept; a row of fewer logits keeps all of them.
             * @return The sampler, or InvalidDescription unless @p k is at least one and
             *         @p temperature is positive and finite.
             */
            [[nodiscard]] static Result<std::unique_ptr<TopKSampler>> make(std::int64_t k,
                                                                           double temperature);

            /** @brief Returns K_TOP_K_SAMPLER. */
            [[nodiscard]] std::string_view name() const noexcept override;

            /**
             * @brief Returns one token drawn among the kept ones, with one draw.
             *
             * Of equal logits at the edge, the lower indices are kept.
             */
            [[nodiscard]] Text::TokenID sample(LogitSpan logits,
                                               RandomGenerator& random) const override;

        private:
            /** @brief Takes the values make() checked. */
            TopKSampler(std::int64_t k, double temperature) noexcept;

            /// @brief Counts the tokens kept; at least one.
            std::int64_t k_;

            /// @brief Divides every logit before the softmax; positive.
            double temperature_;
    };

} // namespace SushiAI::Language::Generation
