/**************************************************************************/
/* generator.hpp                                                          */
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
 * @file generator.hpp
 * @brief Declares Generator, the seam of the loops that continue a prompt token by token.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/generation/sampler.hpp>
#include <SushiAI/language/text/token.hpp>

namespace SushiAI::Language::Generation
{
    /** @brief Continues a prompt one token at a time over a model of a fixed window. */
    class Generator
    {
        public:
            /** @brief Releases the generator through its base. */
            virtual ~Generator() = default;

            /** @brief Returns T, the positions of the window. */
            [[nodiscard]] virtual std::int64_t window() const noexcept = 0;

            /** @brief Returns V, the tokens a position scores. */
            [[nodiscard]] virtual std::int64_t vocabulary() const noexcept = 0;

            /**
             * @brief Appends @p count tokens to @p generated, each sampled from the logits at
             *        the last position the context fills.
             * @param prompt At least one id, each in [0, V); the last T are the context.
             * @return Success, InvalidDescription for a prompt or a count that does not fit
             *         or a logit that is not finite, or the failure of a run of the step.
             */
            [[nodiscard]] virtual Status generate(SushiRuntime::span<const Text::TokenID> prompt,
                                                  std::int64_t count, const Sampler& sampler,
                                                  RandomGenerator& random,
                                                  std::vector<Text::TokenID>& generated) = 0;
    };

} // namespace SushiAI::Language::Generation
