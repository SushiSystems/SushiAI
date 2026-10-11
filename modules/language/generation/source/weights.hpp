/**************************************************************************/
/* weights.hpp                                                            */
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
 * @file weights.hpp
 * @brief Declares what the drawing samplers share: a check, the weights and the draw.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/language/generation/sampler.hpp>

namespace SushiAI::Language::Generation::Detail
{
    /**
     * @brief Returns InvalidDescription unless @p temperature is positive and finite.
     * @param sampler The name of the sampler, for the message.
     */
    [[nodiscard]] Status check_temperature(double temperature, std::string_view sampler);

    /**
     * @brief Writes exp((logit - largest) / temperature) for each logit into @p weights.
     * @param logits Finite and not empty; empty is fatal.
     * @param temperature Positive and finite.
     * @return The sum of the weights, which is at least one.
     */
    double softmax_weights(LogitSpan logits, double temperature, std::vector<double>& weights);

    /** @brief Returns a value in [0, 1) from one draw of @p random. */
    [[nodiscard]] double draw_unit(RandomGenerator& random);

    /**
     * @brief Returns an index of @p weights with the probability weight / total.
     * @param weights Not negative, and not all zero.
     * @param total The sum of @p weights.
     * @return The first index whose running sum exceeds one draw scaled by @p total, or the
     *         last index of a positive weight when rounding leaves none.
     */
    [[nodiscard]] std::size_t draw_index(SushiRuntime::span<const double> weights, double total,
                                         RandomGenerator& random);

} // namespace SushiAI::Language::Generation::Detail
