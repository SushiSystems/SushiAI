/**************************************************************************/
/* request.hpp                                                            */
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
 * @file request.hpp
 * @brief Defines the checks every generator makes of a prompt, a count and a row of logits.
 * @author Mustafa Garip
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/generation/sampler.hpp>
#include <SushiAI/language/text/token.hpp>

namespace SushiAI::Language::Generation::Detail
{
    /**
     * @brief Returns InvalidDescription unless @p count is positive, @p prompt holds an id
     *        and every id is in [0, vocabulary).
     * @param generator Names the generator in the message.
     */
    [[nodiscard]] inline Status check_request(SushiRuntime::span<const Text::TokenID> prompt,
                                              std::int64_t count, std::int64_t vocabulary,
                                              std::string_view generator)
    {
        SA_FAIL_IF(count < 1, StatusCode::InvalidDescription,
                   "{}: {} tokens were asked for; a run generates at least one", generator,
                   count);
        SA_FAIL_IF(prompt.empty(), StatusCode::InvalidDescription,
                   "{}: the prompt holds no id, so there is no position to sample at",
                   generator);

        for (std::size_t at = 0; at < prompt.size(); ++at)
        {
            SA_FAIL_IF(prompt[at] < 0 || prompt[at] >= vocabulary,
                       StatusCode::InvalidDescription,
                       "{}: id {} of the prompt is {}; the model scores ids 0 to {}", generator,
                       at, prompt[at], vocabulary - 1);
        }

        return Status();
    }

    /**
     * @brief Returns InvalidDescription when a score of @p row is not finite.
     * @param position The position of the context the row scores, for the message.
     */
    [[nodiscard]] inline Status check_finite(LogitSpan row, std::size_t position,
                                             std::string_view generator)
    {
        for (std::size_t token = 0; token < row.size(); ++token)
        {
            SA_FAIL_IF(!std::isfinite(row[token]), StatusCode::InvalidDescription,
                       "{}: the logit of token {} at position {} is not finite", generator,
                       token, position);
        }

        return Status();
    }

} // namespace SushiAI::Language::Generation::Detail
