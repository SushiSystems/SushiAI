/**************************************************************************/
/* perplexity.hpp                                                         */
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
 * @file perplexity.hpp
 * @brief Declares the metrics of the next-token objective: the mean loss and the perplexity.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <SushiAI/train/objective.hpp>

namespace SushiAI::Language::Training
{
    /**
     * @brief Returns exp(@p mean_loss), the perplexity of a mean loss in nats per token.
     * @return Infinity for a loss whose exponential a double cannot hold.
     */
    [[nodiscard]] double perplexity(double mean_loss) noexcept;

    /**
     * @brief Returns "loss" and "perplexity" for a step whose mean loss is @p mean_loss.
     * @param mean_loss The value of the next-token objective's loss, read after a step.
     */
    [[nodiscard]] std::array<Train::Metric, 2> next_token_metrics(double mean_loss) noexcept;

} // namespace SushiAI::Language::Training
