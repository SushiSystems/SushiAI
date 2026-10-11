/**************************************************************************/
/* loss.hpp                                                               */
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
 * @file loss.hpp
 * @brief Declares the lowering rows of the loss operations and their table.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <SushiAI/graph/lowering/rule.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /** @brief Records the row-wise log-softmax of a rank-2 operand. */
        [[nodiscard]] Status lower_log_softmax(const LoweringContext& context);

        /** @brief Records dy - exp(log_probabilities) * rowsum(dy) over [dy, log_probabilities]. */
        [[nodiscard]] Status lower_log_softmax_backward(const LoweringContext& context);

        /** @brief Records the mean negative log-likelihood of log-probabilities and targets. */
        [[nodiscard]] Status lower_nll_loss(const LoweringContext& context);

        /** @brief Pairs each loss operation with the row that lowers it. */
        inline constexpr std::array<LoweringEntry, 3> K_LOSS_LOWERINGS = {{
            {Ops::K_LOG_SOFTMAX.id, &lower_log_softmax},
            {Ops::K_LOG_SOFTMAX_BACKWARD.id, &lower_log_softmax_backward},
            {Ops::K_NLL_LOSS.id, &lower_nll_loss},
        }};
    } // namespace Graph
} // namespace SushiAI
