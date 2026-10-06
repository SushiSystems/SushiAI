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
 * @brief Declares backward differentiation rules for loss and softmax operations.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <SushiAI/autograd/rule.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        /**
         * @brief Differentiates standalone log-softmax operations.
         * @param tape Traversal context providing forward nodes and gradient emitters.
         */
        void log_softmax_backward(const GradientTape& tape);

        /**
         * @brief Differentiates negative log-likelihood loss, fusing adjacent log-softmax if present.
         * @param tape Traversal context providing forward nodes and gradient emitters.
         */
        void nll_loss_backward(const GradientTape& tape);

        /** @brief Holds backward rule entries for loss operations. */
        inline constexpr std::array<RuleEntry, 1> LOSS_RULES = {{
            {Ops::NLL_LOSS.id, &nll_loss_backward},
        }};
    }
}
