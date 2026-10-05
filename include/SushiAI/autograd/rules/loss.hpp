/****************************************************************/
/* loss.hpp                                                     */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
