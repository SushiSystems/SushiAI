/****************************************************************/
/* activation.hpp                                               */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file activation.hpp
 * @brief Declares backward differentiation rules for activation operations.
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
         * @brief Differentiates activation operations using metadata in Ops::ACTIVATIONS.
         * @param tape Traversal context providing forward nodes and gradient emitters.
         */
        void activation_backward(const GradientTape& tape);

        /** @brief Activation operations' backward rules. */
        inline constexpr std::array<RuleEntry, 5> ACTIVATION_RULES = {{
            {Ops::RELU.id,       &activation_backward},
            {Ops::GELU.id,       &activation_backward},
            {Ops::SIGMOID.id,    &activation_backward},
            {Ops::TANH.id,       &activation_backward},
            {Ops::LEAKY_RELU.id, &activation_backward},
        }};
    }
}
