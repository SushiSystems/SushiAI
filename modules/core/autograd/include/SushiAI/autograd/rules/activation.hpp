/**************************************************************************/
/* activation.hpp                                                         */
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
        void activation_backward(const GradientTape& tape) noexcept;

        /** @brief Activation operations' backward rules. */
        inline constexpr std::array<RuleEntry, 7> K_ACTIVATION_RULES = {{
            {Ops::K_RELU.id,       &activation_backward},
            {Ops::K_GELU.id,       &activation_backward},
            {Ops::K_SIGMOID.id,    &activation_backward},
            {Ops::K_TANH.id,       &activation_backward},
            {Ops::K_LEAKY_RELU.id, &activation_backward},
            {Ops::K_SILU.id,       &activation_backward},
            {Ops::K_SOFTPLUS.id,   &activation_backward},
        }};
    }
}
