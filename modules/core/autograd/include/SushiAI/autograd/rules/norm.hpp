/**************************************************************************/
/* norm.hpp                                                               */
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
 * @file norm.hpp
 * @brief Declares backward differentiation rules for normalization operations.
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
         * @brief Differentiates 2D batch normalization across input, scale, and shift.
         * @param tape Traversal context providing forward nodes and gradient emitters.
         */
        void batchnorm2d_backward(const GradientTape& tape);

        /**
         * @brief Differentiates channel affine transformation with per-channel parameter sums.
         * @param tape Traversal context providing forward nodes and gradient emitters.
         */
        void channel_affine_backward(const GradientTape& tape);

        /**
         * @brief Differentiates layer normalization across input, gain, and bias.
         * @param tape Traversal context providing forward nodes and gradient emitters.
         */
        void layer_norm_backward(const GradientTape& tape);

        /** @brief Holds backward rule entries for normalization operations. */
        inline constexpr std::array<RuleEntry, 3> K_NORM_RULES = {{
            {Ops::K_BATCHNORM2D.id, &batchnorm2d_backward},
            {Ops::K_CHANNEL_AFFINE.id, &channel_affine_backward},
            {Ops::K_LAYER_NORM.id, &layer_norm_backward},
        }};
    }
}
