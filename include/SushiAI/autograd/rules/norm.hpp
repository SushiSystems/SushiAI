/**************************************************************************/
/* norm.hpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

// Normalisation operations' backward rules: BATCHNORM2D, CHANNEL_AFFINE.
//
// BATCHNORM_UPDATE_STATS is deliberately absent and carries no differentiable
// flag: CV-7 keeps a BatchNorm's running statistics out of the graph, so there
// is no gradient path through them to write a rule for.

#include <array>
#include <SushiAI/autograd/rule.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        /**
         * @brief BATCHNORM2D's gradient: input, scale and shift from one node.
         *
         * The batch mean and variance are functions of the input, so @c dx has
         * three terms rather than one. Treating the statistics as constants
         * gives a gradient that is right on average over the batch and wrong
         * per element - the model still trains, slightly worse, with nothing
         * to point at.
         *
         * @param tape The differentiation state for this node.
         */
        void batchnorm2d_backward(const GradientTape& tape);

        /**
         * @brief CHANNEL_AFFINE's gradient: the same broadcast for @c dx, and
         *        a per-channel sum for each parameter.
         * @param tape The differentiation state for this node.
         */
        void channel_affine_backward(const GradientTape& tape);

        /** @brief Normalisation operations' backward rules. */
        inline constexpr std::array<RuleEntry, 2> NORM_RULES = {{
            {Ops::BATCHNORM2D.id, &batchnorm2d_backward},
            {Ops::CHANNEL_AFFINE.id, &channel_affine_backward},
        }};
    }
}
